import ast
import json
import math
import random
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

VALID_SENTIMENTS = ("Positive", "Negative", "Neutral")
TARGET_MAP = {"Buy": 0, "Neutral": 1, "Sell": 2}
MODEL_LABELS = ["Buy", "Neutral", "Sell"]
DEFAULT_ARTIFACT_CONFIG = {
    "model_name": "hierarchical_market_sentiment_rnn",
    "lookback_days": 14,
    "max_posts_per_day": 128,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 3e-4,
    "weight_decay": 2e-4,
    "early_stopping_patience": 8,
    "post_hidden_dim": 48,
    "sequence_hidden_dim": 128,
    "recurrent_layers": 2,
    "recurrent_dropout": 0.2,
    "classifier_dropout": 0.25,
    "use_balanced_sampler": False,
    "random_seed": 42,
}
MARKET_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "trading_volume_usdt",
    "sma_14",
    "ema_14",
    "rsi_14",
    "atr_14",
    "daily_return",
    "prev_day_return",
    "rolling_return_mean_3d",
    "rolling_volatility_7d",
    "intraday_range",
]
POST_NUMERIC_COLUMNS = [
    "sentiment_confidence",
    "sentiment_polarity",
    "confidence_weighted_polarity",
    "subjectivity_score",
    "token_count",
]
CLIPPED_MARKET_COLUMNS = [
    "daily_return_clipped",
    "prev_day_return_clipped",
    "rolling_return_mean_3d_clipped",
    "rolling_volatility_7d_clipped",
    "intraday_range_log1p",
    "trading_volume_usdt_log1p",
    "sma_14_gap",
    "ema_14_gap",
    "rsi_14_scaled",
    "atr_14_log1p",
    "days_since_previous_record_log1p",
]
CALENDAR_FEATURE_COLUMNS = ["day_of_week", "month_sin", "month_cos"]
AGGREGATED_SENTIMENT_FEATURE_COLUMNS = [
    "post_count_log1p",
    "mean_sentiment_confidence",
    "std_sentiment_confidence",
    "mean_sentiment_polarity",
    "std_sentiment_polarity",
    "mean_confidence_weighted_polarity",
    "std_confidence_weighted_polarity",
    "mean_subjectivity_score",
    "std_subjectivity_score",
    "mean_token_count",
    "std_token_count",
    "price_positive_ratio",
    "price_negative_ratio",
    "bert_low_star_ratio",
    "bert_high_star_ratio",
]
MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS = (
    CLIPPED_MARKET_COLUMNS + CALENDAR_FEATURE_COLUMNS + AGGREGATED_SENTIMENT_FEATURE_COLUMNS
)


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def convert_sentiment_label(label) -> int:
    if isinstance(label, str) and "star" in label:
        try:
            return int(label.split()[0])
        except (TypeError, ValueError):
            return 3
    return 3


def derive_sentiment_polarity(label) -> float:
    star_value = convert_sentiment_label(label)
    return float((star_value - 3) / 2)


def _normalize_subject_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _sorted_copy(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _safe_numeric(value, fallback: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(fallback)
    except TypeError:
        pass
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _standard_deviation_or_zero(series: pd.Series) -> float:
    value = float(series.std(ddof=0)) if len(series) else 0.0
    return 0.0 if np.isnan(value) else value


def _normalize_sentiment_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                parsed = [stripped]
        else:
            parsed = [stripped]
    elif isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        parsed = [value]

    cleaned = []
    for item in parsed:
        item_str = str(item).strip().title()
        if item_str in VALID_SENTIMENTS:
            cleaned.append(item_str)
    return cleaned


def extract_sentiments(value) -> list[str]:
    return _normalize_sentiment_values(value) or ["Neutral"]


def normalize_price_sentiment(value) -> str:
    cleaned = [sentiment for sentiment in extract_sentiments(value) if sentiment in VALID_SENTIMENTS]
    return str(cleaned or ["Neutral"])


def _extract_price_sentiment_primary(value) -> str:
    sentiments = extract_sentiments(value)
    counts = Counter(sentiments)
    return max(VALID_SENTIMENTS, key=lambda item: counts.get(item, 0))


def _build_post_numeric_row(post: dict) -> dict[str, float]:
    sentiment_confidence = _safe_numeric(post.get("sentiment_confidence", post.get("sentiment_score")))
    sentiment_polarity = _safe_numeric(
        post.get("sentiment_polarity", derive_sentiment_polarity(post.get("sentiment_label")))
    )
    confidence_weighted_polarity = _safe_numeric(
        post.get("confidence_weighted_polarity", sentiment_confidence * sentiment_polarity)
    )
    return {
        "sentiment_confidence": sentiment_confidence,
        "sentiment_polarity": sentiment_polarity,
        "confidence_weighted_polarity": confidence_weighted_polarity,
        "subjectivity_score": _safe_numeric(post.get("subjectivity_score")),
        "token_count": _safe_numeric(post.get("token_count")),
    }


def _build_category_mapping(values: Iterable, *, key=None) -> dict[str, int]:
    normalized_values = []
    for value in values:
        text_value = str(value).strip()
        if text_value:
            normalized_values.append(text_value)
    ordered = sorted(set(normalized_values), key=key or (lambda item: item))
    return {value: index for index, value in enumerate(ordered)}


def _fit_post_feature_state(records: list[dict]) -> dict:
    train_records = [record for record in records if record.get("split") == "train"]
    train_posts = [post for record in train_records for post in record.get("posts", [])]
    if not train_posts:
        raise ValueError("No training posts are available for fitting the post feature encoder.")

    numeric_frame = pd.DataFrame([_build_post_numeric_row(post) for post in train_posts]).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(numeric_frame[POST_NUMERIC_COLUMNS])

    source_map = _build_category_mapping((post.get("source", "unknown") for post in train_posts))
    sentiment_label_map = _build_category_mapping(
        (post.get("sentiment_label", "unknown") for post in train_posts),
        key=lambda value: (convert_sentiment_label(value), value),
    )
    price_sentiment_map = _build_category_mapping(
        (_extract_price_sentiment_primary(post.get("price_sentiment")) for post in train_posts)
    )
    return {
        "numeric_scaler": scaler,
        "source_map": source_map,
        "sentiment_label_map": sentiment_label_map,
        "price_sentiment_map": price_sentiment_map,
    }


def _one_hot(index: Optional[int], size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        vector[index] = 1.0
    return vector


def encode_post_feature_vector(post: dict, state: dict) -> np.ndarray:
    numeric_scaler = state["numeric_scaler"]
    numeric_frame = pd.DataFrame([_build_post_numeric_row(post)], columns=POST_NUMERIC_COLUMNS)
    scaled_numeric = numeric_scaler.transform(numeric_frame)[0].astype(np.float32)

    source_value = str(post.get("source", "")).strip()
    sentiment_label_value = str(post.get("sentiment_label", "")).strip()
    price_sentiment_value = _extract_price_sentiment_primary(post.get("price_sentiment"))

    source_vector = _one_hot(state["source_map"].get(source_value), len(state["source_map"]))
    sentiment_label_vector = _one_hot(
        state["sentiment_label_map"].get(sentiment_label_value),
        len(state["sentiment_label_map"]),
    )
    price_sentiment_vector = _one_hot(
        state["price_sentiment_map"].get(price_sentiment_value),
        len(state["price_sentiment_map"]),
    )
    return np.concatenate(
        [scaled_numeric, source_vector, sentiment_label_vector, price_sentiment_vector],
        axis=0,
    ).astype(np.float32)


def _deterministic_even_sample(sequence: list, limit: int) -> list:
    if limit is None or limit <= 0 or len(sequence) <= limit:
        return list(sequence)
    indexes = np.linspace(0, len(sequence) - 1, num=limit, dtype=int)
    return [sequence[index] for index in indexes.tolist()]


def sample_posts_for_model(posts: list[dict], max_posts: Optional[int]) -> list[dict]:
    if max_posts is None or len(posts) <= max_posts:
        return list(posts)

    posts_by_source = {}
    for post in posts:
        source = str(post.get("source", "")).strip().lower()
        posts_by_source.setdefault(source, []).append(post)

    source_names = [source for source in ["news", "reddit", "twitter"] if source in posts_by_source]
    for source in posts_by_source:
        if source not in source_names:
            source_names.append(source)

    sampled_posts = []
    remaining = int(max_posts)
    remaining_sources = len(source_names)
    for source in source_names:
        source_posts = posts_by_source[source]
        quota = max(1, remaining // max(1, remaining_sources))
        selected = _deterministic_even_sample(source_posts, min(quota, len(source_posts), remaining))
        sampled_posts.extend(selected)
        remaining -= len(selected)
        remaining_sources -= 1
        if remaining <= 0:
            break

    if len(sampled_posts) < max_posts:
        chosen_ids = {id(post) for post in sampled_posts}
        leftovers = [post for post in posts if id(post) not in chosen_ids]
        sampled_posts.extend(_deterministic_even_sample(leftovers, max_posts - len(sampled_posts)))

    sampled_posts = sorted(sampled_posts, key=lambda post: (str(post.get("source", "")), str(post.get("text", ""))))
    return sampled_posts[:max_posts]


def load_daily_hierarchical_records(daily_hierarchical_path: str | Path) -> list[dict]:
    records = []
    with Path(daily_hierarchical_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("daily_hierarchical_sentiment.jsonl is empty.")
    return records


def _build_market_frame_from_hierarchical_records(records: list[dict]) -> pd.DataFrame:
    market_rows = []
    for record in records:
        market_rows.append(
            {
                "date": record.get("date"),
                "feature_date": record.get("feature_date"),
                "subject": record.get("subject"),
                "open": record.get("open"),
                "high": record.get("high"),
                "low": record.get("low"),
                "close": record.get("close"),
                "trading_volume_usdt": record.get("trading_volume_usdt"),
                "sma_14": record.get("sma_14"),
                "ema_14": record.get("ema_14"),
                "rsi_14": record.get("rsi_14"),
                "atr_14": record.get("atr_14"),
                "daily_return": record.get("daily_return"),
                "prev_day_return": record.get("prev_day_return"),
                "rolling_return_mean_3d": record.get("rolling_return_mean_3d"),
                "rolling_volatility_7d": record.get("rolling_volatility_7d"),
                "intraday_range": record.get("intraday_range"),
                "days_since_previous_record": record.get("days_since_previous_record"),
                "trade_day_return": record.get("trade_day_return"),
                "target_trade_day": record.get("target_trade_day"),
                "target_trade_day_numeric": record.get("target_trade_day_numeric"),
                "split": record.get("split"),
            }
        )
    market_df = pd.DataFrame(market_rows)
    if market_df.empty:
        raise ValueError("daily_hierarchical_sentiment.jsonl is empty.")
    market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")
    market_df["feature_date"] = pd.to_datetime(market_df["feature_date"], errors="coerce")
    market_df = market_df.dropna(subset=["date"]).copy()
    market_df["subject"] = _normalize_subject_column(market_df["subject"])
    if "trading_volume_usdt" not in market_df.columns:
        market_df["trading_volume_usdt"] = np.nan
    for indicator_column in ["sma_14", "ema_14", "rsi_14", "atr_14"]:
        if indicator_column not in market_df.columns:
            market_df[indicator_column] = np.nan
    market_df = _sorted_copy(market_df, ["date", "subject"])
    if int(market_df.duplicated(subset=["date", "subject"]).sum()) != 0:
        raise ValueError("daily_hierarchical_sentiment.jsonl contains duplicate date/subject rows.")
    return market_df


def _derive_daily_sentiment_aggregate_features(records: list[dict]) -> pd.DataFrame:
    aggregate_rows = []
    for record in records:
        posts = record.get("posts", [])
        post_count = len(posts)
        price_counts = Counter(_extract_price_sentiment_primary(post.get("price_sentiment")) for post in posts)
        star_counts = Counter(_sentiment_star_bucket(post.get("sentiment_label")) for post in posts)
        sentiment_confidences = pd.Series(
            [_safe_numeric(post.get("sentiment_confidence", post.get("sentiment_score"))) for post in posts],
            dtype=float,
        )
        sentiment_polarities = pd.Series(
            [_safe_numeric(post.get("sentiment_polarity", derive_sentiment_polarity(post.get("sentiment_label")))) for post in posts],
            dtype=float,
        )
        confidence_weighted_polarities = pd.Series(
            [
                _safe_numeric(
                    post.get(
                        "confidence_weighted_polarity",
                        _safe_numeric(post.get("sentiment_confidence", post.get("sentiment_score")))
                        * _safe_numeric(post.get("sentiment_polarity", derive_sentiment_polarity(post.get("sentiment_label")))),
                    )
                )
                for post in posts
            ],
            dtype=float,
        )
        subjectivity_scores = pd.Series([_safe_numeric(post.get("subjectivity_score")) for post in posts], dtype=float)
        token_counts = pd.Series([_safe_numeric(post.get("token_count")) for post in posts], dtype=float)

        aggregate_rows.append(
            {
                "date": str(record["date"]),
                "subject": str(record["subject"]).lower().strip(),
                "post_count": int(post_count),
                "post_count_log1p": float(np.log1p(post_count)),
                "mean_sentiment_confidence": float(sentiment_confidences.mean()) if post_count else 0.0,
                "std_sentiment_confidence": float(sentiment_confidences.std(ddof=0)) if post_count else 0.0,
                "mean_sentiment_polarity": float(sentiment_polarities.mean()) if post_count else 0.0,
                "std_sentiment_polarity": float(sentiment_polarities.std(ddof=0)) if post_count else 0.0,
                "mean_confidence_weighted_polarity": float(confidence_weighted_polarities.mean()) if post_count else 0.0,
                "std_confidence_weighted_polarity": float(confidence_weighted_polarities.std(ddof=0)) if post_count else 0.0,
                "mean_subjectivity_score": float(subjectivity_scores.mean()) if post_count else 0.0,
                "std_subjectivity_score": float(subjectivity_scores.std(ddof=0)) if post_count else 0.0,
                "mean_token_count": float(token_counts.mean()) if post_count else 0.0,
                "std_token_count": float(token_counts.std(ddof=0)) if post_count else 0.0,
                "price_positive_ratio": _safe_ratio(price_counts.get("Positive", 0), post_count),
                "price_negative_ratio": _safe_ratio(price_counts.get("Negative", 0), post_count),
                "bert_low_star_ratio": _safe_ratio(star_counts.get("low", 0), post_count),
                "bert_high_star_ratio": _safe_ratio(star_counts.get("high", 0), post_count),
            }
        )
    return pd.DataFrame(aggregate_rows)


def _clip_series_from_train_stats(series: pd.Series, train_mask: pd.Series, z_value: float = 3.0) -> tuple[pd.Series, dict]:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    train_numeric = numeric.loc[train_mask]
    mean_value = float(train_numeric.mean())
    std_value = _standard_deviation_or_zero(train_numeric)
    if std_value == 0.0:
        lower_bound = upper_bound = mean_value
    else:
        lower_bound = mean_value - z_value * std_value
        upper_bound = mean_value + z_value * std_value
    clipped = numeric.clip(lower=lower_bound, upper=upper_bound)
    return clipped, {
        "lower": float(lower_bound),
        "upper": float(upper_bound),
        "train_mean": mean_value,
        "train_std": std_value,
    }


def _apply_clip_summary(series: pd.Series, clip_info: dict) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.clip(lower=float(clip_info["lower"]), upper=float(clip_info["upper"]))


def _sentiment_star_bucket(label) -> str:
    star_value = convert_sentiment_label(label)
    if star_value <= 2:
        return "low"
    if star_value == 3:
        return "mid"
    return "high"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def build_hierarchical_model_repair_bundle(
    hierarchical_records: list[dict],
    *,
    report_path: str | Path | None = None,
    max_posts_per_day_for_model: int = 128,
    clip_summary: Optional[dict] = None,
) -> dict:
    market_df = _build_market_frame_from_hierarchical_records(hierarchical_records)
    market_df["date_key"] = market_df["date"].dt.strftime("%Y-%m-%d")
    market_df["day_of_week"] = market_df["date"].dt.dayofweek.astype(int)
    market_df["month_number"] = market_df["date"].dt.month.astype(int)
    market_df["month_sin"] = np.sin((2.0 * np.pi * market_df["month_number"]) / 12.0)
    market_df["month_cos"] = np.cos((2.0 * np.pi * market_df["month_number"]) / 12.0)
    market_df["days_since_previous_record"] = pd.to_numeric(
        market_df.get("days_since_previous_record", 1.0),
        errors="coerce",
    ).fillna(1.0).clip(lower=1.0)
    train_mask = market_df["split"].eq("train")

    aggregate_df = _derive_daily_sentiment_aggregate_features(hierarchical_records)
    aggregate_df["date_key"] = aggregate_df["date"].astype(str)

    repaired_df = market_df.merge(
        aggregate_df.drop(columns=["date"]),
        on=["date_key", "subject"],
        how="left",
    )
    if int(repaired_df["post_count"].isna().sum()) != 0:
        raise ValueError("Some hierarchical rows are missing aggregated sentiment features after repair merge.")

    derived_clip_summary = {} if clip_summary is None else dict(clip_summary)
    for source_column, target_column in [
        ("daily_return", "daily_return_clipped"),
        ("prev_day_return", "prev_day_return_clipped"),
        ("rolling_return_mean_3d", "rolling_return_mean_3d_clipped"),
        ("rolling_volatility_7d", "rolling_volatility_7d_clipped"),
    ]:
        if clip_summary is None:
            repaired_df[target_column], derived_clip_summary[target_column] = _clip_series_from_train_stats(
                repaired_df[source_column],
                train_mask,
            )
        else:
            repaired_df[target_column] = _apply_clip_summary(repaired_df[source_column], clip_summary[target_column])

    repaired_df["intraday_range_log1p"] = np.log1p(
        pd.to_numeric(repaired_df["intraday_range"], errors="coerce").clip(lower=0).fillna(0.0)
    )
    repaired_df["trading_volume_usdt_log1p"] = np.log1p(
        pd.to_numeric(repaired_df["trading_volume_usdt"], errors="coerce").clip(lower=0).fillna(0.0)
    )
    close_numeric = pd.to_numeric(repaired_df["close"], errors="coerce").replace(0, np.nan)
    repaired_df["sma_14_gap"] = (
        (pd.to_numeric(repaired_df["close"], errors="coerce") - pd.to_numeric(repaired_df["sma_14"], errors="coerce"))
        / close_numeric
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    repaired_df["ema_14_gap"] = (
        (pd.to_numeric(repaired_df["close"], errors="coerce") - pd.to_numeric(repaired_df["ema_14"], errors="coerce"))
        / close_numeric
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    repaired_df["rsi_14_scaled"] = (
        pd.to_numeric(repaired_df["rsi_14"], errors="coerce").fillna(50.0) / 100.0
    )
    repaired_df["atr_14_log1p"] = np.log1p(
        pd.to_numeric(repaired_df["atr_14"], errors="coerce").clip(lower=0).fillna(0.0)
    )
    repaired_df["days_since_previous_record_log1p"] = np.log1p(
        pd.to_numeric(repaired_df["days_since_previous_record"], errors="coerce").clip(lower=1.0).fillna(1.0)
    )

    numeric_fill_columns = list(dict.fromkeys(MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS + ["post_count"]))
    for column in numeric_fill_columns:
        repaired_df[column] = pd.to_numeric(repaired_df[column], errors="coerce").fillna(0.0)

    repaired_df = _sorted_copy(repaired_df, ["date", "subject"])
    report = {
        "rows": int(len(repaired_df)),
        "date_range": {
            "min": repaired_df["date"].min().strftime("%Y-%m-%d"),
            "max": repaired_df["date"].max().strftime("%Y-%m-%d"),
        },
        "clip_summary": derived_clip_summary,
        "feature_missing_counts": {
            column: int(repaired_df[column].isna().sum())
            for column in MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS + ["post_count"]
        },
        "post_count_summary": {
            "before_cap": {
                "min": int(repaired_df["post_count"].min()),
                "median": float(repaired_df["post_count"].median()),
                "mean": float(repaired_df["post_count"].mean()),
                "max": int(repaired_df["post_count"].max()),
            },
            "runtime_cap": int(max_posts_per_day_for_model),
            "after_cap": {
                "min": int(min(max_posts_per_day_for_model, repaired_df["post_count"].min())),
                "median": float(np.median(np.minimum(repaired_df["post_count"], max_posts_per_day_for_model))),
                "mean": float(np.minimum(repaired_df["post_count"], max_posts_per_day_for_model).mean()),
                "max": int(np.minimum(repaired_df["post_count"], max_posts_per_day_for_model).max()),
            },
        },
    }
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return {
        "hierarchical_records": hierarchical_records,
        "market_df": repaired_df,
        "max_posts_per_day_for_model": int(max_posts_per_day_for_model),
        "report": report,
    }


def _fit_feature_scaler(df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    train_mask = df["split"].eq("train")
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    return scaler


def _prepare_market_dataframe_with_scaled_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    enriched = df.copy()
    numeric = enriched[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = scaler.transform(numeric)
    for index, column in enumerate(feature_columns):
        enriched[f"{column}_scaled"] = scaled[:, index]
    return enriched


class HierarchicalSequenceDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "market_sequence": torch.tensor(sample["market_sequence"], dtype=torch.float32),
            "day_posts": [torch.tensor(day_posts, dtype=torch.float32) for day_posts in sample["day_posts"]],
            "day_post_counts": [int(value) for value in sample["day_post_counts"]],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "date": sample["date"],
            "subject": sample["subject"],
            "split": sample["split"],
            "post_count": int(sample["post_count"]),
            "post_count_after_cap": int(sample["post_count_after_cap"]),
        }


def collate_hierarchical_batch(batch: list[dict]) -> dict:
    batch_size = len(batch)
    lookback = batch[0]["market_sequence"].shape[0]
    post_feature_dim = batch[0]["day_posts"][0].shape[1] if batch[0]["day_posts"] else 1
    max_posts = max(
        max(item["day_post_counts"]) if item["day_post_counts"] else 1
        for item in batch
    )
    max_posts = max(1, int(max_posts))

    post_tensor = torch.zeros((batch_size, lookback, max_posts, post_feature_dim), dtype=torch.float32)
    post_mask = torch.zeros((batch_size, lookback, max_posts), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        for day_index, day_posts in enumerate(item["day_posts"]):
            actual_count = int(item["day_post_counts"][day_index])
            if actual_count <= 0:
                continue
            post_tensor[batch_index, day_index, :actual_count, :] = day_posts[:actual_count]
            post_mask[batch_index, day_index, :actual_count] = True

    return {
        "market_sequence": torch.stack([item["market_sequence"] for item in batch], dim=0),
        "posts": post_tensor,
        "post_mask": post_mask,
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "date": [item["date"] for item in batch],
        "subject": [item["subject"] for item in batch],
        "split": [item["split"] for item in batch],
        "post_count": [item["post_count"] for item in batch],
        "post_count_after_cap": [item["post_count_after_cap"] for item in batch],
    }


def build_hierarchical_sequence_bundle(
    repair_bundle: dict,
    *,
    lookback: int,
    max_posts_per_day_for_model: Optional[int] = None,
    market_scaler: Optional[StandardScaler] = None,
    post_state: Optional[dict] = None,
) -> dict:
    hierarchical_records = repair_bundle["hierarchical_records"]
    market_df = repair_bundle["market_df"].copy()
    max_posts_per_day_for_model = (
        repair_bundle.get("max_posts_per_day_for_model")
        if max_posts_per_day_for_model is None
        else max_posts_per_day_for_model
    )

    if market_scaler is None:
        market_scaler = _fit_feature_scaler(market_df, MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS)
    market_df = _prepare_market_dataframe_with_scaled_features(
        market_df,
        market_scaler,
        MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS,
    )

    if post_state is None:
        post_state = _fit_post_feature_state(hierarchical_records)
    post_feature_dim = (
        len(POST_NUMERIC_COLUMNS)
        + len(post_state["source_map"])
        + len(post_state["sentiment_label_map"])
        + len(post_state["price_sentiment_map"])
    )
    scaled_columns = [f"{column}_scaled" for column in MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS]

    market_lookup = {}
    for _, row in market_df.iterrows():
        market_lookup[(row["date"].strftime("%Y-%m-%d"), row["subject"])] = row

    records_by_subject = {}
    for record in hierarchical_records:
        subject = str(record["subject"]).lower().strip()
        key = (str(record["date"]), subject)
        market_row = market_lookup.get(key)
        if market_row is None:
            continue
        sampled_posts = sample_posts_for_model(record.get("posts", []), max_posts_per_day_for_model)
        encoded_posts = [encode_post_feature_vector(post, post_state) for post in sampled_posts]
        subject_record = {
            "date": key[0],
            "subject": subject,
            "split": record.get("split", "unknown"),
            "label": int(record.get("target_trade_day_numeric", 1)),
            "market_vector": market_row[scaled_columns].to_numpy(dtype=np.float32),
            "day_posts": encoded_posts,
            "post_count": int(record.get("post_count", len(record.get("posts", [])))),
            "post_count_after_cap": int(len(sampled_posts)),
        }
        records_by_subject.setdefault(subject, []).append(subject_record)

    samples_by_split: dict[str, list[dict]] = {}
    empty_post_template = np.zeros((1, post_feature_dim), dtype=np.float32)
    for subject, subject_records in records_by_subject.items():
        subject_records = sorted(subject_records, key=lambda item: item["date"])
        for end_index in range(lookback - 1, len(subject_records)):
            history = subject_records[end_index - lookback + 1 : end_index + 1]
            if len(history) != lookback:
                continue
            final_record = history[-1]
            day_posts = []
            day_post_counts = []
            for day_record in history:
                actual_posts = day_record["day_posts"]
                day_post_counts.append(int(len(actual_posts)))
                if actual_posts:
                    day_posts.append(np.stack(actual_posts).astype(np.float32))
                else:
                    day_posts.append(empty_post_template.copy())
            sample = {
                "market_sequence": np.stack([day_record["market_vector"] for day_record in history]).astype(np.float32),
                "day_posts": day_posts,
                "day_post_counts": day_post_counts,
                "label": int(final_record["label"]),
                "date": final_record["date"],
                "subject": subject,
                "split": final_record["split"],
                "post_count": int(final_record["post_count"]),
                "post_count_after_cap": int(final_record["post_count_after_cap"]),
            }
            samples_by_split.setdefault(final_record["split"], []).append(sample)

    datasets = {split_name: HierarchicalSequenceDataset(samples) for split_name, samples in samples_by_split.items()}
    sample_summary = {
        split_name: {
            "rows": int(len(samples)),
            "class_counts": dict(sorted(Counter(sample["label"] for sample in samples).items())),
        }
        for split_name, samples in samples_by_split.items()
    }
    return {
        "datasets": datasets,
        "samples_by_split": samples_by_split,
        "market_df": market_df,
        "hierarchical_records": hierarchical_records,
        "market_feature_columns": MODEL_HIERARCHICAL_DAY_FEATURE_COLUMNS,
        "post_feature_state": post_state,
        "post_feature_dim": int(post_feature_dim),
        "lookback": int(lookback),
        "market_scaler": market_scaler,
        "sample_summary": sample_summary,
        "max_posts_per_day_for_model": None if max_posts_per_day_for_model is None else int(max_posts_per_day_for_model),
    }


def build_dataloader(dataset: Dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_hierarchical_batch,
    )


class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, sequence_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence_outputs * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class HierarchicalRecurrentClassifier(nn.Module):
    def __init__(
        self,
        market_input_dim: int,
        post_input_dim: int,
        *,
        recurrent_type: str,
        post_hidden_dim: int = 48,
        day_hidden_dim: int = 128,
        recurrent_layers: int = 2,
        recurrent_dropout: float = 0.25,
        classifier_dropout: float = 0.30,
    ):
        super().__init__()
        if recurrent_type not in {"lstm", "gru", "rnn"}:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")
        self.recurrent_type = recurrent_type
        self.post_mlp = nn.Sequential(
            nn.Linear(post_input_dim, post_hidden_dim),
            nn.LayerNorm(post_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(post_hidden_dim, post_hidden_dim),
            nn.LayerNorm(post_hidden_dim),
            nn.GELU(),
        )
        self.post_attention = nn.Linear(post_hidden_dim, 1)
        self.day_projection = nn.Sequential(
            nn.Linear(market_input_dim + post_hidden_dim, day_hidden_dim),
            nn.LayerNorm(day_hidden_dim),
            nn.GELU(),
        )
        recurrent_dropout_value = recurrent_dropout if recurrent_layers > 1 else 0.0
        recurrent_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[recurrent_type]
        self.sequence_encoder = recurrent_cls(
            input_size=day_hidden_dim,
            hidden_size=day_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=recurrent_dropout_value,
            bidirectional=True,
        )
        self.temporal_attention = TemporalAttentionPooling(day_hidden_dim * 2)
        self.output_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(day_hidden_dim * 4),
            nn.Linear(day_hidden_dim * 4, day_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(day_hidden_dim * 2, len(MODEL_LABELS)),
        )

    def _pool_posts(self, posts: torch.Tensor, post_mask: torch.Tensor) -> torch.Tensor:
        batch_size, lookback, max_posts, post_input_dim = posts.shape
        encoded_posts = self.post_mlp(posts.reshape(batch_size * lookback * max_posts, post_input_dim))
        encoded_posts = encoded_posts.reshape(batch_size, lookback, max_posts, -1)
        attention_scores = self.post_attention(encoded_posts).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~post_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.where(post_mask, attention_weights, torch.zeros_like(attention_weights))
        pooled_posts = torch.sum(encoded_posts * attention_weights.unsqueeze(-1), dim=2)
        return pooled_posts

    def _final_hidden(self, hidden_state) -> torch.Tensor:
        if self.recurrent_type == "lstm":
            hidden_state = hidden_state[0]
        _, batch_size, hidden_dim = hidden_state.shape
        hidden_state = hidden_state.view(-1, 2, batch_size, hidden_dim)
        final_layer_hidden = hidden_state[-1].transpose(0, 1).reshape(batch_size, hidden_dim * 2)
        return final_layer_hidden

    def forward(self, market_sequence: torch.Tensor, posts: torch.Tensor, post_mask: torch.Tensor) -> torch.Tensor:
        pooled_posts = self._pool_posts(posts, post_mask)
        combined_days = torch.cat([market_sequence, pooled_posts], dim=-1)
        projected_days = self.day_projection(combined_days)
        sequence_outputs, hidden_state = self.sequence_encoder(projected_days)
        attention_pooled, _ = self.temporal_attention(sequence_outputs)
        final_hidden = self._final_hidden(hidden_state)
        features = torch.cat([attention_pooled, final_hidden], dim=-1)
        features = self.output_dropout(features)
        return self.classifier(features)


class HierarchicalMarketSentimentRNN(HierarchicalRecurrentClassifier):
    def __init__(
        self,
        market_input_dim: int,
        post_input_dim: int,
        post_hidden_dim: int = 48,
        day_hidden_dim: int = 128,
        recurrent_layers: int = 2,
        recurrent_dropout: float = 0.25,
        classifier_dropout: float = 0.30,
    ):
        super().__init__(
            market_input_dim,
            post_input_dim,
            recurrent_type="rnn",
            post_hidden_dim=post_hidden_dim,
            day_hidden_dim=day_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout,
            classifier_dropout=classifier_dropout,
        )


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {
        "market_sequence": batch["market_sequence"].to(device),
        "posts": batch["posts"].to(device),
        "post_mask": batch["post_mask"].to(device),
        "label": batch["label"].to(device),
    }


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for batch in dataloader:
            device_batch = _move_batch_to_device(batch, device)
            logits = model(
                device_batch["market_sequence"],
                device_batch["posts"],
                device_batch["post_mask"],
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            all_labels.extend(device_batch["label"].cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
    if not all_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "predictions": []}
    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_predictions, average="weighted", zero_division=0)),
        "predictions": [
            {
                "true_label_numeric": int(label),
                "predicted_label_numeric": int(prediction),
                "probabilities": probability,
            }
            for label, prediction, probability in zip(all_labels, all_predictions, all_probabilities)
        ],
    }


def train_hierarchical_rnn(sequence_bundle: dict, *, config: Optional[dict] = None) -> dict:
    config = {**DEFAULT_ARTIFACT_CONFIG, **(config or {})}
    set_reproducible_seeds(int(config["random_seed"]))
    device = choose_torch_device()
    train_dataset = sequence_bundle["datasets"].get("train")
    validation_dataset = sequence_bundle["datasets"].get("validation")
    test_dataset = sequence_bundle["datasets"].get("test")
    if train_dataset is None or validation_dataset is None or test_dataset is None:
        raise ValueError("Training, validation, and test datasets are required for artifact export.")

    train_loader = build_dataloader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    validation_loader = build_dataloader(validation_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    model = HierarchicalMarketSentimentRNN(
        market_input_dim=len(sequence_bundle["market_feature_columns"]),
        post_input_dim=sequence_bundle["post_feature_dim"],
        post_hidden_dim=int(config["post_hidden_dim"]),
        day_hidden_dim=int(config["sequence_hidden_dim"]),
        recurrent_layers=int(config["recurrent_layers"]),
        recurrent_dropout=float(config["recurrent_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_epoch = 0
    best_validation_macro_f1 = -math.inf
    patience_counter = 0
    history = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            device_batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(
                device_batch["market_sequence"],
                device_batch["posts"],
                device_batch["post_mask"],
            )
            loss = criterion(logits, device_batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = int(device_batch["label"].shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        validation_metrics = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(sample_count, 1),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
            }
        )
        if validation_metrics["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(config["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    validation_metrics = evaluate_model(model, validation_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    return {
        "model": model.cpu(),
        "best_state_dict": best_state,
        "best_epoch": int(best_epoch),
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "device": device,
    }


def save_artifact_bundle(path: str | Path, payload: dict) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, artifact_path)
    return artifact_path


def load_artifact_bundle(path: str | Path, *, map_location: str = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_prediction_record(
    *,
    subject: str,
    feature_date: pd.Timestamp,
    predicted_date: pd.Timestamp,
    market_row: pd.Series,
    posts: list[dict],
) -> dict:
    return {
        "date": predicted_date.strftime("%Y-%m-%d"),
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "subject": subject,
        "open": _safe_numeric(market_row["open"]),
        "high": _safe_numeric(market_row["high"]),
        "low": _safe_numeric(market_row["low"]),
        "close": _safe_numeric(market_row["close"]),
        "trading_volume_usdt": _safe_numeric(market_row["trading_volume_usdt"]),
        "sma_14": _safe_numeric(market_row["sma_14"]),
        "ema_14": _safe_numeric(market_row["ema_14"]),
        "rsi_14": _safe_numeric(market_row["rsi_14"]),
        "atr_14": _safe_numeric(market_row["atr_14"]),
        "daily_return": _safe_numeric(market_row["daily_return"]),
        "prev_day_return": _safe_numeric(market_row["prev_day_return"]),
        "rolling_return_mean_3d": _safe_numeric(market_row["rolling_return_mean_3d"]),
        "rolling_volatility_7d": _safe_numeric(market_row["rolling_volatility_7d"]),
        "intraday_range": _safe_numeric(market_row["intraday_range"]),
        "days_since_previous_record": _safe_numeric(market_row["days_since_previous_record"], 1.0),
        "trade_day_return": None,
        "target_trade_day": "Neutral",
        "target_trade_day_numeric": 1,
        "split": "inference",
        "post_count": int(len(posts)),
        "posts": posts,
    }


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def add_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy().sort_values("date").reset_index(drop=True)
    enriched["daily_return"] = ((enriched["close"] - enriched["open"]) / enriched["open"]) * 100.0
    enriched["prev_day_return"] = enriched["daily_return"].shift(1)
    enriched["rolling_return_mean_3d"] = enriched["daily_return"].rolling(window=3, min_periods=1).mean()
    enriched["rolling_volatility_7d"] = enriched["daily_return"].rolling(window=7, min_periods=2).std().fillna(0.0)
    enriched["intraday_range"] = enriched["high"] - enriched["low"]
    date_deltas = enriched["date"].diff().dt.days
    enriched["days_since_previous_record"] = date_deltas.fillna(1).clip(lower=1).astype(float)
    enriched["sma_14"] = enriched["close"].rolling(window=14, min_periods=1).mean()
    enriched["ema_14"] = enriched["close"].ewm(span=14, adjust=False).mean()
    enriched["rsi_14"] = compute_rsi(enriched["close"], period=14)
    enriched["atr_14"] = compute_atr(enriched, period=14)
    return enriched


def fetch_binance_market_history(symbol: str, *, limit: int = 90, timeout_seconds: int = 20) -> pd.DataFrame:
    params = urllib.parse.urlencode({"symbol": symbol, "interval": "1d", "limit": limit})
    url = f"https://api.binance.com/api/v3/klines?{params}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in payload:
        rows.append(
            {
                "date": datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc).date().isoformat(),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "trading_volume_usdt": float(item[7]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def build_live_market_frame(symbol: str, subject: str, *, end_feature_date: pd.Timestamp, fetch_limit: int = 90) -> pd.DataFrame:
    market_df = fetch_binance_market_history(symbol, limit=fetch_limit)
    market_df = market_df[market_df["date"] <= pd.to_datetime(end_feature_date)].copy()
    if market_df.empty:
        raise ValueError("No market history is available for the requested asset.")
    market_df = add_market_context_features(market_df)
    market_df["subject"] = subject
    market_df["feature_date"] = market_df["date"]
    market_df["date"] = market_df["feature_date"] + pd.Timedelta(days=1)
    market_df["target_trade_day"] = "Neutral"
    market_df["target_trade_day_numeric"] = 1
    market_df["split"] = "inference"
    return market_df.reset_index(drop=True)


def build_inference_records(
    *,
    market_df: pd.DataFrame,
    grouped_posts: dict[str, list[dict]],
    subject: str,
    lookback_days: int,
) -> list[dict]:
    if len(market_df) < lookback_days:
        raise ValueError(f"Need at least {lookback_days} market rows to build an inference window.")
    window_df = market_df.tail(lookback_days).copy().reset_index(drop=True)
    records = []
    for _, row in window_df.iterrows():
        feature_date = pd.to_datetime(row["feature_date"], errors="coerce")
        if pd.isna(feature_date):
            continue
        feature_date_key = feature_date.strftime("%Y-%m-%d")
        posts = grouped_posts.get(feature_date_key, [])
        records.append(
            build_prediction_record(
                subject=subject,
                feature_date=feature_date,
                predicted_date=pd.to_datetime(row["date"], errors="coerce"),
                market_row=row,
                posts=posts,
            )
        )
    if len(records) < lookback_days:
        raise ValueError(f"Need {lookback_days} valid records after alignment, found {len(records)}.")
    return records


def build_inference_batch_from_artifact(records: list[dict], artifact: dict) -> dict:
    repair_bundle = build_hierarchical_model_repair_bundle(
        records,
        clip_summary=artifact["clip_summary"],
        max_posts_per_day_for_model=int(artifact["config"]["max_posts_per_day"]),
    )
    sequence_bundle = build_hierarchical_sequence_bundle(
        repair_bundle,
        lookback=int(artifact["config"]["lookback_days"]),
        max_posts_per_day_for_model=int(artifact["config"]["max_posts_per_day"]),
        market_scaler=artifact["market_scaler"],
        post_state=artifact["post_feature_state"],
    )
    inference_samples = sequence_bundle["samples_by_split"].get("inference", [])
    if not inference_samples:
        raise ValueError("The inference window could not be assembled into a model sample.")
    return collate_hierarchical_batch([sequence_bundle["datasets"]["inference"][len(inference_samples) - 1]])


def build_model_from_artifact(artifact: dict) -> HierarchicalMarketSentimentRNN:
    config = artifact["config"]
    model = HierarchicalMarketSentimentRNN(
        market_input_dim=len(artifact["market_feature_columns"]),
        post_input_dim=int(artifact["post_feature_dim"]),
        post_hidden_dim=int(config["post_hidden_dim"]),
        day_hidden_dim=int(config["sequence_hidden_dim"]),
        recurrent_layers=int(config["recurrent_layers"]),
        recurrent_dropout=float(config["recurrent_dropout"]),
        classifier_dropout=float(config["classifier_dropout"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model


def predict_batch(model: nn.Module, batch: dict, *, device: str = "cpu") -> dict:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        device_batch = _move_batch_to_device(batch, device)
        logits = model(
            device_batch["market_sequence"],
            device_batch["posts"],
            device_batch["post_mask"],
        )
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_label_numeric": predicted_index,
        "prediction_label": MODEL_LABELS[predicted_index],
        "probabilities": {
            MODEL_LABELS[index]: float(probabilities[index])
            for index in range(len(MODEL_LABELS))
        },
        "confidence_percentage": float(max(probabilities) * 100.0),
    }
