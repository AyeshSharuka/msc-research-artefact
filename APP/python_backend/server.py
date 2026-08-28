import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from textblob import TextBlob
from transformers import AutoTokenizer, pipeline

from hierarchical_pipeline import (
    MODEL_LABELS,
    build_inference_batch_from_artifact,
    build_inference_records,
    build_live_market_frame,
    build_model_from_artifact,
    choose_torch_device,
    derive_sentiment_polarity,
    load_artifact_bundle,
    normalize_price_sentiment,
    predict_batch,
)

try:
    from pyabsa import ATEPCCheckpointManager
except Exception as exc:  # pragma: no cover - optional runtime dependency
    ATEPCCheckpointManager = None
    PYABSA_IMPORT_ERROR = str(exc)
else:
    PYABSA_IMPORT_ERROR = None

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "artifacts" / "hierarchical_rnn_bundle.pt"
PYABSA_CHECKPOINT = (
    APP_DIR.parent
    / "required_modules_upgrade_2026-07-28"
    / "checkpoints"
    / "ATEPC_ENGLISH_CHECKPOINT"
    / "fast_lcf_atepc_English_cdw_apcacc_82.36_apcf1_81.89_atef1_75.43"
)
BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
TOKENIZER_NAME = "bert-base-uncased"
ASSET_SYMBOL_MAP = {
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "btcusdt": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "eth": "ETHUSDT",
    "ethusdt": "ETHUSDT",
}
ASSET_SUBJECT_MAP = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "btcusdt": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "ethusdt": "ethereum",
}


class Article(BaseModel):
    text: str
    date: datetime
    source: Optional[str] = None


class RequestData(BaseModel):
    asset: str
    news: List[Article] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction_label: str
    predicted_label_numeric: int
    probabilities: dict[str, float]
    confidence_percentage: float
    model_name: str
    lookback_days: int
    input_summary: dict
    warnings: List[str] = Field(default_factory=list)


def normalize_asset(asset: str) -> tuple[str, str]:
    normalized = str(asset).strip().lower()
    subject = ASSET_SUBJECT_MAP.get(normalized)
    symbol = ASSET_SYMBOL_MAP.get(normalized)
    if subject is None or symbol is None:
        raise HTTPException(status_code=400, detail=f"Unsupported asset '{asset}'. Use bitcoin/BTCUSDT or ethereum/ETHUSDT.")
    return subject, symbol


def clean_text_pipeline(text: str) -> str:
    cleaned = re.sub(r"http\S+|www\.\S+", " ", str(text))
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@contextmanager
def temporary_chdir(target_dir: str | Path):
    original_dir = Path.cwd()
    os.chdir(Path(target_dir))
    try:
        yield
    finally:
        os.chdir(original_dir)


class HierarchicalPredictionService:
    def __init__(self) -> None:
        self.device = choose_torch_device()
        self.artifact_path = ARTIFACT_PATH
        self.artifact = None
        self.model = None
        self.model_load_error: Optional[str] = None
        self.tokenizer = None
        self.sentiment_analyzer = None
        self.text_stack_error: Optional[str] = None
        self.aspect_model = None
        self.pyabsa_error: Optional[str] = PYABSA_IMPORT_ERROR
        self._load_artifact()

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            self.model_load_error = f"Artifact not found at {self.artifact_path}"
            return
        try:
            self.artifact = load_artifact_bundle(self.artifact_path, map_location="cpu")
            self.model = build_model_from_artifact(self.artifact)
            self.model_load_error = None
        except Exception as exc:  # pragma: no cover - startup failure path
            self.artifact = None
            self.model = None
            self.model_load_error = str(exc)

    def _ensure_text_stack(self) -> None:
        if self.sentiment_analyzer is not None and self.tokenizer is not None:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            self.sentiment_analyzer = pipeline("sentiment-analysis", model=BERT_MODEL_NAME)
            self.text_stack_error = None
        except Exception as exc:  # pragma: no cover - depends on model availability
            self.text_stack_error = str(exc)
            raise HTTPException(status_code=503, detail=f"Text model unavailable: {exc}") from exc

    def _ensure_pyabsa(self) -> None:
        if self.aspect_model is not None or ATEPCCheckpointManager is None:
            return
        try:
            checkpoint = str(PYABSA_CHECKPOINT) if PYABSA_CHECKPOINT.exists() else "english"
            if checkpoint == "english":
                self.aspect_model = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=checkpoint)
            else:
                checkpoint_path = Path(checkpoint).resolve()
                common_root = Path(os.path.commonpath([str(APP_DIR.parent.resolve()), str(checkpoint_path)]))
                with temporary_chdir(common_root):
                    self.aspect_model = ATEPCCheckpointManager.get_aspect_extractor(checkpoint=str(checkpoint_path))
            self.pyabsa_error = None
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self.aspect_model = None
            self.pyabsa_error = str(exc)

    def health_payload(self) -> dict:
        config = self.artifact["config"] if self.artifact else None
        return {
            "status": "ok" if self.model is not None else "degraded",
            "model_loaded": self.model is not None,
            "model_error": self.model_load_error,
            "artifact_path": str(self.artifact_path),
            "model_name": config["model_name"] if config else None,
            "lookback_days": config["lookback_days"] if config else None,
            "max_posts_per_day": config["max_posts_per_day"] if config else None,
            "device": self.device,
            "text_stack_loaded": self.sentiment_analyzer is not None and self.tokenizer is not None,
            "text_stack_error": self.text_stack_error,
            "pyabsa_loaded": self.aspect_model is not None,
            "pyabsa_error": self.pyabsa_error,
        }

    def _score_sentiment_batch(self, texts: list[str]) -> list[dict]:
        self._ensure_text_stack()
        results = self.sentiment_analyzer(texts, truncation=True)
        return results

    def _extract_price_sentiment(self, text: str) -> tuple[str, Optional[str]]:
        self._ensure_pyabsa()
        if self.aspect_model is None:
            return normalize_price_sentiment("Neutral"), self.pyabsa_error
        try:
            result = self.aspect_model.extract_aspect([text])
            sentiments = []
            for item in result:
                aspect = str(item.get("aspect", "")).lower()
                sentiment = str(item.get("sentiment", "")).title()
                if "price" in aspect and sentiment in {"Positive", "Negative", "Neutral"}:
                    sentiments.append(sentiment)
            if sentiments:
                return normalize_price_sentiment(sentiments), None
        except Exception as exc:  # pragma: no cover - optional runtime
            self.pyabsa_error = str(exc)
        return normalize_price_sentiment("Neutral"), self.pyabsa_error

    def _fallback_price_sentiment(self, polarity: float) -> str:
        if polarity >= 0.25:
            return normalize_price_sentiment("Positive")
        if polarity <= -0.25:
            return normalize_price_sentiment("Negative")
        return normalize_price_sentiment("Neutral")

    def normalize_news(self, subject: str, news_items: List[Article]) -> tuple[list[dict], list[str]]:
        warnings: list[str] = []
        cleaned_rows = []
        raw_rows = []
        for item in news_items:
            text = clean_text_pipeline(item.text)
            if not text:
                continue
            raw_rows.append(
                {
                    "text": text,
                    "date": item.date,
                    "source": item.source or "news",
                }
            )
        if not raw_rows:
            raise HTTPException(status_code=422, detail="No usable news remained after text cleaning.")

        sentiment_rows = self._score_sentiment_batch([row["text"] for row in raw_rows])
        for row, sentiment in zip(raw_rows, sentiment_rows):
            token_count = len(self.tokenizer.tokenize(row["text"]))
            word_count = len(row["text"].split())
            if word_count <= 10 or token_count < 5:
                continue
            label = sentiment["label"]
            confidence = float(sentiment["score"])
            polarity = derive_sentiment_polarity(label)
            price_sentiment, pyabsa_error = self._extract_price_sentiment(row["text"])
            if pyabsa_error:
                warnings.append("PyABSA unavailable; using deterministic price-sentiment fallback.")
                price_sentiment = self._fallback_price_sentiment(polarity)
            cleaned_rows.append(
                {
                    "text": row["text"],
                    "source": str(row["source"]).strip().lower() or "news",
                    "subject": subject,
                    "date": row["date"].date().isoformat(),
                    "token_count": token_count,
                    "sentiment_label": label,
                    "sentiment_score": confidence,
                    "sentiment_confidence": confidence,
                    "sentiment_polarity": polarity,
                    "confidence_weighted_polarity": confidence * polarity,
                    "subjectivity_score": float(TextBlob(row["text"]).sentiment.subjectivity),
                    "price_sentiment": price_sentiment,
                }
            )
        if not cleaned_rows:
            raise HTTPException(status_code=422, detail="No usable news remained after notebook-style filtering.")
        return cleaned_rows, sorted(set(warnings))

    def predict(self, request_data: RequestData) -> PredictionResponse:
        if self.model is None or self.artifact is None:
            raise HTTPException(status_code=503, detail=self.model_load_error or "Model artifact is not available.")

        subject, symbol = normalize_asset(request_data.asset)
        normalized_posts, warnings = self.normalize_news(subject, request_data.news)
        grouped_posts: dict[str, list[dict]] = {}
        for post in normalized_posts:
            grouped_posts.setdefault(post["date"], []).append(post)

        unique_news_days = len(grouped_posts)
        if unique_news_days == 1:
            warnings.append("All usable posts are concentrated on a single feature day.")

        latest_feature_date = max(pd.to_datetime(date_key) for date_key in grouped_posts)
        try:
            market_df = build_live_market_frame(
                symbol,
                subject,
                end_feature_date=latest_feature_date,
                fetch_limit=max(90, int(self.artifact["config"]["lookback_days"]) * 6),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Unable to fetch live market history: {exc}") from exc

        lookback_days = int(self.artifact["config"]["lookback_days"])
        if len(market_df) < lookback_days:
            raise HTTPException(
                status_code=503,
                detail=f"Need at least {lookback_days} days of market history, found {len(market_df)}.",
            )

        records = build_inference_records(
            market_df=market_df,
            grouped_posts=grouped_posts,
            subject=subject,
            lookback_days=lookback_days,
        )
        batch = build_inference_batch_from_artifact(records, self.artifact)
        prediction = predict_batch(self.model, batch, device="cpu")

        populated_days = sum(1 for record in records if record["posts"])
        if populated_days < max(2, lookback_days // 2):
            warnings.append("Sparse news coverage across the inference window.")

        return PredictionResponse(
            prediction_label=prediction["prediction_label"],
            predicted_label_numeric=prediction["predicted_label_numeric"],
            probabilities=prediction["probabilities"],
            confidence_percentage=prediction["confidence_percentage"],
            model_name=str(self.artifact["config"]["model_name"]),
            lookback_days=lookback_days,
            input_summary={
                "asset": subject,
                "symbol": symbol,
                "raw_news_items": len(request_data.news),
                "usable_news_items": len(normalized_posts),
                "news_days_populated": populated_days,
                "unique_news_days": unique_news_days,
                "window_days": len(records),
                "target_trade_day": records[-1]["date"],
                "latest_feature_day": records[-1]["feature_date"],
            },
            warnings=sorted(set(warnings)),
        )


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = HierarchicalPredictionService()


@app.get("/health")
async def healthcheck():
    return service.health_payload()


@app.post("/predict", response_model=PredictionResponse)
@app.post("/predict/", response_model=PredictionResponse)
async def predict(request_data: RequestData):
    return service.predict(request_data)
