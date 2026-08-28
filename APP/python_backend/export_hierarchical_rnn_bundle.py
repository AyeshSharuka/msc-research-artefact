import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from hierarchical_pipeline import (
    DEFAULT_ARTIFACT_CONFIG,
    build_hierarchical_model_repair_bundle,
    build_hierarchical_sequence_bundle,
    load_daily_hierarchical_records,
    save_artifact_bundle,
    train_hierarchical_rnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the hierarchical RNN backend artifact.")
    parser.add_argument(
        "--daily-path",
        default="../required_modules_upgrade_2026-07-28/data/daily_hierarchical_sentiment.jsonl",
        help="Path to the notebook-generated hierarchical JSONL dataset.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/hierarchical_rnn_bundle.pt",
        help="Where to save the exported artifact bundle.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/hierarchical_rnn_bundle_metrics.json",
        help="Where to save a readable export summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily_path = Path(args.daily_path).resolve()
    output_path = Path(args.output_path).resolve()
    metrics_path = Path(args.metrics_path).resolve()

    records = load_daily_hierarchical_records(daily_path)
    repair_bundle = build_hierarchical_model_repair_bundle(
        records,
        max_posts_per_day_for_model=int(DEFAULT_ARTIFACT_CONFIG["max_posts_per_day"]),
    )
    sequence_bundle = build_hierarchical_sequence_bundle(
        repair_bundle,
        lookback=int(DEFAULT_ARTIFACT_CONFIG["lookback_days"]),
        max_posts_per_day_for_model=int(DEFAULT_ARTIFACT_CONFIG["max_posts_per_day"]),
    )
    training_result = train_hierarchical_rnn(sequence_bundle, config=DEFAULT_ARTIFACT_CONFIG)

    artifact_payload = {
        "artifact_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_daily_path": str(daily_path),
        "config": dict(DEFAULT_ARTIFACT_CONFIG),
        "market_feature_columns": list(sequence_bundle["market_feature_columns"]),
        "post_feature_dim": int(sequence_bundle["post_feature_dim"]),
        "post_numeric_columns": list(sequence_bundle["post_feature_state"]["numeric_scaler"].feature_names_in_),
        "label_map": dict(DEFAULT_ARTIFACT_CONFIG.get("label_map", {})),
        "model_state_dict": training_result["best_state_dict"],
        "model_state_name": DEFAULT_ARTIFACT_CONFIG["model_name"],
        "model_state_best_epoch": int(training_result["best_epoch"]),
        "market_scaler": sequence_bundle["market_scaler"],
        "post_feature_state": sequence_bundle["post_feature_state"],
        "clip_summary": repair_bundle["report"]["clip_summary"],
        "sample_summary": sequence_bundle["sample_summary"],
        "validation_metrics": training_result["validation_metrics"],
        "test_metrics": training_result["test_metrics"],
    }
    artifact_payload["label_map"] = {"Buy": 0, "Neutral": 1, "Sell": 2}
    save_artifact_bundle(output_path, artifact_payload)

    metrics_payload = {
        "artifact_path": str(output_path),
        "config": artifact_payload["config"],
        "best_epoch": artifact_payload["model_state_best_epoch"],
        "sample_summary": artifact_payload["sample_summary"],
        "validation_metrics": artifact_payload["validation_metrics"],
        "test_metrics": artifact_payload["test_metrics"],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
