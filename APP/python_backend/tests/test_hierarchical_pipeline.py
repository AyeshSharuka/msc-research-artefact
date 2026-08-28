import unittest

import pandas as pd

from hierarchical_pipeline import (
    build_inference_records,
    normalize_price_sentiment,
    sample_posts_for_model,
)


class HierarchicalPipelineTests(unittest.TestCase):
    def test_normalize_price_sentiment_defaults_to_neutral(self):
        self.assertEqual(normalize_price_sentiment(None), "['Neutral']")
        self.assertEqual(normalize_price_sentiment("[]"), "['Neutral']")

    def test_sample_posts_for_model_caps_deterministically(self):
        posts = [
            {"source": "news", "text": f"n{i}"}
            for i in range(4)
        ] + [
            {"source": "reddit", "text": f"r{i}"}
            for i in range(4)
        ]
        sampled = sample_posts_for_model(posts, 4)
        self.assertEqual(len(sampled), 4)
        self.assertTrue(any(post["source"] == "news" for post in sampled))
        self.assertTrue(any(post["source"] == "reddit" for post in sampled))

    def test_build_inference_records_preserves_zero_post_days(self):
        market_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-23"),
                    "feature_date": pd.Timestamp("2026-08-22"),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100.5,
                    "trading_volume_usdt": 1000,
                    "sma_14": 99.5,
                    "ema_14": 99.0,
                    "rsi_14": 52,
                    "atr_14": 2.2,
                    "daily_return": 0.5,
                    "prev_day_return": 0.1,
                    "rolling_return_mean_3d": 0.2,
                    "rolling_volatility_7d": 1.0,
                    "intraday_range": 2.0,
                    "days_since_previous_record": 1.0,
                },
                {
                    "date": pd.Timestamp("2026-08-24"),
                    "feature_date": pd.Timestamp("2026-08-23"),
                    "open": 101,
                    "high": 102,
                    "low": 100,
                    "close": 101.5,
                    "trading_volume_usdt": 1100,
                    "sma_14": 100.0,
                    "ema_14": 99.8,
                    "rsi_14": 54,
                    "atr_14": 2.0,
                    "daily_return": 0.4,
                    "prev_day_return": 0.5,
                    "rolling_return_mean_3d": 0.3,
                    "rolling_volatility_7d": 0.9,
                    "intraday_range": 2.0,
                    "days_since_previous_record": 1.0,
                },
            ]
        )
        grouped_posts = {
            "2026-08-23": [{"text": "btc news"}],
        }
        records = build_inference_records(
            market_df=market_df,
            grouped_posts=grouped_posts,
            subject="bitcoin",
            lookback_days=2,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["posts"], [])
        self.assertEqual(len(records[1]["posts"]), 1)


if __name__ == "__main__":
    unittest.main()
