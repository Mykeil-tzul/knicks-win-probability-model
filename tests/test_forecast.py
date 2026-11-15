import json
import unittest
from datetime import datetime, timezone

from forecast import KNICKS, ROOT, historical_games, pregame_rows
from scripts.refresh import build_forecasts


class ForecastTests(unittest.TestCase):
    def test_same_day_results_cannot_change_features(self):
        first = {"id": "1", "date": "2026-10-20", "season": 2027, "home": KNICKS,
                 "away": 1610612738, "home_score": 100, "away_score": 90}
        second = dict(first, id="2")
        changed = dict(first, home_score=80)
        baseline = list(pregame_rows([first, second]))[1][1]
        self.assertEqual(baseline, list(pregame_rows([changed, second]))[1][1])
        next_day = dict(second, date="2026-10-21")
        self.assertNotEqual(list(pregame_rows([first, next_day]))[1][1],
                            list(pregame_rows([changed, next_day]))[1][1])

    def test_lock_only_before_tip_and_keep_original(self):
        game = {"id": "1", "date": "2026-11-15", "season": 2027, "home": KNICKS,
                "away": 1610612738, "state": "scheduled", "tip": "2026-11-15T23:00:00+00:00"}
        model = {"intercept": 1, "weights": [0] * 5}
        before = datetime(2026, 11, 15, 20, tzinfo=timezone.utc)
        after = datetime(2026, 11, 16, 0, tzinfo=timezone.utc)
        _, ledger = build_forecasts([], [game], model, {}, before, before)
        self.assertGreater(ledger["1"]["probability"], 0.5)
        early, unchanged = build_forecasts([], [game], dict(model, intercept=-1), ledger, before, before)
        self.assertLess(early["1"]["probability"], 0.5)
        self.assertEqual(unchanged, ledger)
        self.assertEqual(build_forecasts([], [game], model, {}, before, after), ({}, {}))

    def test_bundle_and_app(self):
        from streamlit.testing.v1 import AppTest

        self.assertEqual(len(historical_games()), 4920)
        snapshot = json.loads((ROOT / "data/season.json").read_text())
        self.assertGreaterEqual(len(snapshot["knicks"]), 75)
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(any("Model performance" in tab.label for tab in app.tabs))
