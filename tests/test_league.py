import json
import unittest
from datetime import datetime, timezone

from forecast import ROOT, KNICKS, historical_games
from league import TEAMS, prediction, scenario_features, recorded_pick, predict_all
from scripts.refresh import build_league_forecasts

MODEL = json.loads((ROOT/"data/league-model.json").read_text())


class LeagueTests(unittest.TestCase):
    def test_all_30_teams_and_no_self_matchup(self):
        self.assertEqual(len(TEAMS),30)
        with self.assertRaises(ValueError):
            scenario_features([],[],KNICKS,KNICKS,"2026-11-22")
        for home in TEAMS:
            away = next(team for team in TEAMS if team != home)
            features = scenario_features(historical_games(),[],home,away,"2026-11-22")
            output = prediction(MODEL,features)
            self.assertTrue(0 <= output["home_probability"] <= 1)
            self.assertLess(output["margin_low"],output["home_margin"])
            self.assertLess(output["home_margin"],output["margin_high"])
        with self.assertRaises(ValueError):
            scenario_features([],[],KNICKS,KNICKS,"2026-11-22")

    def test_today_and_future_scores_do_not_enter_scenario(self):
        game={"id":"1","date":"2026-11-22","season":2027,"home":KNICKS,
              "away":1610612738,"home_score":150,"away_score":80}
        original=scenario_features([],[],KNICKS,1610612738,"2026-11-22")
        self.assertEqual(original,scenario_features([],[game],KNICKS,1610612738,"2026-11-22"))
        self.assertNotEqual(original,scenario_features([],[game],KNICKS,1610612738,"2026-11-23"))

    def test_home_and_away_orientation_on_neutral_court(self):
        forward=scenario_features(historical_games(),[],KNICKS,1610612738,"2026-11-22",True)
        reverse=scenario_features(historical_games(),[],1610612738,KNICKS,"2026-11-22",True)
        self.assertEqual(forward[0],0)
        self.assertEqual(forward,[-x for x in reverse])
        home=prediction(MODEL,forward)
        self.assertGreaterEqual(home["home_probability"],0)
        self.assertLessEqual(home["home_probability"],1)
        self.assertGreaterEqual(home["margin_high"],home["margin_low"])

    def test_league_records_before_tip_and_preserves_existing(self):
        game={"id":"new","date":"2026-11-22","season":2027,"home":1610612738,
              "away":1610612747,"state":"scheduled","tip":"2026-11-22T23:00:00+00:00"}
        early=datetime(2026,11,20,0,tzinfo=timezone.utc)
        before=datetime(2026,11,22,20,tzinfo=timezone.utc)
        after=datetime(2026,11,23,0,tzinfo=timezone.utc)
        projections,ledger=build_league_forecasts([], [game],MODEL,{},early,early)
        self.assertIn("new",projections)
        self.assertFalse(ledger)
        _,ledger=build_league_forecasts([], [game],MODEL,{},before,before)
        self.assertIn("new",ledger)
        stored=json.loads(json.dumps(ledger))
        _,unchanged=build_league_forecasts([], [game],MODEL,ledger,before,before)
        self.assertEqual(stored,unchanged)
        self.assertEqual(build_league_forecasts([], [game],MODEL,{},after,after),({},{}))
        self.assertIsNotNone(recorded_pick(game,ledger))
        self.assertIsNone(recorded_pick(dict(game,tip="2026-11-23T23:00:00+00:00"),ledger))

    def test_frozen_development_selection_and_final_predictions(self):
        self.assertEqual(MODEL["selected_win"],min(MODEL["validation"]["wins"],key=lambda n:MODEL["validation"]["wins"][n]["brier"]))
        self.assertEqual(MODEL["selected_margin"],min(MODEL["validation"]["margins"],key=lambda n:MODEL["validation"]["margins"][n]["mae"]))
        self.assertEqual(MODEL["final_test"]["games"],1230)
        self.assertEqual(sum(row["games"] for row in MODEL["team_test"].values()),2460)
        self.assertEqual(MODEL["training_games"],4920)
        self.assertEqual(len(predict_all(MODEL,[1,0,0,0,0])["wins"]),5)

    def test_app_can_build_a_custom_matchup(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/"app.py")).run(timeout=20)
        chooser=next(widget for widget in app.radio if widget.label=="Choose a matchup")
        chooser.set_value("Build a matchup").run()
        self.assertFalse(app.exception)
        away=next(widget for widget in app.selectbox if widget.label=="Away team")
        home=next(widget for widget in app.selectbox if widget.label=="Home team")
        self.assertEqual(len(away.options),30)
        self.assertEqual(len(home.options),29)
        self.assertNotEqual(away.value,home.value)
        next(widget for widget in app.checkbox if widget.label=="Neutral court").check().run()
        self.assertFalse(app.exception)
        next(widget for widget in app.radio if widget.label=="Evaluation period").set_value("Development · 2024–25").run()
        self.assertFalse(app.exception)

    def test_full_schedule_snapshot(self):
        snapshot=json.loads((ROOT/"data/season.json").read_text())
        games=snapshot["league_games"]
        self.assertEqual(len({g["id"] for g in games}),len(games))
        self.assertEqual({team for g in games for team in [g["home"],g["away"]]},set(TEAMS))
        self.assertIsInstance(snapshot["league_projections"],dict)
        ids={game["id"] for game in games}
        self.assertTrue(set(snapshot["league_projections"]).issubset(ids))
        for entry in snapshot["league_projections"].values():
            self.assertGreaterEqual(entry["home_probability"],0)
            self.assertLessEqual(entry["home_probability"],1)
        for key,entry in snapshot["league_ledger"].items():
            self.assertLess(datetime.fromisoformat(entry["generated"]),datetime.fromisoformat(entry["tip"]))


if __name__=="__main__":
    unittest.main()
