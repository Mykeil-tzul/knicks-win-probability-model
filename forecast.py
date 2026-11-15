"""Pregame features and season-by-season NBA backtesting."""
import csv
from collections import defaultdict, deque
from datetime import date
from itertools import groupby
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
KNICKS = 1610612752
# NBA franchise IDs in their consecutive ID order.
CODES = "ATL BOS CLE NO CHI DAL DEN GS HOU LAC LAL MIA MIL MIN BKN NY ORL IND PHI PHX POR SAC SA OKC TOR UTA MEM WSH DET CHA".split()
TEAM_IDS = {code: 1610612737 + i for i, code in enumerate(CODES)}
TEAM_IDS.update({alias: TEAM_IDS[code] for alias, code in {
    "UTAH": "UTA", "GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "WAS": "WSH", "BRK": "BKN"
}.items()})
FEATURES = ("Home court", "Team strength", "Recent point margin", "Recent win rate", "Rest")


def historical_games():
    with (ROOT / "data/history.csv").open() as source:
        return [
            {**row, "neutral": row["neutral"] == "True", **{key: int(row[key]) for key in ("home", "away", "home_score", "away_score", "season")}}
            for row in csv.DictReader(source)
        ]


def elo_home_probability(features):
    home_edge, elo_gap, *_ = features
    return 1 / (1 + 10 ** (-(100 * elo_gap + 65 * home_edge) / 400))


class Form:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.margins = defaultdict(lambda: deque(maxlen=10))
        self.last = {}
        self.season = None

    def start_season(self, season):
        if self.season == season:
            return
        for team in self.elo:
            self.elo[team] = 1500 + 0.75 * (self.elo[team] - 1500)
        self.margins.clear()
        self.last.clear()
        self.season = season

    def features(self, game):
        self.start_season(game["season"])
        home, away = game["home"], game["away"]
        day = date.fromisoformat(game["date"])

        def recent(team):
            margins = self.margins[team]
            # Five league-average pseudo-games stabilize the opening weeks.
            return sum(margins) / (len(margins) + 5), (sum(x > 0 for x in margins) + 2.5) / (len(margins) + 5)

        def rest(team):
            return min(4, (day - self.last[team]).days) if team in self.last else 4

        hm, hw = recent(home)
        am, aw = recent(away)
        return [int(not game.get("neutral")), (self.elo[home] - self.elo[away]) / 100,
                hm - am, hw - aw, rest(home) - rest(away)]

    def update(self, game, features):
        home, away = game["home"], game["away"]
        margin = game["home_score"] - game["away_score"]
        expected = elo_home_probability(features)
        change = 20 * (int(margin > 0) - expected)
        self.elo[home] += change
        self.elo[away] -= change
        self.margins[home].append(margin)
        self.margins[away].append(-margin)
        self.last[home] = self.last[away] = date.fromisoformat(game["date"])


def pregame_rows(games):
    """Calculate a full day's features before consuming any score from that day."""
    form = Form()
    ordered = sorted(games, key=lambda g: (g["date"], g["id"]))
    for _, group in groupby(ordered, key=lambda g: g["date"]):
        day = [(game, form.features(game)) for game in group]
        yield from day
        for game, features in day:
            if game.get("state", "final") == "final":
                form.update(game, features)


def is_knicks(game):
    return KNICKS in (game["home"], game["away"])


def knicks_probability(game, probability):
    return probability if game["home"] == KNICKS else 1 - probability


def result(game):
    return int(game["home_score"] > game["away_score"])


def metrics(outcomes, probabilities):
    return {
        "accuracy": mean(int((p >= 0.5) == y) for y, p in zip(outcomes, probabilities)),
        "brier": mean((p - y) ** 2 for y, p in zip(outcomes, probabilities)),
        "games": len(outcomes),
    }


def train(history):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = list(pregame_rows(history))

    def fit(train_rows):
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return estimator.fit([x for _, x in train_rows], [result(g) for g, _ in train_rows])

    labels, predictions, baselines, elo_predictions = [], [], [], []
    folds = []
    for season in (2025, 2026):
        earlier = [(g, x) for g, x in rows if g["season"] < season]
        later = [(g, x) for g, x in rows if g["season"] == season and is_knicks(g)]
        estimator = fit(earlier)
        home_rate = mean(result(g) for g, _ in earlier)
        actual = [result(g) if g["home"] == KNICKS else 1 - result(g) for g, _ in later]
        probs = [knicks_probability(g, p) for (g, _), (_, p) in zip(later, estimator.predict_proba([x for _, x in later]))]
        baseline = [knicks_probability(g, home_rate) for g, _ in later]
        elo = [knicks_probability(g, elo_home_probability(x)) for g, x in later]
        elo_predictions.extend(elo)
        labels.extend(actual)
        predictions.extend(probs)
        baselines.extend(baseline)
        folds.append({"season": season, "model": metrics(actual, probs), "baseline": metrics(actual, baseline),
                      "elo": metrics(actual, elo)})
    estimator = fit(rows)
    scaler, regression = estimator.steps[0][1], estimator.steps[1][1]
    return {
        "weights": (regression.coef_[0] / scaler.scale_).tolist(),
        "intercept": float(regression.intercept_[0] - sum(regression.coef_[0] * scaler.mean_ / scaler.scale_)),
        "training_games": len(rows),
        "backtest": {"model": metrics(labels, predictions), "baseline": metrics(labels, baselines),
                     "elo": metrics(labels, elo_predictions), "folds": folds},
    }
