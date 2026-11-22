"""League-wide pregame inference using portable, versioned model artifacts."""
import math
from array import array
from datetime import date, datetime
from itertools import groupby

from forecast import CODES, TEAM_IDS, Form

NAMES = [
    "Atlanta Hawks", "Boston Celtics", "Cleveland Cavaliers", "New Orleans Pelicans",
    "Chicago Bulls", "Dallas Mavericks", "Denver Nuggets", "Golden State Warriors",
    "Houston Rockets", "Los Angeles Clippers", "Los Angeles Lakers", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "Brooklyn Nets", "New York Knicks",
    "Orlando Magic", "Indiana Pacers", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Oklahoma City Thunder",
    "Toronto Raptors", "Utah Jazz", "Memphis Grizzlies", "Washington Wizards",
    "Detroit Pistons", "Charlotte Hornets",
]
TEAMS = {TEAM_IDS[code]: {"code": code, "name": name} for code, name in zip(CODES, NAMES)}
WIN_MODELS = ("Home baseline", "Elo", "Logistic regression", "Boosted trees", "Probability blend")
MARGIN_MODELS = ("Home margin baseline", "Ridge regression", "Boosted margin", "Margin blend")


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def linear(model, features):
    return model["intercept"] + sum(w * x for w, x in zip(model["weights"], features))


def tree_value(tree, features):
    node = 0
    while tree["left"][node] != -1:
        node = tree["left"][node] if features[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
    return tree["value"][node]


def boosted(model, features):
    features32 = array("f", features)
    return model["intercept"] + model["learning_rate"] * sum(tree_value(tree, features32) for tree in model["trees"])


def predict_all(artifact, features):
    from forecast import elo_home_probability

    exported = artifact["models"]
    logistic = sigmoid(linear(exported["logistic"], features))
    trees = sigmoid(boosted(exported["classifier"], features))
    ridge = linear(exported["ridge"], features)
    margin_trees = boosted(exported["regressor"], features)
    return {
        "wins": dict(zip(WIN_MODELS, [
            exported["home_rate"], elo_home_probability(features), logistic, trees, (logistic + trees) / 2,
        ])),
        "margins": dict(zip(MARGIN_MODELS, [
            exported["home_margin"], ridge, margin_trees, (ridge + margin_trees) / 2,
        ])),
    }


def prediction(artifact, features):
    estimates = predict_all(artifact, features)
    p = estimates["wins"][artifact["selected_win"]]
    margin = estimates["margins"][artifact["selected_margin"]]
    radius = artifact["margin_band_radius"]
    return {
        "home_probability": p, "home_margin": margin,
        "margin_low": margin - radius, "margin_high": margin + radius,
        "win_model": artifact["selected_win"], "margin_model": artifact["selected_margin"],
        "model_version": artifact["model_version"],
    }


def recorded_pick(game, ledger):
    entry = ledger.get(game["id"])
    if entry and entry.get("tip") == game["tip"] and datetime.fromisoformat(entry["generated"]) < datetime.fromisoformat(game["tip"]):
        return entry
    return None


def scenario_features(history, finals, home, away, day, neutral=False):
    if home not in TEAMS or away not in TEAMS or home == away:
        raise ValueError("Choose two different NBA teams.")
    target = date.fromisoformat(str(day))
    if target.year not in (2026, 2027):
        raise ValueError("Choose a date in the 2026–27 season.")
    form = Form()
    games = sorted([*history, *finals], key=lambda g: (g["date"], g["id"]))
    for _, batch in groupby((game for game in games if date.fromisoformat(game["date"]) < target), key=lambda g: g["date"]):
        prepared = [(game, form.features(game)) for game in batch]
        for game, features in prepared:
            form.update(game, features)
    return form.features({"season": 2027, "date": str(day), "home": home, "away": away, "neutral": neutral})
