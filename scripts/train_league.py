"""Select models on 2024–25, test on 2025–26, then export portable inference."""
import csv
import hashlib
import json
import math

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecast import ROOT, historical_games, pregame_rows, result
from league import TEAMS, WIN_MODELS, MARGIN_MODELS, predict_all, prediction

CONFIG = {"n_estimators": 100, "learning_rate": .04, "max_depth": 2, "min_samples_leaf": 30, "random_state": 42}


def export_linear(pipeline):
    scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1]
    weights = np.asarray(estimator.coef_).reshape(-1) / scaler.scale_
    return {"weights": weights.tolist(), "intercept": float(np.asarray(estimator.intercept_).reshape(-1)[0] - weights @ scaler.mean_)}


def export_boosted(estimator, classification=False):
    prior = estimator.init_.class_prior_[1] if classification else None
    intercept = math.log(prior / (1-prior)) if classification else float(estimator.init_.constant_[0, 0])
    trees = []
    for fitted in estimator.estimators_.reshape(-1):
        tree = fitted.tree_
        trees.append({
            "left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
        })
    return {"intercept": intercept, "learning_rate": estimator.learning_rate, "trees": trees}


def fit(rows):
    x = np.asarray([features for _, features in rows])
    y = np.asarray([result(game) for game, _ in rows])
    margins = np.asarray([game["home_score"] - game["away_score"] for game, _ in rows])
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(x, margins)
    classifier = GradientBoostingClassifier(**CONFIG).fit(x, y)
    regressor = GradientBoostingRegressor(**CONFIG).fit(x, margins)
    artifact = {"models": {
        "home_rate": float(y.mean()), "home_margin": float(margins.mean()),
        "logistic": export_linear(logistic), "ridge": export_linear(ridge),
        "classifier": export_boosted(classifier, True), "regressor": export_boosted(regressor),
    }}
    # Check the plain-Python export through the actual inference path.
    exported = [predict_all(artifact, features) for _, features in rows]
    for label, expected, kind in [
        ("Logistic regression", logistic.predict_proba(x)[:, 1], "wins"),
        ("Boosted trees", classifier.predict_proba(x)[:, 1], "wins"),
        ("Ridge regression", ridge.predict(x), "margins"),
        ("Boosted margin", regressor.predict(x), "margins"),
    ]:
        np.testing.assert_allclose([row[kind][label] for row in exported], expected, atol=1e-8)
    return artifact


def evaluate(rows, artifact):
    predictions = [predict_all(artifact, x) for _, x in rows]
    y = np.asarray([result(game) for game, _ in rows])
    margin = np.asarray([game["home_score"]-game["away_score"] for game, _ in rows])
    wins, margins = {}, {}
    for name in WIN_MODELS:
        p = np.asarray([row["wins"][name] for row in predictions])
        clipped = np.clip(p, 1e-12, 1-1e-12)
        wins[name] = {"accuracy": float(np.mean((p >= .5) == y)), "brier": float(np.mean((p-y)**2)),
                      "log_loss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped)))}
    for name in MARGIN_MODELS:
        estimated = np.asarray([row["margins"][name] for row in predictions])
        margins[name] = {"mae": float(np.mean(np.abs(estimated-margin))), "rmse": float(np.sqrt(np.mean((estimated-margin)**2)))}
    return {"games": len(rows), "wins": wins, "margins": margins}, predictions


def bootstrap_difference(rows, predictions, selected, task, repeats=500):
    groups = {}
    for (game, _), estimates in zip(rows, predictions):
        if task == "wins":
            actual = result(game)
            difference = (estimates["wins"][selected]-actual)**2 - (estimates["wins"]["Home baseline"]-actual)**2
        else:
            actual = game["home_score"]-game["away_score"]
            difference = abs(estimates["margins"][selected]-actual) - abs(estimates["margins"]["Home margin baseline"]-actual)
        groups.setdefault(game["date"], []).append(difference)
    totals = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(groups), len(groups))
        samples.append(float(totals[indices].sum()/counts[indices].sum()))
    return {"difference": float(totals.sum()/counts.sum()), "low": float(np.quantile(samples,.025)),
            "high": float(np.quantile(samples,.975)), "resamples": repeats, "unit": "game date", "seed": 42}


def train_league():
    rows = list(pregame_rows(historical_games()))
    development = [(g,x) for g,x in rows if g["season"] == 2025]
    earlier = [(g,x) for g,x in rows if g["season"] < 2025]
    validation_artifact = fit(earlier)
    validation, development_predictions = evaluate(development, validation_artifact)
    # Selection is frozen before evaluating any final-test scores.
    selected_win = min(WIN_MODELS, key=lambda name: validation["wins"][name]["brier"])
    selected_margin = min(MARGIN_MODELS, key=lambda name: validation["margins"][name]["mae"])
    residuals = [abs(row["margins"][selected_margin]-(g["home_score"]-g["away_score"]))
                 for (g,_),row in zip(development, development_predictions)]
    radius = float(np.quantile(residuals,.8,method="higher"))
    final_rows = [(g,x) for g,x in rows if g["season"] == 2026]
    final_model = fit([(g,x) for g,x in rows if g["season"] < 2026])
    final, final_predictions = evaluate(final_rows, final_model)
    coverage = sum(abs(row["margins"][selected_margin]-(g["home_score"]-g["away_score"])) <= radius
                   for (g,_),row in zip(final_rows, final_predictions))/len(final_rows)
    artifact = fit(rows)
    fingerprint = hashlib.sha256((ROOT/"data/history.csv").read_bytes()).hexdigest()
    artifact.update({
        "schema_version": 1, "model_version": fingerprint[:12]+"-league-v1",
        "history_sha256": fingerprint, "training_games": len(rows),
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "config": CONFIG,
        "selected_win": selected_win, "selected_margin": selected_margin,
        "selection": "Lowest development Brier score for wins and development MAE for margins. Fixed equal-weight blends.",
        "margin_band_radius": radius, "band_target": .8, "final_band_coverage": coverage,
        "validation": validation, "final_test": final,
        "paired_bootstrap": {
            "brier": bootstrap_difference(final_rows, final_predictions, selected_win,"wins"),
            "mae": bootstrap_difference(final_rows, final_predictions, selected_margin,"margins"),
        },
        "team_test": {},
    })
    for team in TEAMS:
        subset = [i for i,(g,_) in enumerate(final_rows) if team in (g["home"],g["away"])]
        correct = []; errors = []
        for i in subset:
            game = final_rows[i][0]; row = final_predictions[i]
            probability = row["wins"][selected_win] if game["home"] == team else 1-row["wins"][selected_win]
            won = result(game) if game["home"] == team else 1-result(game)
            correct.append((probability>=.5)==won)
            errors.append(abs(row["margins"][selected_margin]-(game["home_score"]-game["away_score"])))
        artifact["team_test"][str(team)] = {"games": len(subset), "accuracy": float(np.mean(correct)), "mae": float(np.mean(errors))}
    output = ROOT/"data/league-model.json"
    output.write_text(json.dumps(artifact,indent=2)+"\n")
    with (ROOT/"data/league-test-predictions.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=["id","date","home","away","actual_home_margin",*WIN_MODELS,*MARGIN_MODELS])
        writer.writeheader()
        for (game,_),row in zip(final_rows,final_predictions):
            writer.writerow({**{k:game[k] for k in ("id","date","home","away")},
                             "actual_home_margin":game["home_score"]-game["away_score"],**row["wins"],**row["margins"]})
    print(json.dumps({key:artifact[key] for key in ("selected_win","selected_margin","validation","final_test","paired_bootstrap","final_band_coverage")},indent=2))
    return artifact


if __name__ == "__main__":
    train_league()
