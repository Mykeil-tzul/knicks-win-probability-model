"""Refresh ESPN schedules, finals and the timestamped pregame forecast ledger."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from forecast import ROOT, TEAM_IDS, historical_games, is_knicks, knicks_probability, pregame_rows

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
ET = ZoneInfo("America/New_York")
START = date(2026, 10, 20)
END = date(2027, 4, 30)
SNAPSHOT = ROOT / "data/season.json"


def fetch(path):
    request = Request(ESPN + path, headers={"User-Agent": "StatMatch-KnicksPortfolio/1.0"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=12) as response:
                return json.load(response)["events"]
        except (OSError, ValueError, KeyError):
            if attempt:
                raise


def parse(event):
    season = event["season"]
    phase = event.get("seasonType", {}).get("type", season.get("type"))
    if (season["year"], phase) != (2027, 2):
        return None
    competition = event["competitions"][0]
    teams = {entry["homeAway"]: entry for entry in competition["competitors"]}
    if set(teams) != {"home", "away"}:
        return None
    codes = [teams[side]["team"]["abbreviation"] for side in ("home", "away")]
    if any(code not in TEAM_IDS for code in codes):
        if all(teams[side]["team"].get("id") for side in teams) and competition["status"]["type"]["completed"]:
            raise ValueError(f"Unknown teams in final: {codes}")
        return None
    tip = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    status = competition["status"]["type"]
    state = {"pre": "scheduled", "in": "live", "post": "unavailable"}.get(status["state"], "unavailable")
    if status["name"] in {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_DELAYED", "STATUS_SUSPENDED"}:
        state = "unavailable"
    if state == "scheduled" and not competition.get("timeValid", event.get("timeValid", True)):
        state = "unavailable"
    game = {
        "id": event["id"], "season": 2027, "tip": tip.isoformat(), "date": tip.astimezone(ET).date().isoformat(),
        "home": TEAM_IDS[codes[0]], "away": TEAM_IDS[codes[1]],
        "home_name": teams["home"]["team"]["displayName"], "away_name": teams["away"]["team"]["displayName"],
        "home_code": codes[0], "away_code": codes[1], "neutral": bool(competition.get("neutralSite")),
        "state": state, "status": status["description"],
    }
    if status["name"] in {"STATUS_FINAL", "STATUS_FINAL_OVERTIME"} and status["completed"]:
        for side, entry in teams.items():
            score = entry["score"]
            game[f"{side}_score"] = int(score["value"] if isinstance(score, dict) else score)
        if game["home_score"] == game["away_score"] or min(game["home_score"], game["away_score"]) <= 0:
            raise ValueError(f"Invalid final score for {game['id']}")
        game["state"] = "final"
    return game


def build_forecasts(history, games, model, old_ledger, check_started, generated):
    ledger = dict(old_ledger)
    projections = {}
    for game, features in pregame_rows([*history, *games]):
        if game["season"] != 2027 or not is_knicks(game):
            continue
        tip = datetime.fromisoformat(game["tip"])
        if game["state"] != "scheduled" or tip <= generated:
            continue
        logit = model["intercept"] + sum(w * x for w, x in zip(model["weights"], features))
        probability = knicks_probability(game, 1 / (1 + math.exp(-logit)))
        pick = {
            "probability": probability, "generated": generated.isoformat(), "tip": game["tip"],
            "factors": [round(w * x * (1 if game["home"] == TEAM_IDS["NY"] else -1), 4)
                        for w, x in zip(model["weights"], features)],
        }
        projections[game["id"]] = pick
        if tip - check_started <= timedelta(hours=24):
            ledger.setdefault(game["id"], pick)
    return projections, ledger


def write_snapshot(payload):
    with NamedTemporaryFile(mode="w", dir=SNAPSHOT.parent, delete=False) as temporary:
        json.dump(payload, temporary, separators=(",", ":"))
        temporary.write("\n")
    Path(temporary.name).replace(SNAPSHOT)


def main():
    started = datetime.now(timezone.utc)
    local_day = started.astimezone(ET).date()
    if local_day > END + timedelta(days=5):
        print("The 2026–27 regular-season refresh window has ended.")
        return
    day = min(local_day, END)
    current = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    last = date.fromisoformat(current["through"]) if current else START
    first = max(START, last - timedelta(days=4))
    dates = [first + timedelta(days=offset) for offset in range((day - first).days + 1)]
    paths = ["teams/ny/schedule?season=2027", *[f"scoreboard?dates={d:%Y%m%d}" for d in dates]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(fetch, paths))
    schedule = {game["id"]: game for event in responses[0] if (game := parse(event))}
    if len(schedule) < 75 or any(not is_knicks(game) for game in schedule.values()):
        raise ValueError("Knicks feed returned an incomplete schedule")
    finals = {game["id"]: game for game in current.get("finals", [])}
    for event in (event for response in responses[1:] for event in response):
        game = parse(event)
        if game and game["state"] == "final":
            finals[game["id"]] = game
    for game in schedule.values():
        if game["state"] == "final":
            finals[game["id"]] = game
    games = {**schedule, **finals}
    generated = datetime.now(timezone.utc)
    model = json.loads((ROOT / "data/model.json").read_text())
    projections, ledger = build_forecasts(
        historical_games(), games.values(), model, current.get("ledger", {}), started, generated,
    )
    write_snapshot({
        "updated": generated.isoformat(), "through": day.isoformat(), "finals": sorted(finals.values(), key=lambda g: g["tip"]),
        "knicks": sorted((game for game in games.values() if is_knicks(game)), key=lambda g: g["tip"]),
        "projections": projections, "ledger": ledger,
    })
    print(f"Published {len(schedule)} Knicks games, {len(finals)} league finals, and {len(projections)} upcoming projections.")


if __name__ == "__main__":
    main()
