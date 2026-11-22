"""League matchup experience and the chronological research report."""
import json
from datetime import date, datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from forecast import FEATURES, ROOT, historical_games
from league import TEAMS, prediction, scenario_features, recorded_pick
from scripts.refresh import ET


def team_name(team):
    return TEAMS[team]["name"]


def tip(value):
    return datetime.fromisoformat(value)



def team_metrics(games, ledger):
    rows = []
    for game in games:
        entry = recorded_pick(game, ledger)
        if not entry or game["state"] != "final":
            continue
        margin = game["home_score"]-game["away_score"]
        rows.append({"correct": (entry["home_probability"] >= .5) == (margin > 0),
                     "brier": (entry["home_probability"]-(margin > 0))**2,
                     "margin_error": abs(entry["home_margin"]-margin)})
    return pd.DataFrame(rows)


def render_forecast(game, forecast, features, artifact, source):
    home, away = game["home"], game["away"]
    a, b = st.columns(2)
    for column, team, venue in [(a, away, "Away"), (b, home, "Home" if not game.get("neutral") else "Neutral designation")]:
        column.image(str(ROOT / "images/teams" / f"{team}.png"), width=64)
        column.subheader(team_name(team))
        column.caption(venue)
    perspective = st.radio("View forecast for", [away, home], format_func=team_name, horizontal=True, key="league-perspective")
    sign = 1 if perspective == home else -1
    p = forecast["home_probability"] if sign == 1 else 1-forecast["home_probability"]
    margin = sign*forecast["home_margin"]
    low, high = ((forecast["margin_low"], forecast["margin_high"]) if sign == 1 else
                 (-forecast["margin_high"], -forecast["margin_low"]))
    winner = home if forecast["home_probability"] >= .5 else away
    first, second, third = st.columns(3)
    first.metric("Projected winner", team_name(winner))
    second.metric(f"{team_name(perspective)} win chance", f"{p:.1%}")
    third.metric("Expected margin", f"{margin:+.1f} points")
    st.progress(p)
    st.caption(f"{team_name(perspective)} minus opponent. Positive favors this team. {source}")
    st.info(f"Empirical margin range · {low:+.1f} to {high:+.1f} points. "
            f"The range covered {artifact['final_band_coverage']:.1%} of final-test game margins. "
            "It reflects individual-game uncertainty.")
    if (p >= .5) != (margin >= 0):
        st.caption("The win and margin models disagree on the favored side. They were trained for different targets.")
    fan = st.radio("Your call", ["No pick yet", team_name(away), team_name(home)], horizontal=True, key=f"league-fan-{game['id']}")
    if fan != "No pick yet":
        st.write("Your pick agrees with the win model." if fan == team_name(winner) else "Your pick differs from the win model.")
    if features is not None:
        st.subheader("Pregame matchup profile")
        st.dataframe(pd.DataFrame({
            "Input": FEATURES, "Home minus away": [round(x,3) for x in features],
            "Meaning": ["1 for home court, 0 at a neutral site", "Elo rating difference / 100",
                        "Smoothed recent scoring margin, points", "Smoothed recent win-rate difference",
                        "Rest-day difference, capped at four days per team"],
        }), hide_index=True, use_container_width=True)
    st.caption(f"Win model · {forecast['win_model']}  |  Margin model · {forecast['margin_model']}. "
               "Injuries, lineups, betting odds, and travel distance are outside the inputs.")


def render_league(season, artifact):
    now = datetime.now(timezone.utc)
    all_games = season.get("league_games", [])
    ledger = season.get("league_ledger", {})
    st.header("Any team. Any matchup.")
    updated = tip(season["updated"])
    st.caption(f"ESPN schedule and finals · Snapshot {updated.astimezone(ET):%b %d, %Y at %I:%M %p ET}")
    if now-updated > timedelta(hours=3):
        st.warning("The latest feed is delayed. These forecasts use the last successful snapshot.")
    mode = st.radio("Choose a matchup", ["Scheduled game", "Build a matchup"], horizontal=True)
    history = historical_games()
    finals = season.get("finals", [])
    if mode == "Scheduled game":
        team = st.selectbox("Filter scheduled games", [None, *sorted(TEAMS,key=team_name)],
                            format_func=lambda value: "All 30 teams" if value is None else team_name(value))
        upcoming = [g for g in all_games if g["state"] == "scheduled" and tip(g["tip"]) > now
                    and (team is None or team in (g["home"], g["away"]))]
        if upcoming:
            game = st.selectbox("Upcoming NBA matchup", upcoming, format_func=lambda g:
                                f"{tip(g['tip']).astimezone(ET):%b %d, %I:%M %p ET} · {g['away_name']} at {g['home_name']}")
            saved = recorded_pick(game, ledger)
            forecast = saved or season.get("league_projections", {}).get(game["id"])
            if forecast:
                render_forecast(game, forecast, None, artifact, "Saved before tipoff." if saved else "Early projection from the latest snapshot.")
            else:
                st.info("This fixture is waiting for the next successful forecast refresh.")
        else:
            st.info("No upcoming fixtures match this snapshot and filter. Use Build a matchup to explore any two teams.")
    else:
        teams = sorted(TEAMS,key=team_name)
        left, right = st.columns(2)
        away = left.selectbox("Away team", teams, index=teams.index(1610612738), format_func=team_name)
        home = right.selectbox("Home team", [team for team in teams if team != away],
                               index=[team for team in teams if team != away].index(1610612752) if away != 1610612752 else 0,
                               format_func=team_name)
        earliest = max(now.astimezone(ET).date(), date.fromisoformat(season["through"]))
        day = st.date_input("Scenario date", value=earliest, min_value=earliest,
                            max_value=max(earliest,date(2027,4,30)))
        neutral = st.checkbox("Neutral court")
        features = scenario_features(history, finals, home, away, day, neutral)
        game = {"id": f"custom-{away}-{home}-{day}-{neutral}", "home": home, "away": away, "neutral": neutral}
        render_forecast(game, prediction(artifact,features), features, artifact,
                        "Hypothetical matchup. Results from the scenario date and later are excluded. "
                        "Future intervening games have not been simulated.")
    st.divider()
    st.subheader("League season tracker")
    selected_team = st.selectbox("Team in season tracker", [None, *sorted(TEAMS,key=team_name)],
                                 format_func=lambda value: "All teams" if value is None else team_name(value))
    state = st.selectbox("Game status", ["All", "Scheduled", "Live", "Final", "Unavailable"])
    visible = [g for g in all_games if (selected_team is None or selected_team in (g["home"],g["away"]))
               and (state=="All" or g["state"]==state.lower())]
    rows = []
    for game in visible:
        saved = recorded_pick(game, ledger)
        forecast = saved or (season.get("league_projections",{}).get(game["id"]) if game["state"]=="scheduled" and tip(game["tip"])>now else None)
        final = game["state"]=="final"
        margin = game["home_score"]-game["away_score"] if final else None
        rows.append({
            "Tipoff ET":tip(game["tip"]).astimezone(ET).strftime("%Y-%m-%d %H:%M"),
            "Away":game["away_name"],"Home":game["home_name"],"Status":game["status"],
            "Home win chance":f"{forecast['home_probability']:.1%}" if forecast else "—",
            "Home margin forecast":round(forecast["home_margin"],1) if forecast else None,
            "Final away–home":f"{game['away_score']}–{game['home_score']}" if final else "—",
            "Actual home margin":margin,
            "Forecast record":"Graded" if saved and final else "Saved pregame" if saved else "Early" if forecast else "No recorded pick",
            "Pick correct":((forecast["home_probability"]>=.5)==(margin>0)) if saved and final else None,
        })
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button("Download NBA game view",frame.to_csv(index=False),"nba-matchups.csv","text/csv")
    else:
        st.info("No games match these filters.")
    live = team_metrics(all_games,ledger)
    one,two,three,four = st.columns(4)
    one.metric("Graded league picks",len(live))
    two.metric("Live W/L accuracy",f"{live.correct.mean():.1%}" if len(live) else "Pending")
    three.metric("Live margin MAE",f"{live.margin_error.mean():.2f} points" if len(live) else "Pending")
    four.metric("Live Brier error",f"{live.brier.mean():.3f}" if len(live) else "Pending")
    st.caption("Only the first saved pick within 24 hours before tipoff is graded. Changed tipoffs invalidate old picks. "
               "League-wide tracking begins with this expansion. Earlier final scores have no retrospective forecast record.")


def render_research(artifact):
    st.header("What improved on the baselines?")
    st.write("Development: fit on 2022–23 and 2023–24, evaluate on 2024–25. "
             "Choose the win model by Brier score and the margin model by MAE. "
             "Final test: refit using seasons through 2024–25 and evaluate 2025–26 once under that selection rule.")
    stage = st.radio("Evaluation period", ["Final test · 2025–26", "Development · 2024–25"],horizontal=True)
    report = artifact["final_test"] if stage.startswith("Final") else artifact["validation"]
    wins = pd.DataFrame([{"Model":name,"Accuracy":row["accuracy"],"Brier error":row["brier"],"Log loss":row["log_loss"],
                          "Chosen on development":name==artifact["selected_win"]} for name,row in report["wins"].items()])
    margins = pd.DataFrame([{"Model":name,"MAE (points)":row["mae"],"RMSE (points)":row["rmse"],
                             "Chosen on development":name==artifact["selected_margin"]} for name,row in report["margins"].items()])
    a,b,c = st.columns(3)
    a.metric("Test games",report["games"])
    b.metric("Selected win model",artifact["selected_win"])
    c.metric("Selected margin model",artifact["selected_margin"])
    st.subheader("Winner predictions")
    st.dataframe(wins,hide_index=True,use_container_width=True)
    st.subheader("Point-margin predictions")
    st.dataframe(margins,hide_index=True,use_container_width=True)
    chart = alt.Chart(margins).mark_bar(color="#ff8547",cornerRadius=4).encode(
        x=alt.X("MAE (points):Q",title="Average absolute margin error in points · lower is better"),
        y=alt.Y("Model:N",sort="-x",title=None),tooltip=["Model",alt.Tooltip("MAE (points):Q",format=".2f")])
    st.altair_chart(chart,use_container_width=True)
    st.write("The fixed 50/50 probability blend had a slightly lower final-test Brier score than logistic regression, "
             "but logistic regression was selected on the earlier development season. "
             "The final-test result did not change that selection. Extra complexity did not improve the margin forecast.")
    intervals = artifact["paired_bootstrap"]
    st.caption("Date-block bootstrap, 500 resamples, compares the selected models with their constant home baselines. "
               f"Brier difference {intervals['brier']['difference']:+.4f}, 95% bootstrap interval "
               f"[{intervals['brier']['low']:+.4f}, {intervals['brier']['high']:+.4f}]. "
               f"MAE difference {intervals['mae']['difference']:+.2f} points, interval "
               f"[{intervals['mae']['low']:+.2f}, {intervals['mae']['high']:+.2f}]. "
               "Games across dates remain correlated, so these intervals are descriptive.")
    st.subheader("Team-level final test")
    st.dataframe(pd.DataFrame([{"Team":team_name(int(team)),"Games":row["games"],"W/L accuracy":row["accuracy"],
                               "Margin MAE":row["mae"]} for team,row in artifact["team_test"].items()]),hide_index=True,use_container_width=True)
    st.caption("Each game contributes to two team rows. The league headline counts each game once.")
    st.subheader("Research poster")
    poster = ROOT/"research/nba-matchup-poster.pdf"
    preview = ROOT/"research/nba-matchup-poster.png"
    if preview.exists():
        st.image(str(preview),caption="NBA Matchup Lab · Question, method, results, and limitations")
    if poster.exists():
        st.download_button("Download research poster",poster.read_bytes(),"nba-matchup-poster.pdf","application/pdf")
    st.download_button("Download final-test predictions",(ROOT/"data/league-test-predictions.csv").read_bytes(),
                       "nba-final-test-predictions.csv","text/csv")
    st.download_button("Download model report",json.dumps(artifact,indent=2),"nba-model-report.json","application/json")
    st.write("Inputs use earlier dates only. No injuries, current lineups, odds, player availability, or travel estimates. "
             "Only two full later seasons were evaluated, and historical data already existed in this repository. "
             "The empirical margin range used development residuals and covered "
             f"{artifact['final_band_coverage']:.1%} of final-test outcomes against an 80% target. "
             "A point estimate can miss an individual game's result by a large margin.")
