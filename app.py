"""Knicks pregame forecasts and prediction results."""
import base64
import json
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import streamlit as st

from forecast import FEATURES, KNICKS, ROOT
from scripts.refresh import ET

ORANGE = "#ff8547"
BLUE = "#79abff"


def time(value):
    return datetime.fromisoformat(value).astimezone(ET)


def matchup(game):
    home = game["home"] == KNICKS
    return game["away_name"] if home else game["home_name"], "Neutral site" if game["neutral"] else ("Home" if home else "Away")


def actual(game):
    if game["state"] != "final":
        return None, "—"
    first, second = ("home", "away") if game["home"] == KNICKS else ("away", "home")
    a, b = game[f"{first}_score"], game[f"{second}_score"]
    return "W" if a > b else "L", f"{a}–{b}"


def pick(probability):
    return "W" if probability >= 0.5 else "L"


def saved_pick(game, ledger):
    entry = ledger.get(game["id"])
    if entry and entry["tip"] == game["tip"] and time(entry["generated"]) < time(game["tip"]):
        return entry
    return None


def grade_picks(graded):
    """Measure only picks actually recorded before tipoff."""
    correct, errors = 0, []
    for game, entry in graded:
        outcome, _ = actual(game)
        probability = entry["probability"]
        correct += outcome == pick(probability)
        errors.append((probability - (outcome == "W")) ** 2)
    return correct, sum(errors) / len(errors) if errors else None


FILM = {
    "The Knicks are champions · Team all-access": "https://www.youtube.com/watch?v=rsHQ96CLq70",
    "2026 Finals trophy ceremony · ESPN": "https://www.youtube.com/watch?v=ITwzTEeiVeA",
}


def render():
    st.set_page_config(page_title="Knicks Win Predictor", page_icon="🏀", layout="wide")
    st.markdown("""
        <style>
        .stApp {background: #081221;}
        .knicks-hero {display:flex; align-items:center; gap:1.45rem; flex-wrap:wrap;
          border:1px solid #345788; border-left:5px solid #ff8547; border-radius:20px;
          padding:1.6rem 2rem;
          background:linear-gradient(100deg,#10366dfc 10%,#111e35f2 52%,#111e355c),
          var(--garden-photo) center 50% / cover #111e35;
          margin-bottom:1rem}
        .knicks-hero img {width:118px;height:105px;object-fit:contain}
        .knicks-hero h1 {font-size:clamp(2rem,5vw,3.7rem);line-height:1.05;margin:0;padding:0}
        .knicks-hero p {margin:.5rem 0 0;color:#cddcf5}
        .knicks-nav {display:flex; flex-wrap:wrap; gap:.45rem 1.2rem; margin:.15rem 0 .6rem}
        .knicks-nav a {color:#ffc4a1; text-underline-offset:.2rem}
        div[data-testid="stMetric"] {background:#101f36;border:1px solid #2a4262;
          border-radius:13px;padding:.7rem 1rem}
        </style>
        """, unsafe_allow_html=True)
    logo = base64.b64encode((ROOT / "images/knicks.svg").read_bytes()).decode()
    garden = base64.b64encode((ROOT / "images/garden-knicks-game.jpg").read_bytes()).decode()
    st.markdown(f"""
        <div class="knicks-hero" style="--garden-photo:url(data:image/jpeg;base64,{garden})"><img src="data:image/svg+xml;base64,{logo}" alt="New York Knicks logo">
        <div><h1>Knicks Win Predictor</h1>
        <p>2026–27 regular season · Pregame win probabilities and final results</p></div></div>
        <nav class="knicks-nav" aria-label="Jump to a section">
          <a href="#knicks-film-room">Film room</a>
          <a href="#game-preview">Game preview</a>
          <a href="#season-tracker">Season tracker</a>
          <a href="#model-performance">Model performance</a>
        </nav>
        """, unsafe_allow_html=True)

    season = json.loads((ROOT / "data/season.json").read_text())
    model = json.loads((ROOT / "data/model.json").read_text())
    games, ledger = season["knicks"], season["ledger"]
    now = datetime.now(timezone.utc)
    st.caption(f"Schedule and scores from ESPN · Updated {time(season['updated']):%b %d at %I:%M %p ET}")
    if now - time(season["updated"]) > timedelta(hours=3):
        st.warning("The latest feed update is delayed. Results and projections below come from the last successful refresh.")
    finals = [g for g in games if g["state"] == "final"]
    upcoming = [g for g in games if g["state"] == "scheduled" and time(g["tip"]) > now]
    graded = [(g, entry) for g in finals if (entry := saved_pick(g, ledger))]
    correct, live_brier = grade_picks(graded)
    wins = sum(actual(g)[0] == "W" for g in finals)

    with st.container():
        st.header("Knicks Film Room", anchor="knicks-film-room")
        st.write("I grew up watching the Knicks with my dad. These are two moments from the title run I keep coming back to.")
        for column, (title, url) in zip(st.columns(2), FILM.items()):
            with column:
                st.subheader(title)
                st.video(url)
                st.link_button("Watch on YouTube", url)

    with st.container():
        st.header("Game preview", anchor="game-preview")
        if upcoming:
            selected = st.selectbox("Choose an upcoming game", upcoming, format_func=lambda g:
                f"{time(g['tip']):%b %d} · {matchup(g)[0]} · {matchup(g)[1]}")
            name, venue = matchup(selected)
            st.subheader(f"Knicks vs. {name}")
            st.caption(f"{time(selected['tip']):%A, %B %d · %I:%M %p ET} · {venue}")
            saved = saved_pick(selected, ledger)
            forecast = saved or season["projections"].get(selected["id"])
            if forecast:
                p = forecast["probability"]
                a, b, c = st.columns(3)
                a.metric("Model pick", "W · Win" if pick(p) == "W" else "L · Loss")
                b.metric("Win probability", f"{p:.1%}")
                c.metric("Loss probability", f"{1-p:.1%}")
                st.progress(p)
                st.caption("Lean" if abs(p - .5) < .10 else "Stronger lean")
                fan = st.radio("Make your own call", ["No pick", "W · Win", "L · Loss"],
                               horizontal=True, key=f"fan-{selected['id']}")
                if fan != "No pick":
                    st.write("You and the model agree." if fan[0] == pick(p) else "You and the model disagree.")
                st.caption("Recorded pregame pick." if saved else "Early projection. It changes as more games are played.")
                st.subheader("What drives this pick?")
                factors = pd.DataFrame({"Factor": FEATURES, "Contribution": forecast["factors"]})
                chart = alt.Chart(factors).mark_bar(cornerRadius=4).encode(
                    y=alt.Y("Factor:N", sort=list(FEATURES), title=None),
                    x=alt.X("Contribution:Q", title="Contribution on the model’s log-odds scale"),
                    color=alt.condition(alt.datum.Contribution >= 0, alt.value(ORANGE), alt.value(BLUE)),
                    tooltip=["Factor", alt.Tooltip("Contribution:Q", format=".3f")],
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)
                st.caption("Orange favors a Knicks win. Blue favors an opponent win. Injuries and lineup changes are outside this model.")
            else:
                st.info("A projection will appear after the next successful schedule refresh.")
        else:
            st.info("There are no future tipoffs in the latest schedule.")
        st.divider()
        one, two, three = st.columns(3)
        one.metric("Knicks actual record", f"{wins}–{len(finals)-wins}")
        two.metric("Correct recorded picks", f"{correct}/{len(graded)}" if graded else "Pending")
        three.metric("Last final", actual(finals[-1])[1] if finals else "Pending")

    with st.container():
        st.header("Season tracker", anchor="season-tracker")
        filters, arena = st.columns([3, 1])
        arena.image(str(ROOT / "images/garden-interior.png"), caption="Inside Madison Square Garden")
        view = filters.selectbox("Show games", ["All", "Upcoming", "Final"])
        opponent = filters.selectbox("Opponent", ["All opponents", *sorted({matchup(game)[0] for game in games})])
        result_filter = filters.selectbox("Result", ["Any result", "Wins", "Losses"])
        graded_filter = filters.checkbox("Show only graded games")
        active_ids = {g["id"] for g in upcoming}
        rows = []
        for game in games:
            if view == "Upcoming" and game["id"] not in active_ids or view == "Final" and game["state"] != "final":
                continue
            saved = saved_pick(game, ledger)
            forecast = saved or (season["projections"].get(game["id"]) if game["id"] in active_ids else None)
            p = forecast["probability"] if forecast else None
            outcome, score = actual(game)
            name, venue = matchup(game)
            if opponent != "All opponents" and name != opponent:
                continue
            if result_filter != "Any result" and outcome != result_filter[0]:
                continue
            if graded_filter and not (saved and outcome):
                continue
            rows.append({
                "Tipoff (ET)": time(game["tip"]).strftime("%b %d · %I:%M %p"), "Opponent": name, "Court": venue,
                "Pick": pick(p) if p is not None else "—", "Win chance": f"{p:.1%}" if p is not None else "—",
                "Score (NYK–OPP)": score, "Result": outcome or game["status"],
                "Forecast": ("Correct" if outcome == pick(p) else "Miss") if saved and outcome
                else ("Recorded" if saved else ("Early" if forecast else "—")),
            })
        if rows:
            visible = pd.DataFrame(rows)
            st.dataframe(visible, hide_index=True, use_container_width=True)
            st.download_button("Download this view", visible.to_csv(index=False), "knicks-games.csv", "text/csv")
        else:
            st.info("No games match these filters.")
        st.caption("Games completed before the first recorded pick show the final score only. TBD Cup matchups appear once teams are assigned.")

    with st.container():
        st.header("Model performance", anchor="model-performance")
        backtest = model["backtest"]
        a, b, c, d = st.columns(4)
        a.metric("Model W/L accuracy", f"{backtest['model']['accuracy']:.1%}")
        b.metric("Home-court baseline", f"{backtest['baseline']['accuracy']:.1%}")
        c.metric("Elo W/L accuracy", f"{backtest['elo']['accuracy']:.1%}")
        d.metric("Held-out Knicks games", backtest["model"]["games"])
        st.write("The backtest predicts 2024–25 and 2025–26 Knicks games with a model fitted only on earlier seasons. Each prediction uses league results from earlier dates.")
        st.dataframe(pd.DataFrame([
            {"Season": f"{f['season']-1}–{str(f['season'])[-2:]}", "Games": f["model"]["games"],
             "Model accuracy": f"{f['model']['accuracy']:.1%}", "Baseline accuracy": f"{f['baseline']['accuracy']:.1%}",
             "Elo accuracy": f"{f['elo']['accuracy']:.1%}", "Model Brier": f"{f['model']['brier']:.3f}",
             "Baseline Brier": f"{f['baseline']['brier']:.3f}", "Elo Brier": f"{f['elo']['brier']:.3f}"}
            for f in backtest["folds"]
        ]), hide_index=True, use_container_width=True)
        st.caption("Brier score measures probability error. Lower is better. The home-court baseline uses the historical league home-win rate. The Elo baseline uses team strength and venue without recent form or rest.")
        st.subheader("This season’s saved picks")
        a, b, c = st.columns(3)
        a.metric("Graded picks", len(graded))
        b.metric("W/L hit rate", f"{correct / len(graded):.1%}" if graded else "Pending")
        c.metric("Live Brier score", f"{live_brier:.3f}" if live_brier is not None else "Pending")
        st.write(f"Live season tracking starts with the first saved pregame pick. So far, {correct} of {len(graded)} graded picks were correct.")


if __name__ == "__main__":
    render()
