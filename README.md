# Knicks Win Predictor

**[Open the live Streamlit dashboard](https://knicks-win-probability-model-bymt.streamlit.app/)**

I built this app to answer one question before each Knicks game in the 2026–27 regular season: how likely are they to win? Pick a matchup to see the model's W/L call, its win and loss probabilities, and the factors that moved the prediction. The season tracker puts those picks next to the actual scores.

## Why I built this

I was born and raised in New York City, and my family has rooted for the Knicks for as long as I can remember. My dad has been a die-hard Knicks fan my whole life. Seeing how happy he was when we won the championship last season is something I’ll always remember. Building this app felt like a fun way to follow the next season together.

## How it works

The model is a logistic regression trained on 4,920 NBA regular-season games from 2022–23 through 2025–26. It learns from the entire league so it can compare the Knicks with the team they're about to play. It uses home court, the gap in team strength (Elo), recent scoring margins, recent win rates, and rest. The recent-form inputs use up to 10 games, with a small league-average starting point so the first few games of a season have less influence.

Features use results from earlier dates only. The model does not use the game score it's trying to predict, betting lines, injuries, or lineups. A W pick means the Knicks' estimated win probability is at least 50%.

## How I tested it

I fit the model on seasons before each test season and predicted the Knicks' games in that later season. I compared it with a baseline that uses the historical league home-win rate for every game.

| Knicks games in the test | Games | Model W/L accuracy | Baseline W/L accuracy |
| --- | ---: | ---: | ---: |
| 2024–25 | 82 | 67.1% | 53.7% |
| 2025–26 | 82 | 69.5% | 61.0% |
| Combined | 164 | 68.3% | 57.3% |

The combined Brier score was 0.195 for the model and 0.245 for the baseline. This measures how far the probabilities were from actual outcomes, with lower scores indicating smaller errors. These are historical results. The app tracks real 2026–27 picks separately.

## Game-day updates

GitHub Actions is configured to refresh the ESPN schedule and league scores hourly. The dashboard reads the last published snapshot and shows its timestamp. Future fixtures show early projections based on results available at refresh time. The first successfully saved pick within 24 hours of tipoff is preserved and compared with the final score. An altered tipoff invalidates that saved pick for grading. Games played before tracking started show final scores only. Unassigned NBA Cup matchups appear when ESPN lists the teams. UTC timestamps are displayed in Eastern Time.

## Run it locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m unittest discover -s tests
```

To reproduce the trained weights, install `requirements-training.txt` and run `python -m scripts.train`. Run `python -m scripts.refresh` to fetch the current season. Set up Streamlit Community Cloud with this repo, branch `main`, and entry point `app.py`. The original `model.py` path also launches the dashboard.

Historical results were aggregated from the [NBA Cloud Data Platform](https://github.com/Mykeil-tzul/wonderworlds/tree/373000e259ef1c68681160fa04a329004e43606d) player game logs, sourced from the [nba_data archive](https://github.com/llimllib/nba_data). Current fixtures and scores come from [ESPN](https://www.espn.com/nba/team/schedule/_/name/ny/new-york-knicks). Team logo from the [NBA](https://www.nba.com/knicks/).

## For Knicks fans

Make your own W/L call for the matchup and compare it with the model. In the Season tracker, filter by opponent, wins or losses, or games with a graded pregame pick. You can download the filtered results. Model Performance shows the historical holdout separately from the accuracy and probability error of picks recorded this season. The Knicks Film Room has the team’s 2026 championship all-access and ESPN’s trophy ceremony.

This update changes the dashboard and adds live-season evaluation. It uses the same trained model and leaves the existing forecast ledger intact.

### A stronger comparison

Plain Elo uses team strength and a fixed home-court adjustment, with ratings updated from earlier results. On the same 164 Knicks games, Elo picked the winner **69.5%** of the time versus **68.3%** for logistic regression. Logistic regression had lower Brier error (**0.195** versus **0.202**). I kept the logistic probabilities in the app and added both results so fans can see the difference between calling the winner and estimating the chance of a win.

The dashboard is one scrollable page. Both championship videos are embedded in the Film Room near the top, followed by Game Preview, Season Tracker and Model Performance. The links under the players open the same videos on YouTube.

Photos: [Knicks playing at Madison Square Garden](https://commons.wikimedia.org/wiki/File:Knicks_playing_at_Madison_Square_Garden.jpg) by GoCuse44, public domain, and [Interior of Madison Square Garden, 2024](https://commons.wikimedia.org/wiki/File:Interior_of_Madison_Square_Garden_2024.png) by FloridaMan21, CC0.
