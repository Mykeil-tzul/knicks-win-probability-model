import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load raw Knicks game log
# -----------------------------
raw = pd.read_csv("knicks_games.csv", header=None)

# Based on your raw CSV structure:
# 0 = Rk
# 1 = G
# 2 = Date
# 3 = Home/Away marker (blank or "@")
# 4 = Opponent
# 5 = Result (W/L)
# 6 = Knicks points
# 7 = Opponent points

df = pd.DataFrame({
    "Date": raw.iloc[:, 2],
    "HomeAway": raw.iloc[:, 3],
    "Opponent": raw.iloc[:, 4],
    "Result": raw.iloc[:, 5],
    "Tm_PTS": raw.iloc[:, 6],
    "Opp_PTS": raw.iloc[:, 7],
})

# -----------------------------
# 2. Clean core fields
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Tm_PTS"] = pd.to_numeric(df["Tm_PTS"], errors="coerce")
df["Opp_PTS"] = pd.to_numeric(df["Opp_PTS"], errors="coerce")

# Home game = 1, away game = 0
df["Home"] = df["HomeAway"].apply(lambda x: 0 if str(x).strip() == "@" else 1)

# Win target
df["Win"] = df["Result"].apply(lambda x: 1 if str(x).strip() == "W" else 0)

# Point margin
df["Margin"] = df["Tm_PTS"] - df["Opp_PTS"]

# Drop bad rows
df = df.dropna(subset=["Date", "Tm_PTS", "Opp_PTS"]).copy()

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# 3. Feature engineering
# Use only PRIOR games (shift(1))
# -----------------------------
df["PTS_avg_5"] = df["Tm_PTS"].shift(1).rolling(5).mean()
df["Opp_PTS_avg_5"] = df["Opp_PTS"].shift(1).rolling(5).mean()
df["Margin_avg_5"] = df["Margin"].shift(1).rolling(5).mean()
df["Win_pct_last_5"] = df["Win"].shift(1).rolling(5).mean()

# Optional: short-term trend
df["PTS_avg_3"] = df["Tm_PTS"].shift(1).rolling(3).mean()
df["Opp_PTS_avg_3"] = df["Opp_PTS"].shift(1).rolling(3).mean()

# Keep only rows where rolling features exist
df = df.dropna(subset=[
    "PTS_avg_5",
    "Opp_PTS_avg_5",
    "Margin_avg_5",
    "Win_pct_last_5",
    "PTS_avg_3",
    "Opp_PTS_avg_3"
]).reset_index(drop=True)

print("Cleaned shape:", df.shape)
print("\nCleaned preview:\n")
print(df[[
    "Date", "Opponent", "Home", "Win",
    "Tm_PTS", "Opp_PTS",
    "PTS_avg_5", "Opp_PTS_avg_5",
    "Margin_avg_5", "Win_pct_last_5"
]].head())

# -----------------------------
# 4. Build Win Probability Model
# -----------------------------
features = [
    "Home",
    "PTS_avg_5",
    "Opp_PTS_avg_5",
    "Margin_avg_5",
    "Win_pct_last_5",
    "PTS_avg_3",
    "Opp_PTS_avg_3"
]

X = df[features]
y = df["Win"]

# Time-based split: earlier games train, later games test
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel accuracy:", round(accuracy, 3))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

results = X_test.copy()
results["Actual_Win"] = y_test.values
results["Predicted_Win"] = y_pred
results["Win_Probability"] = y_prob

print("\nPrediction sample:\n")
print(results.head(10))

# -----------------------------
# 5. Predict next game probability
# Use the latest rolling features
# -----------------------------
latest = df.iloc[-1]

next_game_input = pd.DataFrame([{
    "Home": 1,  # change to 0 for away
    "PTS_avg_5": latest["PTS_avg_5"],
    "Opp_PTS_avg_5": latest["Opp_PTS_avg_5"],
    "Margin_avg_5": latest["Margin_avg_5"],
    "Win_pct_last_5": latest["Win_pct_last_5"],
    "PTS_avg_3": latest["PTS_avg_3"],
    "Opp_PTS_avg_3": latest["Opp_PTS_avg_3"],
}])

next_game_prob = model.predict_proba(next_game_input)[0, 1]

print("\nNext game predicted Knicks win probability:", round(next_game_prob, 3))