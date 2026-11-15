"""Export one completed game per row from the NBA Cloud Data Platform snapshot."""
import argparse

import duckdb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("parquet")
args = parser.parse_args()
with duckdb.connect() as db:
    rows = db.execute("""
        WITH teams AS (
            SELECT game_date AS date, season, game_id AS id, team_id,
                   bool_or(home) AS home_flag, sum(pts) AS score
            FROM read_parquet(?) WHERE season BETWEEN 2023 AND 2026
            GROUP BY ALL
        )
        SELECT a.date, a.season, a.id, a.team_id AS home, b.team_id AS away,
               a.score AS home_score, b.score AS away_score,
               a.home_flag IS NULL AND b.home_flag IS NULL AS neutral
        FROM teams a JOIN teams b USING (date, season, id)
        WHERE (a.home_flag AND NOT b.home_flag)
           OR (a.home_flag IS NULL AND b.home_flag IS NULL AND a.team_id < b.team_id)
        ORDER BY a.date, a.id
    """, [args.parquet]).fetchdf()
    if len(rows) != 4920 or rows.isna().any().any() or (rows.home_score == rows.away_score).any():
        raise ValueError(f"Expected four complete seasons: {len(rows)=}, {rows.isna().sum().to_dict()}, {(rows.home_score == rows.away_score).sum()} ties")
    rows = rows.astype({"home_score": int, "away_score": int})
    rows.to_csv("data/history.csv", index=False)
