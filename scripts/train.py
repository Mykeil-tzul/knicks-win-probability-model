"""Train and export the model. Run as python -m scripts.train."""
import json

from forecast import ROOT, historical_games, train

if __name__ == "__main__":
    artifact = train(historical_games())
    (ROOT / "data/model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["backtest"], indent=2))
