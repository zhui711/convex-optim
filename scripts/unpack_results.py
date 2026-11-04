"""从无损快照恢复全部 JSON 与逐次 CSV，无需运行优化算法。"""
from pathlib import Path
import csv
import gzip
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    folder = ROOT / "results"
    packed = folder / "experiment_data.json.gz"
    with gzip.open(packed, "rt", encoding="utf-8") as stream:
        text = stream.read()
    payload = json.loads(text)
    (folder / "experiment_data.json").write_text(text, encoding="utf-8")
    traces = folder / "trajectories"
    traces.mkdir(exist_ok=True)
    for run in payload["runs"]:
        rows = run["history"]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (traces / (run["run_id"] + ".csv")).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Restored {len(payload['runs'])} runs to {folder}")

if __name__ == "__main__":
    main()
