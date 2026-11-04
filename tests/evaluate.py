"""Run the same five case prompts on the Agent and direct LLM baseline."""
import json
import argparse
import time
from pathlib import Path
from src.agent import run_agent, run_direct
from src.tools import get_wifi_metrics


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("both", "direct", "agent"), default="both")
parser.add_argument("--case", help="Only rerun this case ID")
args = parser.parse_args()
mode_choice = args.mode
cases = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
path = ROOT / "results" / "raw_runs.json"
path.parent.mkdir(exist_ok=True)
runs = json.loads(path.read_text(encoding="utf-8")) if path.exists() and mode_choice != "both" else []
for case in cases:
    if args.case and case["id"] != args.case:
        continue
    row = next((r for r in runs if r["id"] == case["id"]), None)
    if row is None:
        row = dict(case)
        runs.append(row)
    for mode in (("direct", "agent") if mode_choice == "both" else (mode_choice,)):
        start = time.perf_counter()
        try:
            row[mode] = run_direct(case["question"], get_wifi_metrics(case["ap_id"])) if mode == "direct" else run_agent(case["question"])
        except Exception as exc:
            row[mode] = {"error": str(exc)}
        row[mode + "_seconds"] = round(time.perf_counter() - start, 2)
    path.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(case["id"], "完成；已保存", flush=True)
