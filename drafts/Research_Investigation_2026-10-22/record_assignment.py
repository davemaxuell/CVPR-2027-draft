"""Coordinator-only ledger update with a three-delegate concurrency check."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("number", choices=[f"{i:02}" for i in range(1, 11)])
parser.add_argument("status", choices=["In progress", "Complete"])
parser.add_argument("--agent")
parser.add_argument("--outputs", nargs="*", default=[])
args = parser.parse_args()
path = ROOT / "assignment_ledger.json"
data = json.loads(path.read_text(encoding="utf-8"))
row = next(r for r in data if r["assignment"] == args.number)
now = datetime.now(timezone.utc).isoformat()
if args.status == "Complete":
    if not args.outputs:
        raise SystemExit("Completion requires concrete output paths")
    missing = [p for p in args.outputs if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("Missing completion outputs: " + str(missing))
    row["completed_utc"] = now
    row["outputs"] = args.outputs
else:
    row.setdefault("started_utc", now)
    if row.get("completed_utc"):
        row.setdefault("prior_completion_utc", []).append(row.pop("completed_utc"))
    row["last_resumed_utc"] = now
row["status"] = args.status
if args.agent:
    row["agent"] = args.agent
active = [r["assignment"] for r in data if r["status"] == "In progress"]
if len(active) > 3:
    raise SystemExit("Ledger would exceed three active delegates plus coordinator: " + str(active))
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
events_path = ROOT / "assignments/ledger_events.json"
events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
events.append({"time_utc": now, "assignment": args.number, "status": args.status,
               "active_delegated_assignments_after_event": active,
               "coordinator_active": True})
events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(row, ensure_ascii=False, indent=2))
