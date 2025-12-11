#!/usr/bin/env python3
"""
Traverse the first N numbered subdirectories (e.g., '1_Stephen_King'…'21_50_Cent'),
and in each, read a specified JSON file and compute Phi-3 Mini 4K token stats over
only the specified fields.

Supports fields being strings or lists (flattens lists into joined strings).

Usage:
    pip install tokenizers
    python traverse_and_count_json_fields.py \
      --root /path/to/UNLEARNING/PURGE \
      --tokenizer-json /path/to/tokenizer.json \
      --json-file reject_phi.json \
      --fields prompt response \
      --limit 21
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---- Tokenizer loader ----
def load_phi3_tokenizer(tokenizer_json_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: This script requires the 'tokenizers' package.\nInstall it with: pip install tokenizers",
              file=sys.stderr)
        sys.exit(1)
    try:
        return Tokenizer.from_file(str(tokenizer_json_path))
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {tokenizer_json_path}: {e}", file=sys.stderr)
        sys.exit(1)

# ---- JSON loader ----
def load_json_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # try JSON lines
        objs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                pass
        return objs

# ---- Field normalization ----
def normalize_field_value(val: Any) -> str:
    """Ensure a field value is turned into a clean string."""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for v in val:
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                # Flatten dict by joining values
                parts.extend(str(x) for x in v.values())
            else:
                parts.append(str(v))
        return " ".join(parts)
    if isinstance(val, dict):
        return " ".join(str(x) for x in val.values())
    return str(val)

# ---- Field extraction ----
def extract_records(data: Any, fields: List[str]) -> List[str]:
    """Extract values for specified fields from list-of-dicts or dict JSON."""
    out: List[str] = []
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            parts = []
            for f in fields:
                if f in rec:
                    val = normalize_field_value(rec[f])
                    if val:
                        parts.append(val)
            if parts:
                out.append(" ".join(parts))
    elif isinstance(data, dict):
        parts = []
        for f in fields:
            if f in data:
                val = normalize_field_value(data[f])
                if val:
                    parts.append(val)
        if parts:
            out.append(" ".join(parts))
    return out

# ---- Token counting ----
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)

# ---- Directory selection ----
LEADING_NUM_RE = re.compile(r"^(\d+)")

def pick_first_n_numeric_dirs(root: Path, n: int) -> List[Path]:
    dirs = []
    for p in root.iterdir():
        if p.is_dir():
            m = LEADING_NUM_RE.match(p.name)
            if m:
                idx = int(m.group(1))
                dirs.append((idx, p))
    dirs.sort()
    return [p for _, p in dirs[:n]]

# ---- Main ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Count Phi-3 Mini 4K tokens for specified fields in JSON across first N numbered dirs.")
    ap.add_argument("--root", type=Path, required=True, help="Root folder containing numbered subdirectories.")
    ap.add_argument("--tokenizer-json", type=Path, required=True, help="Path to Phi-3 tokenizer.json.")
    ap.add_argument("--json-file", type=str, required=True, help="Filename inside each dir (e.g., reject_phi.json).")
    ap.add_argument("--fields", nargs="+", required=True, help="Field names to concatenate and tokenize (e.g., prompt response).")
    ap.add_argument("--limit", type=int, default=21, help="How many numeric dirs to scan (default: 21)")
    ap.add_argument("--csv-out", type=Path, default=Path("field_token_stats.csv"), help="CSV output path")
    args = ap.parse_args()

    tokenizer = load_phi3_tokenizer(args.tokenizer_json)
    dirs = pick_first_n_numeric_dirs(args.root, args.limit)
    if not dirs:
        print(f"No numeric dirs found under {args.root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for d in dirs:
        f = d / args.json_file
        if not f.exists():
            print(f"Skipping missing: {f}", file=sys.stderr)
            continue
        data = load_json_any(f)
        records = extract_records(data, args.fields) or []
        counts = [count_tokens(tokenizer, rec) for rec in records]
        total = sum(counts)
        avg = (total / len(counts)) if counts else 0.0
        rows.append({
            "dir": d.name,
            "file": str(f),
            "records_counted": len(counts),
            "total_tokens": total,
            "avg_tokens": avg,
        })

    # Console report
    print(f"Phi-3 Mini 4K token stats over fields {args.fields}:")
    for r in rows:
        print(f"- {r['dir']}: records={r['records_counted']}, total_tokens={r['total_tokens']}")

    # Overall
    if rows:
        overall_tot = sum(r["total_tokens"] for r in rows)
        overall_records = sum(r["records_counted"] for r in rows)
        overall_avg = (overall_tot / overall_records) if overall_records else 0.0
        print(f"OVERALL: scanned {len(rows)} dirs, total_records={overall_records}, total_tokens={overall_tot}, avg_tokens={overall_avg:.2f}")

    # Write CSV
    with args.csv_out.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["dir","file","records_counted","total_tokens","avg_tokens"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to: {args.csv_out.resolve()}")
