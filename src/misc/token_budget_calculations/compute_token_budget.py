#!/usr/bin/env python3
"""
Traverse the first N numbered subdirectories (e.g., '1_Stephen_King'…'20_Marlon_Brando'),
and in each, read its qa_pairs.json and compute Phi‑3 Mini 4K token stats over only the
first M QA pairs.

- Limits the scan to the first --limit numeric dirs by prefix order.
- Within each qa_pairs.json, extracts Q/A pairs via common keys (q, question, prompt, input / a, answer, completion, response, output).
- Truncates to the first --first-n pairs if set.
- Counts tokens for each question + answer concatenation (or separately) and reports per-file totals and averages.
- Outputs a CSV summary and prints a console report.

Usage:
    pip install tokenizers
    python compute_token_budget.py \
      --root /path/to/UNLEARNING/PURGE \
      --tokenizer-json /path/to/tokenizer.json \
      --limit 20 --first-n 100

"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- Tokenizer loader ----
def load_phi3_tokenizer(tokenizer_json_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        print(
            "ERROR: This script requires the 'tokenizers' package."
            "\nInstall it with: pip install tokenizers",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        return Tokenizer.from_file(str(tokenizer_json_path))
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {tokenizer_json_path}: {e}", file=sys.stderr)
        sys.exit(1)

# ---- JSON helpers ----
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

# ---- QA extraction ----n
Q_KEYS = {"prompt", "instruction", "text"}
A_KEYS = {"response", "output"}


def extract_qa_pairs(data: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            q_val = None
            a_val = None
            for k, v in rec.items():
                lk = k.lower()
                if lk in Q_KEYS and isinstance(v, str):
                    q_val = v
                if lk in A_KEYS and isinstance(v, str):
                    a_val = v
            # nested fallback
            if (q_val is None or a_val is None):
                for v in rec.values():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            lk2 = k2.lower()
                            if q_val is None and lk2 in Q_KEYS and isinstance(v2, str):
                                q_val = v2
                            if a_val is None and lk2 in A_KEYS and isinstance(v2, str):
                                a_val = v2
            if q_val is not None and a_val is not None:
                pairs.append((q_val, a_val))
    return pairs

# ---- Token counting ----
def count_pair_tokens(tokenizer, pair: Tuple[str, str]) -> int:
    q, a = pair
    return len(tokenizer.encode(q).ids) + len(tokenizer.encode(a).ids)

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
    ap = argparse.ArgumentParser(description="Count Phi‑3 Mini 4K tokens for first M QA pairs across first N numbered dirs.")
    ap.add_argument("--root", type=Path, required=True, help="Root folder containing numbered subdirectories.")
    ap.add_argument("--tokenizer-json", type=Path, required=True, help="Path to Phi‑3 tokenizer.json.")
    ap.add_argument("--limit", type=int, default=20, help="First how many numeric dirs to scan (default: 20)")
    ap.add_argument("--first-n", type=int, default=100, help="Only consider first M QA pairs in each file (default: 100)")
    ap.add_argument("--csv-out", type=Path, default=Path("first20_qa100_token_stats.csv"), help="CSV output path")
    args = ap.parse_args()

    tokenizer = load_phi3_tokenizer(args.tokenizer_json)
    dirs = pick_first_n_numeric_dirs(args.root, args.limit)
    if not dirs:
        print(f"No numeric dirs found under {args.root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for d in dirs:
        f = d / "qa_pairs.json"
        if not f.exists():
            print(f"Skipping missing: {f}", file=sys.stderr)
            continue
        data = load_json_any(f)
        qa_pairs = extract_qa_pairs(data) or []
        subset = qa_pairs[: args.first_n]
        counts = [count_pair_tokens(tokenizer, p) for p in subset]
        total = sum(counts)
        avg = (total / len(counts)) if counts else 0.0
        rows.append({
            "dir": d.name,
            "file": str(f),
            "pairs_counted": len(counts),
            "total_tokens": total,
            "avg_tokens_per_pair": avg,
        })

    # Print report
    print("Phi‑3 Mini 4K token stats for first", args.first_n, "QA pairs:")
    for r in rows:
        print(f"- {r['dir']}: pairs={r['pairs_counted']}, total_tokens={r['total_tokens']}, avg_tokens_per_pair={r['avg_tokens_per_pair']:.2f}")

    # Overall
    if rows:
        overall_tot = sum(r["total_tokens"] for r in rows)
        overall_pairs = sum(r["pairs_counted"] for r in rows)
        overall_avg = (overall_tot / overall_pairs) if overall_pairs else 0.0
        print(f"OVERALL: scanned {len(rows)} dirs, total_pairs={overall_pairs}, total_tokens={overall_tot}, avg_tokens_per_pair={overall_avg:.2f}")

    # Write CSV
    with args.csv_out.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["dir","file","pairs_counted","total_tokens","avg_tokens_per_pair"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to: {args.csv_out.resolve()}")
