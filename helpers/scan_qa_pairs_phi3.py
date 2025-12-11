#!/usr/bin/env python3
"""
Scan a directory tree for qa_pairs.json files and compute Phi-3 Mini 4K token stats.

- Counts tokens over text fields only (robust for arbitrary JSON/JSONL).
- If the structure looks like Q/A records (e.g., has "q"/"a" or "question"/"answer"),
  it also computes average tokens per (Q+A) pair.
- Saves a CSV summary and prints a readable report.

Usage:
  python scan_qa_pairs_phi3.py --root /path/to/UNLEARNING --tokenizer-json /path/to/tokenizer.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# ---- Tokenizer loader ----
def load_phi3_tokenizer(tokenizer_json_path: Path):
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        print(
            "ERROR: This script requires the 'tokenizers' package.\n"
            "Install it with: pip install tokenizers\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        tok = Tokenizer.from_file(str(tokenizer_json_path))
        return tok
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {tokenizer_json_path}.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

# ---- JSON helpers ----
def load_json_any(path: Path) -> Any:
    """
    Load JSON or JSONL. Returns a Python object:
    - object/array if JSON
    - list of objects if JSONL
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None
    # Try plain JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try JSON Lines
    objs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            # skip malformed lines
            continue
    return objs

def iter_string_values(obj: Any) -> Iterable[str]:
    """Yield all string values from a nested Python structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_string_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_string_values(item)
    elif isinstance(obj, str):
        yield obj
    else:
        return

# ---- Heuristics for QA records ----
Q_KEYS = {"q", "question", "prompt", "input"}
A_KEYS = {"a", "answer", "completion", "response", "output"}

def extract_qa_pairs(data: Any) -> List[Tuple[str, str]]:
    """
    Try to extract (Q, A) pairs from common schemas.
    Returns a list of (question_text, answer_text).
    """
    pairs: List[Tuple[str, str]] = []
    # Common case: list of dicts
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict):
                q_val = None
                a_val = None
                # direct keys first
                for k, v in rec.items():
                    lk = str(k).lower()
                    if lk in Q_KEYS and isinstance(v, str):
                        q_val = v
                    if lk in A_KEYS and isinstance(v, str):
                        a_val = v
                # nested structures (e.g., {"qa": {"q": "...", "a": "..."}})
                if (q_val is None or a_val is None):
                    # try nested dicts
                    for v in rec.values():
                        if isinstance(v, dict):
                            qq, aa = None, None
                            for k2, v2 in v.items():
                                lk2 = str(k2).lower()
                                if lk2 in Q_KEYS and isinstance(v2, str):
                                    qq = v2
                                if lk2 in A_KEYS and isinstance(v2, str):
                                    aa = v2
                            if qq is not None and aa is not None:
                                q_val = q_val or qq
                                a_val = a_val or aa
                if q_val is not None and a_val is not None:
                    pairs.append((q_val, a_val))
    # Could add more schemas as needed
    return pairs

# ---- Token counting ----
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)

def analyze_file(path: Path, tokenizer) -> Dict[str, Any]:
    data = load_json_any(path)
    strings = list(iter_string_values(data)) if data is not None else []
    total_tokens_strings = sum(count_tokens(tokenizer, s) for s in strings) if strings else 0
    avg_tokens_per_string = (total_tokens_strings / len(strings)) if strings else 0.0

    qa_pairs = extract_qa_pairs(data)
    qa_pair_tokens = []
    for q, a in qa_pairs:
        qa_pair_tokens.append(count_tokens(tokenizer, q) + count_tokens(tokenizer, a))
    avg_tokens_per_qa_pair = (sum(qa_pair_tokens) / len(qa_pair_tokens)) if qa_pair_tokens else 0.0
    total_tokens_qa_pairs = sum(qa_pair_tokens) if qa_pair_tokens else 0

    return {
        "file": str(path),
        "strings_count": len(strings),
        "total_tokens_strings": total_tokens_strings,
        "avg_tokens_per_string": avg_tokens_per_string,
        "qa_pairs_count": len(qa_pairs),
        "total_tokens_qa_pairs": total_tokens_qa_pairs,
        "avg_tokens_per_qa_pair": avg_tokens_per_qa_pair,
    }

def main():
    ap = argparse.ArgumentParser(description="Compute Phi-3 Mini 4K token stats for qa_pairs.json files.")
    ap.add_argument("--root", type=Path, required=True, help="Root directory to scan (e.g., /path/to/UNLEARNING)")
    ap.add_argument("--tokenizer-json", type=Path, required=True, help="Path to Phi-3 Mini 4K tokenizer.json")
    ap.add_argument("--pattern", default="**/qa_pairs.json", help="Glob pattern to find files (default: **/qa_pairs.json)")
    ap.add_argument("--csv-out", type=Path, default=Path("qa_pairs_phi3_token_stats.csv"), help="Path to write CSV summary")
    args = ap.parse_args()

    tokenizer = load_phi3_tokenizer(args.tokenizer_json)

    files = sorted(args.root.glob(args.pattern))
    if not files:
        print(f"No files matched pattern '{args.pattern}' under {args.root}", file=sys.stderr)
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    for f in files:
        stats = analyze_file(f, tokenizer)
        rows.append(stats)

    # Print a readable summary
    print("Exact Phi-3 Mini 4K token stats (text fields only):\n")
    for r in rows:
        print(f"- {r['file']}")
        print(f"    strings: {r['strings_count']}, total_tokens(strings): {r['total_tokens_strings']}, avg_tokens/string: {r['avg_tokens_per_string']:.2f}")
        if r["qa_pairs_count"]:
            print(f"    qa_pairs: {r['qa_pairs_count']}, total_tokens(Q+A): {r['total_tokens_qa_pairs']}, avg_tokens/qa_pair: {r['avg_tokens_per_qa_pair']:.2f}")
        else:
            print(f"    qa_pairs: 0 (no recognizable Q/A structure)")
        print()

    # Compute overall averages
    if rows:
        avg_string = sum(r["avg_tokens_per_string"] for r in rows if r["strings_count"]) / max(1, sum(1 for r in rows if r["strings_count"]))
        qa_rows = [r for r in rows if r["qa_pairs_count"]]
        avg_pair = (sum(r["avg_tokens_per_qa_pair"] for r in qa_rows) / len(qa_rows)) if qa_rows else 0.0
        print("OVERALL AVERAGES (across files that had data):")
        print(f"  mean(avg_tokens_per_string): {avg_string:.2f}")
        if qa_rows:
            print(f"  mean(avg_tokens_per_qa_pair): {avg_pair:.2f}")
        else:
            print("  mean(avg_tokens_per_qa_pair): n/a (no recognizable Q/A pairs)")

    # Write CSV
    fieldnames = [
        "file",
        "strings_count",
        "total_tokens_strings",
        "avg_tokens_per_string",
        "qa_pairs_count",
        "total_tokens_qa_pairs",
        "avg_tokens_per_qa_pair",
    ]
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote CSV summary to: {args.csv_out.resolve()}")

if __name__ == "__main__":
    main()
