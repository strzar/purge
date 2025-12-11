#!/usr/bin/env python3
"""
Prepare manual NER prompts for ChatGPT.

For every directory under /home/stratis/unlearning/PURGE/<NAME>:
  • read qa_pairs.json
  • concatenate all 'response' entries into `context`
  • create ner_request_<NAME>.txt with:
      ┌─────────────────────────────────────────────
      | Suppose you are doing named entity …
      |
      | === RESPONSES ===
      | … <context> …
      └─────────────────────────────────────────────
You can then open the .txt file and paste its entire
contents into a ChatGPT Plus conversation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

ROOT_DIR = Path("PURGE")


def human_readable(dir_name: str) -> str:
    """
    "1_Stephen_King"  -> "Stephen King"
    "10_Prince_Harry,_Duke_of_Sussex" -> "Prince Harry, Duke of Sussex"
    """
    no_prefix = re.sub(r"^\d+_", "", dir_name, count=1)
    return no_prefix.replace("_", " ")


def build_context(qa_file: Path) -> str:
    with qa_file.open("r", encoding="utf-8") as f:
        qa_pairs: List[Dict] = json.load(f)
    responses = [q["response"] for q in qa_pairs if "response" in q]
    return "\n".join(responses)


def main() -> None:
    if not ROOT_DIR.exists():
        raise SystemExit(f"Directory {ROOT_DIR} not found.")

    for target_path in ROOT_DIR.iterdir():
        if not target_path.is_dir():
            continue

        qa_file = target_path / "qa_pairs.json"
        if not qa_file.exists():
            print(f"[SKIP] {qa_file} missing.")
            continue

        pretty_name = human_readable(target_path.name)
        context = build_context(qa_file)

        prompt = (
            f"Suppose you are doing named entity recognition on {pretty_name} "
            "based only on the responses below. "
            "What are the top 60 entities that are unique and specific to it? "
            "Try avoiding generic words. Give the answer in pure JSON format.\n\n"
            "=== RESPONSES ===\n"
        )

        out_file = target_path / f"ner_request_{pretty_name}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            f.write(prompt)
            f.write(context)

        print(f"[✓] Wrote {out_file.relative_to(ROOT_DIR.parent)}")


if __name__ == "__main__":
    main()