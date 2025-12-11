#!/usr/bin/env python3
"""
Batch NER extraction for “unlearning” targets.

For every directory under /home/stratis/unlearning/PURGE/<NAME>
  • read qa_pairs.json
  • build `context` from the "response" fields
  • ask GPT-4 to list the top-60 unique entities
  • save the result in ner_entities_<NAME>.json
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict

import openai
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROOT_DIR = Path("/home/stratis/unlearning/PURGE")
MODEL = "gpt-4o-mini"  # or "gpt-4o" / "gpt-4o-large" / "gpt-4-turbo"
API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
TEMPERATURE = 0.0
MAX_RETRIES = 3

openai.api_key = API_KEY
# ---------------------------------------------------------------------------

def human_readable(dir_name: str) -> str:
    """
    "1_Stephen_King"         -> "Stephen King"
    "10_Prince_Harry,_Duke_of_Sussex" -> "Prince Harry, Duke of Sussex"
    """
    # drop the leading digits + single underscore, then turn the rest of the
    # underscores into spaces
    no_prefix   = re.sub(r"^\d+_", "", dir_name)
    return no_prefix.replace("_", " ")


def load_responses(qa_path: Path) -> str:
    """Return concatenated 'response' strings from a qa_pairs.json file."""
    with qa_path.open("r", encoding="utf-8") as f:
        qa_data: List[Dict] = json.load(f)

    responses = [item.get("response", "") for item in qa_data if "response" in item]
    return "\n".join(responses)


def call_openai(unlearning_target: str, context: str) -> Dict:
    """Query GPT-4 with the required prompt and return parsed JSON."""
    system_msg = (
        "You are an expert data annotator specialising in Named Entity Recognition (NER). "
        "Do not explain your work; only reply with valid JSON."
    )
    user_msg = (
        f"Suppose you are doing named entity recognition on {human_readable(unlearning_target)} "
        f"based *only* on the text below. "
        "List the **top 60** entities that are *unique and specific* to it. "
        "Return them in pure JSON (no markdown, no comments) as a flat array of strings.\n\n"
        f"{context}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                timeout=60,
            )

            content = response.choices[0].message.content.strip()

            # The model should return JSON; attempt to parse it.
            return json.loads(content)
        except (json.JSONDecodeError, openai.error.OpenAIError) as err:
            if attempt == MAX_RETRIES:
                raise
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed for {unlearning_target}: {err}")
    # Should never reach here
    return {}


def main() -> None:
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        sys.exit("❌  OPENAI_API_KEY is not set. Edit the script or export the variable and retry.")

    if not ROOT_DIR.exists():
        sys.exit(f"❌  Directory {ROOT_DIR} not found.")

    # Discover targets (only directories that actually exist)
    targets = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])

    if not targets:
        sys.exit("❌  No sub-directories found; nothing to process.")

    for target_path in tqdm(targets, desc="Processing targets"):
        qa_file = target_path / "qa_pairs.json"
        if not qa_file.exists():
            print(f"[SKIP] {qa_file} missing.")
            continue

        try:
            context = load_responses(qa_file)
            entities = call_openai(target_path.name, context)

            # Persist output
            out_file = target_path / f"ner_entities_{target_path.name}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[ERROR] {target_path.name}: {e}")


if __name__ == "__main__":
    main()
