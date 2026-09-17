"""Generate the v2 corpus metadata file (data/controllability_v2_corpus.json).

This is bookkeeping ABOUT the 12 stories in data/items.jsonl/data/stories/ --
never anything shown to a model. It exists so the frozen-design lock (see
freeze_controllability_v2.py) can detect if a story file changes after the
study is frozen, and so a reader of the results can see the corpus's basic
provenance (a single anonymised author, naturalistic writing, an
approximate collection window) without any of it ever entering a prompt.

Deliberately excludes the writer's real identity: `author_group` is a fixed
anonymised label shared by all 12 items, never a name.

Run this file directly to (re)write data/controllability_v2_corpus.json. No
model API calls happen here.
"""

import hashlib
import json

from context_trials import ITEMS_FILE, load_items

CORPUS_FILE = "data/controllability_v2_corpus.json"
CORPUS_ID = "controllability_v2_corpus_v1"

# A single shared anonymised label for all 12 items' author -- never a name,
# and never derived from one. All 12 stories come from the same author.
AUTHOR_GROUP = "author_group_anon_1"

# Approximate, not a precise date range (which could help re-identify the
# author via publication metadata elsewhere) -- see the spec's "approximate
# collection window of 18 months".
COLLECTION_WINDOW = {"description": "approximately 18 months", "approximate_duration_months": 18}


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_story_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_corpus_metadata(items_path=ITEMS_FILE):
    items = load_items(items_path)
    story_records = []
    for item in items:
        text = load_story_text(item["path"])
        story_records.append(
            {
                "id": item["id"],
                "sha256": sha256_hex(text),
                "word_count": len(text.split()),
            }
        )
    story_records.sort(key=lambda r: r["id"])

    corpus_hash = sha256_hex("\n".join(f"{r['id']}:{r['sha256']}" for r in story_records))

    return {
        "corpus_id": CORPUS_ID,
        "item_ids": [r["id"] for r in story_records],
        "author_group": AUTHOR_GROUP,
        "naturalistic_writing": True,
        "collection_window": COLLECTION_WINDOW,
        "stories": story_records,
        "corpus_hash": corpus_hash,
    }


def main():
    metadata = build_corpus_metadata()
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Corpus id: {metadata['corpus_id']}")
    print(f"Items: {len(metadata['item_ids'])}")
    print(f"Author group (anonymised): {metadata['author_group']}")
    print(f"Corpus hash: {metadata['corpus_hash']}")
    print(f"Output: {CORPUS_FILE}")


if __name__ == "__main__":
    main()
