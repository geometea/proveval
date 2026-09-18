"""v3 corpus metadata (data/controllability_v3_corpus.json): bookkeeping
ABOUT the same 12 stories v2 used -- per-story sha256 + word count -- so the
v3 design lock can detect any story-file change. Never shown to a model.
The per-story hashes must equal v2's (data/controllability_v2_corpus.json,
read-only) -- assert_matches_v2_corpus enforces that the writing itself is
unchanged. Run directly to (re)write the file; no API calls."""

import hashlib
import json

from context_trials import ITEMS_FILE, load_items
from context_comparisons import load_story

CORPUS_FILE = "data/controllability_v3_corpus.json"
CORPUS_ID = "controllability_v3_corpus_v1"
V2_CORPUS_FILE = "data/controllability_v2_corpus.json"


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_corpus_metadata(items_path=ITEMS_FILE):
    items = load_items(items_path)
    stories = []
    for item in items:
        text = load_story(item["path"])
        stories.append({"id": item["id"], "path": item["path"], "sha256": sha256_hex(text), "word_count": len(text.split())})
    stories.sort(key=lambda r: r["id"])
    corpus_hash = sha256_hex("\n".join(f"{r['id']}:{r['sha256']}" for r in stories))
    return {"corpus_id": CORPUS_ID, "item_ids": [r["id"] for r in stories], "stories": stories, "corpus_hash": corpus_hash}


def assert_matches_v2_corpus(metadata, v2_corpus_path=V2_CORPUS_FILE):
    """The v3 corpus must be the v2 corpus, story for story, byte for byte."""
    with open(v2_corpus_path, "r", encoding="utf-8") as f:
        v2 = json.load(f)
    v2_hashes = {r["id"]: r["sha256"] for r in v2["stories"]}
    v3_hashes = {r["id"]: r["sha256"] for r in metadata["stories"]}
    if v2_hashes != v3_hashes:
        raise ValueError("v3 corpus does not match the frozen v2 corpus (a story file changed or the item list differs)")
    if metadata["corpus_hash"] != v2["corpus_hash"]:
        raise ValueError("v3 corpus_hash differs from v2's")


def main():
    metadata = build_corpus_metadata()
    assert_matches_v2_corpus(metadata)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Corpus id: {metadata['corpus_id']}  Items: {len(metadata['item_ids'])}  Hash: {metadata['corpus_hash']}")
    print(f"Matches v2 corpus: yes  Output: {CORPUS_FILE}")


if __name__ == "__main__":
    main()
