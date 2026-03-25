"""
preprocess.py
-------------
Clean, deduplicate and split all raw data into train/val/test sets.

Usage:
    python data/preprocess.py --input_dir data/raw --output_dir data/processed
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    """NFC-normalize and strip whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def is_valid_pair(ckb: str, swe: str) -> bool:
    """Basic quality filters."""
    if not ckb or not swe:
        return False
    # Reject pairs where both sides are identical (bad alignment)
    if ckb == swe:
        return False
    # Reject very short segments (< 3 chars)
    if len(ckb) < 3 or len(swe) < 3:
        return False
    # Reject extreme length ratios (likely misalignment)
    ratio = len(ckb) / len(swe)
    if ratio < 0.2 or ratio > 5.0:
        return False
    # Kurdish Sorani must contain Arabic-script characters (U+0600–U+06FF)
    arabic_chars = sum(1 for c in ckb if "\u0600" <= c <= "\u06ff")
    if arabic_chars < 2:
        return False
    return True


def load_jsonl(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all raw data ──────────────────────────────────────────────────
    all_pairs: list[dict] = []
    for jsonl_file in input_dir.glob("*.jsonl"):
        batch = load_jsonl(jsonl_file)
        print(f"Loaded {len(batch):,} pairs from {jsonl_file.name}")
        all_pairs.extend(batch)

    print(f"\nRaw total: {len(all_pairs):,} pairs")

    # ── Normalize ──────────────────────────────────────────────────────────
    for pair in all_pairs:
        pair["ckb"] = normalize(pair.get("ckb", ""))
        pair["swe"] = normalize(pair.get("swe", ""))

    # ── Filter ────────────────────────────────────────────────────────────
    valid = [p for p in all_pairs if is_valid_pair(p["ckb"], p["swe"])]
    print(f"After filtering: {len(valid):,} pairs")

    # ── Deduplicate on (ckb, swe) ──────────────────────────────────────────
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for pair in valid:
        key = (pair["ckb"], pair["swe"])
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    print(f"After deduplication: {len(deduped):,} pairs")

    # ── Prioritize poetry data ─────────────────────────────────────────────
    # Synthetic and manual pairs first so they appear in val/test
    poetry_sources = {"synthetic-claude", "manual"}
    poetry = [p for p in deduped if p.get("source", "") in poetry_sources]
    general = [p for p in deduped if p.get("source", "") not in poetry_sources]

    random.seed(args.seed)
    random.shuffle(poetry)
    random.shuffle(general)

    # Combine: poetry first
    ordered = poetry + general
    n = len(ordered)

    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test

    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]

    # ── Save ───────────────────────────────────────────────────────────────
    def save_split(pairs: list[dict], name: str):
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs):,} pairs → {path}")

    save_split(train, "train")
    save_split(val, "val")
    save_split(test, "test")

    # Summary
    print(f"\n{'='*40}")
    print(f"Train : {len(train):>6,}")
    print(f"Val   : {len(val):>6,}")
    print(f"Test  : {len(test):>6,}")
    print(f"Total : {n:>6,}")

    # Source breakdown
    from collections import Counter
    sources = Counter(p.get("source", "unknown") for p in ordered)
    print("\nBy source:")
    for src, cnt in sources.most_common():
        print(f"  {src:<30} {cnt:>6,}")


if __name__ == "__main__":
    main()
