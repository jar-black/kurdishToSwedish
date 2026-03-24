"""
collect_opus.py
---------------
Download Kurdish Sorani (ckb) ↔ Swedish (swe) parallel sentence pairs from the
OPUS corpus using the opustools Python library.

Usage:
    pip install opustools
    python data/collect_opus.py --output_dir data/raw
"""

import argparse
import json
import os
from pathlib import Path


OPUS_CORPORA = [
    "CCAligned",
    "WikiMatrix",
    "FLORES-200",
    "MultiUN",
    "OpenSubtitles",
    "QED",
    "TED2020",
    "ELRC_2922",
]


def download_corpus(corpus: str, src_lang: str, tgt_lang: str, output_dir: Path) -> list[dict]:
    """Download a single OPUS corpus and return sentence pairs."""
    try:
        from opustools import OpusGet, OpusRead
    except ImportError:
        raise SystemExit("Install opustools first: pip install opustools")

    pairs = []
    corpus_dir = output_dir / corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {corpus} ({src_lang}↔{tgt_lang})...")
    try:
        getter = OpusGet(
            directory=corpus,
            source=src_lang,
            target=tgt_lang,
            release="latest",
            preprocess="moses",
            download_dir=str(corpus_dir),
            suppress_prompts=True,
        )
        getter.fetch_files()
    except Exception as e:
        print(f"  Skipped {corpus}: {e}")
        return pairs

    src_file = next(corpus_dir.glob(f"*.{src_lang}"), None)
    tgt_file = next(corpus_dir.glob(f"*.{tgt_lang}"), None)

    if not src_file or not tgt_file:
        print(f"  No aligned files found for {corpus}")
        return pairs

    with open(src_file, encoding="utf-8") as sf, open(tgt_file, encoding="utf-8") as tf:
        for src_line, tgt_line in zip(sf, tf):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                pairs.append({"ckb": src_line, "swe": tgt_line, "source": corpus})

    print(f"  Collected {len(pairs):,} pairs from {corpus}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw", help="Where to save downloaded files")
    parser.add_argument("--src_lang", default="ckb", help="Source language code")
    parser.add_argument("--tgt_lang", default="swe", help="Target language code")
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=OPUS_CORPORA,
        help="Which OPUS corpora to try",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    for corpus in args.corpora:
        pairs = download_corpus(corpus, args.src_lang, args.tgt_lang, output_dir)
        all_pairs.extend(pairs)

    # Also try to download FLORES-200 dev/devtest directly (reliable ckb support)
    all_pairs.extend(download_flores200(output_dir))

    out_file = output_dir / "opus_pairs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_pairs):,} pairs saved to {out_file}")


def download_flores200(output_dir: Path) -> list[dict]:
    """
    Download FLORES-200 devtest for ckb and swe directly from the HuggingFace hub.
    FLORES-200 has professional translations into 200 languages including ckb and swe.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed, skipping FLORES-200. pip install datasets")
        return []

    print("Downloading FLORES-200 (ckb + swe)...")
    pairs = []
    try:
        ckb_ds = load_dataset("facebook/flores", "ckb_Arab", trust_remote_code=True)
        swe_ds = load_dataset("facebook/flores", "swe_Latn", trust_remote_code=True)

        for split in ("dev", "devtest"):
            for ckb_row, swe_row in zip(ckb_ds[split], swe_ds[split]):
                pairs.append({
                    "ckb": ckb_row["sentence"],
                    "swe": swe_row["sentence"],
                    "source": f"FLORES-200-{split}",
                })
        print(f"  Collected {len(pairs):,} pairs from FLORES-200")
    except Exception as e:
        print(f"  FLORES-200 download failed: {e}")

    return pairs


if __name__ == "__main__":
    main()
