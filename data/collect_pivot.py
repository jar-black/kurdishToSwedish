"""
collect_pivot.py
----------------
Collect Kurdish Sorani (ckb) ↔ English (en) parallel data from OPUS and
automatically pivot-translate the English side into Swedish using the base
NLLB-200 model (no fine-tuning needed — we just use the off-the-shelf
multilingual model for the en→swe leg).

Result: additional ckb↔swe sentence pairs labelled source="pivot-en".

These pivoted pairs are lower quality than direct translations but give the
fine-tuning pipeline more Kurdish Sorani signal when direct ckb↔swe data is
scarce.

Usage:
    # CPU-only (slow, good for testing):
    python data/collect_pivot.py --n_pairs 500 --device cpu

    # GPU (much faster):
    python data/collect_pivot.py --n_pairs 5000

    # Skip the translation step and only download raw ckb↔en pairs:
    python data/collect_pivot.py --download_only

Output files (in --output_dir, default: data/raw):
    opus_ckb_en.jsonl   – raw ckb↔en pairs
    pivot_ckb_swe.jsonl – pivoted ckb↔swe pairs (ready for preprocess.py)
"""

import argparse
import json
from pathlib import Path

import torch


# ── OPUS corpora that have reasonable ckb↔en coverage ────────────────────────
OPUS_CORPORA_EN = [
    "CCAligned",
    "WikiMatrix",
    "FLORES-200",
    "MultiUN",
    "QED",
    "TED2020",
    "OpenSubtitles",
    "Tanzil",       # Quran translations – formal register, good for vocab
    "GlobalVoices",
]

NLLB_MODEL = "facebook/nllb-200-distilled-600M"   # Smaller model — fast pivot
BATCH_SIZE = 32


# ── Download helpers ───────────────────────────────────────────────────────────

def download_opus_ckb_en(corpora: list[str], output_dir: Path) -> list[dict]:
    """Download ckb↔en pairs from OPUS corpora."""
    try:
        from opustools import OpusGet
    except ImportError:
        raise SystemExit("Install opustools first: pip install opustools")

    all_pairs: list[dict] = []

    for corpus in corpora:
        corpus_dir = output_dir / "raw_en" / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {corpus} (ckb↔en)...")
        try:
            getter = OpusGet(
                directory=corpus,
                source="ckb",
                target="en",
                release="latest",
                preprocess="moses",
                download_dir=str(corpus_dir),
                suppress_prompts=True,
            )
            getter.get_files()
        except Exception as e:
            print(f"  Skipped {corpus}: {e}")
            continue

        ckb_file = next(corpus_dir.glob("*.ckb"), None)
        en_file = next(corpus_dir.glob("*.en"), None)

        if not ckb_file or not en_file:
            print(f"  No aligned files found for {corpus}")
            continue

        count = 0
        with open(ckb_file, encoding="utf-8") as sf, open(en_file, encoding="utf-8") as tf:
            for ckb_line, en_line in zip(sf, tf):
                ckb_line = ckb_line.strip()
                en_line = en_line.strip()
                if ckb_line and en_line:
                    all_pairs.append({"ckb": ckb_line, "en": en_line, "source": corpus})
                    count += 1

        print(f"  Collected {count:,} pairs from {corpus}")

    return all_pairs


def download_flores200_ckb_en(output_dir: Path) -> list[dict]:
    """
    Download FLORES-200 ckb↔en from HuggingFace (~1 000 pairs).
    Tries multiple dataset IDs in order of preference.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed, skipping FLORES-200. pip install datasets")
        return []

    # Candidate dataset IDs (tried in order)
    FLORES_IDS = [
        ("openlanguagedata/flores_plus", "ckb_Arab", "eng_Latn"),
        ("Muennighoff/flores200",        "ckb_Arab", "eng_Latn"),
    ]

    print("Downloading FLORES-200 (ckb↔en)...")
    for dataset_id, ckb_config, en_config in FLORES_IDS:
        try:
            ckb_ds = load_dataset(dataset_id, ckb_config)
            en_ds  = load_dataset(dataset_id, en_config)
            pairs: list[dict] = []
            for split in ("dev", "devtest"):
                if split not in ckb_ds or split not in en_ds:
                    continue
                for ckb_row, en_row in zip(ckb_ds[split], en_ds[split]):
                    sent_key = "sentence" if "sentence" in ckb_row else "text"
                    pairs.append({
                        "ckb": ckb_row[sent_key],
                        "en":  en_row[sent_key],
                        "source": f"FLORES-200-{split}",
                    })
            if pairs:
                print(f"  Collected {len(pairs):,} pairs from FLORES-200 ({dataset_id})")
                return pairs
        except Exception as e:
            print(f"  {dataset_id} failed: {e}")

    print("  Could not download FLORES-200 from any source.")
    return []


# ── Pivot translation: en → swe via NLLB ──────────────────────────────────────

def load_nllb_translator(model_name: str, device: str):
    """Load NLLB tokenizer and model for translation."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {model_name} for en→swe pivot translation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def translate_en_to_swe_batch(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 256,
) -> list[str]:
    """Translate a batch of English sentences to Swedish using NLLB."""
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos = tokenizer.convert_tokens_to_ids("swe_Latn")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def pivot_translate(
    pairs: list[dict],
    model,
    tokenizer,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    For each pair {ckb, en, source}, translate en→swe and produce
    {ckb, swe, source="pivot-en/<original_source>"}.
    """
    from tqdm import tqdm

    pivoted: list[dict] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Pivoting en→swe"):
        batch = pairs[i : i + batch_size]
        en_texts = [p["en"] for p in batch]
        try:
            swe_texts = translate_en_to_swe_batch(model, tokenizer, en_texts)
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
            continue

        for pair, swe in zip(batch, swe_texts):
            pivoted.append({
                "ckb": pair["ckb"],
                "swe": swe,
                "source": f"pivot-en/{pair['source']}",
            })

    return pivoted


# ── Quality filter for pivot pairs ────────────────────────────────────────────

def basic_filter(pairs: list[dict]) -> list[dict]:
    """
    Remove obviously bad pivot pairs:
    - Empty strings
    - Identical ckb/swe (translation failure)
    - Very short Swedish output (< 5 chars — truncation failure)
    - Kurdish without Arabic script
    """
    clean = []
    for p in pairs:
        ckb = p.get("ckb", "").strip()
        swe = p.get("swe", "").strip()
        if not ckb or not swe:
            continue
        if ckb == swe:
            continue
        if len(swe) < 5:
            continue
        arabic_chars = sum(1 for c in ckb if "\u0600" <= c <= "\u06ff")
        if arabic_chars < 2:
            continue
        clean.append(p)
    return clean


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect ckb↔en from OPUS and pivot-translate en→swe via NLLB."
    )
    parser.add_argument(
        "--output_dir", default="data/raw",
        help="Directory for output JSONL files (default: data/raw)"
    )
    parser.add_argument(
        "--n_pairs", type=int, default=0,
        help="Max ckb↔en pairs to translate (0 = all)"
    )
    parser.add_argument(
        "--corpora", nargs="+", default=OPUS_CORPORA_EN,
        help="OPUS corpora to try for ckb↔en"
    )
    parser.add_argument(
        "--nllb_model", default=NLLB_MODEL,
        help=f"NLLB model for pivot translation (default: {NLLB_MODEL})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for pivot translation (default: 32)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for NLLB: 'auto', 'cuda', 'cpu' (default: auto)"
    )
    parser.add_argument(
        "--download_only", action="store_true",
        help="Only download raw ckb↔en pairs, skip pivot translation"
    )
    parser.add_argument(
        "--skip_opus", action="store_true",
        help="Skip OPUS download (use existing opus_ckb_en.jsonl)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_en_path = output_dir / "opus_ckb_en.jsonl"
    pivot_path = output_dir / "pivot_ckb_swe.jsonl"

    # ── Step 1: gather ckb↔en pairs ───────────────────────────────────────
    if args.skip_opus and raw_en_path.exists():
        print(f"Loading existing ckb↔en pairs from {raw_en_path}...")
        pairs_en: list[dict] = []
        with open(raw_en_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs_en.append(json.loads(line))
        print(f"  Loaded {len(pairs_en):,} pairs")
    else:
        pairs_en = download_opus_ckb_en(args.corpora, output_dir)
        pairs_en += download_flores200_ckb_en(output_dir)

        # Save raw ckb↔en
        with open(raw_en_path, "w", encoding="utf-8") as f:
            for pair in pairs_en:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(pairs_en):,} raw ckb↔en pairs → {raw_en_path}")

    if args.download_only:
        print("--download_only set, stopping before pivot translation.")
        return

    # ── Step 2: optional truncation ───────────────────────────────────────
    if args.n_pairs > 0 and len(pairs_en) > args.n_pairs:
        # Prefer FLORES-200 (higher quality) then take remainder randomly
        import random
        flores = [p for p in pairs_en if "FLORES" in p.get("source", "")]
        others = [p for p in pairs_en if "FLORES" not in p.get("source", "")]
        random.shuffle(others)
        pairs_en = flores + others
        pairs_en = pairs_en[: args.n_pairs]
        print(f"Truncated to {len(pairs_en):,} pairs for translation")

    # ── Step 3: pivot translate en → swe ──────────────────────────────────
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, tokenizer = load_nllb_translator(args.nllb_model, device)

    pivoted = pivot_translate(pairs_en, model, tokenizer, batch_size=args.batch_size)
    print(f"\nPivot-translated {len(pivoted):,} pairs")

    # ── Step 4: quality filter ─────────────────────────────────────────────
    pivoted_clean = basic_filter(pivoted)
    print(f"After quality filter: {len(pivoted_clean):,} pairs")

    # ── Step 5: save ──────────────────────────────────────────────────────
    with open(pivot_path, "w", encoding="utf-8") as f:
        for pair in pivoted_clean:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pivoted_clean):,} pivoted ckb↔swe pairs → {pivot_path}")
    print("\nNext step: run data/preprocess.py (it will pick up pivot_ckb_swe.jsonl automatically)")


if __name__ == "__main__":
    main()
