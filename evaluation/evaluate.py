"""
evaluate.py
-----------
Evaluate the fine-tuned model on the test split using chrF++, BLEU,
and optional back-translation consistency.

Usage:
    python evaluation/evaluate.py \
        --model_dir checkpoints/final \
        --base_model facebook/nllb-200-distilled-1.3B \
        --test_file data/processed/test.jsonl \
        --output_file evaluation/results.jsonl
"""

import argparse
import json
from pathlib import Path

import evaluate
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SRC_LANG = "ckb_Arab"
TGT_LANG = "swe_Latn"
BATCH_SIZE = 8


def load_model(model_dir: str, base_model: str):
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Loading base model {base_model}...")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Applying LoRA adapters from {model_dir}...")
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer


def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int = 256,
) -> list[str]:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/final")
    parser.add_argument("--base_model", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--test_file", default="data/processed/test.jsonl")
    parser.add_argument("--output_file", default="evaluation/results.jsonl")
    parser.add_argument("--back_translate", action="store_true",
                        help="Also compute back-translation consistency (2× inference time)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    # Load test data
    test_pairs = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_pairs.append(json.loads(line))

    print(f"Evaluating on {len(test_pairs):,} test pairs...")

    sources = [p["ckb"] for p in test_pairs]
    references = [p["swe"] for p in test_pairs]

    # Translate in batches
    predictions: list[str] = []
    for i in tqdm(range(0, len(sources), BATCH_SIZE), desc="Translating"):
        batch = sources[i : i + BATCH_SIZE]
        preds = translate_batch(model, tokenizer, batch, SRC_LANG, TGT_LANG)
        predictions.extend(preds)

    # ── Metrics ────────────────────────────────────────────────────────────
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("sacrebleu")

    chrf_result = chrf.compute(
        predictions=predictions,
        references=[[r] for r in references],
        word_order=2,
    )
    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    print("\n" + "=" * 50)
    print(f"chrF++ : {chrf_result['score']:.2f}")
    print(f"BLEU   : {bleu_result['score']:.2f}")

    # ── Optional back-translation ──────────────────────────────────────────
    bt_consistency = None
    if args.back_translate:
        print("\nRunning back-translation (swe → ckb)...")
        back_translations: list[str] = []
        for i in tqdm(range(0, len(predictions), BATCH_SIZE), desc="Back-translating"):
            batch = predictions[i : i + BATCH_SIZE]
            bt = translate_batch(model, tokenizer, batch, TGT_LANG, SRC_LANG)
            back_translations.extend(bt)

        chrf_bt = chrf.compute(
            predictions=back_translations,
            references=[[s] for s in sources],
            word_order=2,
        )
        bt_consistency = chrf_bt["score"]
        print(f"Back-translation chrF++ (swe→ckb vs original ckb): {bt_consistency:.2f}")

    # ── Save detailed results ──────────────────────────────────────────────
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (src, ref, pred) in enumerate(zip(sources, references, predictions)):
            row = {
                "id": i,
                "ckb": src,
                "reference_swe": ref,
                "predicted_swe": pred,
            }
            if args.back_translate:
                row["back_translation_ckb"] = back_translations[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDetailed results saved to {output_path}")

    # Print a few examples
    print("\n--- Sample translations ---")
    for i in range(min(3, len(test_pairs))):
        print(f"\n[{i+1}] Kurdish:\n  {sources[i]}")
        print(f"    Reference: {references[i]}")
        print(f"    Predicted: {predictions[i]}")


if __name__ == "__main__":
    main()
