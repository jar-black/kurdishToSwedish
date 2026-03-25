"""
translate.py
------------
Interactive and batch inference for the fine-tuned Kurdish Sorani → Swedish
poetry translation model.

Usage (single poem):
    python evaluation/translate.py \
        --model_dir checkpoints/final \
        --text "ئەی گوڵ ئەی گوڵ ئەی گوڵی سوور"

Usage (batch from file, one poem per line or blank-line separated stanzas):
    python evaluation/translate.py \
        --model_dir checkpoints/final \
        --input poems.txt \
        --output translations.txt

Usage (interactive REPL):
    python evaluation/translate.py --model_dir checkpoints/final --interactive
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SRC_LANG = "ckb_Arab"
TGT_LANG = "swe_Latn"


def load_model(model_dir: str, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer


def translate(
    model,
    tokenizer,
    text: str,
    num_beams: int = 5,
    max_length: int = 512,
) -> str:
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def read_poems(path: str) -> list[str]:
    """Read poems separated by blank lines."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [block.strip() for block in content.split("\n\n") if block.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/final")
    parser.add_argument("--base_model", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--text", help="Single poem text to translate")
    parser.add_argument("--input", help="File with poems (blank-line separated)")
    parser.add_argument("--output", help="File to write translations to")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL")
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model, tokenizer = load_model(args.model_dir, args.base_model)
    print("Model ready.\n")

    if args.text:
        result = translate(model, tokenizer, args.text, num_beams=args.num_beams)
        print(result)

    elif args.input:
        poems = read_poems(args.input)
        print(f"Translating {len(poems)} poems...")
        translations = [translate(model, tokenizer, p, num_beams=args.num_beams) for p in poems]

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for i, (poem, trans) in enumerate(zip(poems, translations)):
                    f.write(f"=== Poem {i+1} (Kurdish) ===\n{poem}\n\n")
                    f.write(f"=== Poem {i+1} (Swedish) ===\n{trans}\n\n")
            print(f"Saved to {args.output}")
        else:
            for i, (poem, trans) in enumerate(zip(poems, translations)):
                print(f"\n--- Poem {i+1} ---")
                print(f"Kurdish:\n{poem}")
                print(f"\nSwedish:\n{trans}")

    elif args.interactive:
        print("Interactive mode. Paste your Kurdish poem, then press Enter twice to translate.")
        print("Type 'quit' to exit.\n")
        while True:
            lines = []
            print("Kurdish > ", end="", flush=True)
            while True:
                line = sys.stdin.readline()
                if not line or line.strip() == "quit":
                    print("Bye!")
                    return
                if line.strip() == "":
                    if lines:
                        break
                else:
                    lines.append(line.rstrip())

            poem = "\n".join(lines)
            result = translate(model, tokenizer, poem, num_beams=args.num_beams)
            print(f"\nSwedish >\n{result}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
