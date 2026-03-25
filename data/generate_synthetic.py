"""
generate_synthetic.py
---------------------
Use the Anthropic Claude API to generate synthetic Kurdish Sorani ↔ Swedish
poetry pairs for fine-tuning data augmentation.

The script asks Claude to:
  1. Write a short poem in Kurdish Sorani on a given theme.
  2. Translate it to Swedish, preserving poetic form and imagery.
  3. Return structured JSON so we can parse it reliably.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python data/generate_synthetic.py --n_poems 300 --output data/raw/synthetic.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic

# ── Themes & forms to sample from ────────────────────────────────────────────

THEMES = [
    "nature and mountains", "homeland and exile", "love and longing",
    "spring and renewal", "loss and grief", "freedom and resistance",
    "mother and child", "rivers and water", "the moon and stars",
    "bread and harvest", "war and peace", "old age and memory",
    "friendship and loyalty", "dawn and hope", "seasons changing",
]

FORMS = [
    "a ghazal (غەزەل) of 5–7 couplets",
    "a short lyric of 3–4 stanzas",
    "a classical qasida fragment of 8 lines",
    "a free-verse poem of 10–15 lines",
    "a quatrain (چارینە) — 4 lines with AABA or AAAA rhyme",
]

SYSTEM_PROMPT = """\
You are an expert poet and translator fluent in Kurdish Sorani and Swedish.
When asked, you write original Kurdish Sorani poetry and provide high-quality
Swedish translations that preserve the poetic spirit, imagery, and form.

You ALWAYS respond with valid JSON and nothing else.
"""

USER_PROMPT_TEMPLATE = """\
Write {form} in Kurdish Sorani on the theme of "{theme}".
Then translate it into Swedish, keeping the poetic register and imagery.

Respond ONLY with this JSON (no markdown fences, no extra text):
{{
  "theme": "{theme}",
  "form": "{form}",
  "ckb": "<the full poem in Kurdish Sorani Arabic script>",
  "swe": "<the Swedish translation>"
}}
"""


def generate_poem(client: anthropic.Anthropic, theme: str, form: str) -> dict | None:
    prompt = USER_PROMPT_TEMPLATE.format(theme=theme, form=form)
    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
        print(f"  Warning: failed to parse response: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_poems", type=int, default=300, help="Number of poems to generate")
    parser.add_argument(
        "--output", default="data/raw/synthetic.jsonl", help="Output JSONL file"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds between API calls (rate limiting)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY environment variable first.")

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    with open(output_path, "a", encoding="utf-8") as f:
        while generated < args.n_poems:
            theme = random.choice(THEMES)
            form = random.choice(FORMS)

            print(f"[{generated+1}/{args.n_poems}] Generating: {form!r} on {theme!r}")
            result = generate_poem(client, theme, form)

            if result and result.get("ckb") and result.get("swe"):
                result["source"] = "synthetic-claude"
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                generated += 1
            else:
                failed += 1
                print(f"  Skipped (failed: {failed} total)")

            time.sleep(args.delay)

    print(f"\nDone. Generated {generated} poems, {failed} failed. Saved to {output_path}")


if __name__ == "__main__":
    main()
