# Kurdish Sorani → Swedish Poetry Translation — LLM Fine-tuning

A pipeline for fine-tuning a multilingual LLM to translate Kurdish Sorani (`ckb_Arab`) poems
into Swedish (`swe_Latn`), designed to run on a single **GTX 3060 12 GB VRAM** card.

---

## Architecture Overview

```
Kurdish Sorani poem
        │
        ▼
  [NLLB-200-1.3B]          ← fine-tuned with LoRA on poetry pairs
  (seq2seq translation)
        │
        ▼
  Swedish translation
```

### Why NLLB-200?
- Meta's No Language Left Behind model natively supports **both** `ckb_Arab` and `swe_Latn`
- 1.3B parameter distilled version fits in ~2.6 GB fp16 → plenty of headroom on 12 GB
- Far better starting point than mT5 for a low-resource pair like Kurdish Sorani ↔ Swedish
- LoRA fine-tuning keeps memory usage low and training fast

---

## Project Structure

```
kurdishToSwedish/
├── data/
│   ├── collect_opus.py          # Download OPUS ckb↔swe parallel sentences
│   ├── collect_pivot.py         # Download ckb↔en, pivot-translate en→swe via NLLB
│   ├── generate_synthetic.py    # Use Claude API to create poetry pairs
│   └── preprocess.py            # Clean, deduplicate, train/val/test split
├── training/
│   ├── train_nllb.py            # Main fine-tuning script (LoRA + seq2seq)
│   └── config.yaml              # All hyperparameters in one place
├── evaluation/
│   ├── evaluate.py              # BLEU, chrF, ChrF++ scoring
│   └── translate.py             # Interactive / batch inference
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU       | GTX 3060 12 GB (or any ≥ 10 GB VRAM) |
| RAM       | ≥ 16 GB system RAM |
| Disk      | ≥ 20 GB free (model weights + dataset) |
| CUDA      | 11.8 or 12.x |

**VRAM budget during training (NLLB-200-distilled-1.3B + LoRA):**
| Item | VRAM |
|------|------|
| Model weights (fp16) | ~2.6 GB |
| LoRA adapters | ~0.1 GB |
| Optimizer states (AdamW, fp32 masters) | ~2.0 GB |
| Activations (batch 4, seq 256) | ~2.0 GB |
| Gradient checkpointing savings | −1.5 GB |
| **Total** | **~5–7 GB** ✓ |

---

## Step-by-Step Plan

### Phase 1 — Data Collection

Kurdish Sorani ↔ Swedish is an extremely low-resource pair.
Strategy: layer multiple data sources from largest to smallest.

#### 1a. OPUS General Corpus (automated)
```bash
python data/collect_opus.py
```
Downloads all OPUS sub-corpora that contain `ckb`–`swe` sentence pairs
(CCAligned, WikiMatrix, FLORES-200 dev/devtest, etc.).
Expected yield: **500–5 000 sentence pairs** (mostly news/wiki, not poetry).

#### 1b. Synthetic Poetry Pairs via Claude API (automated)
```bash
python data/generate_synthetic.py --n_poems 500
```
Prompts Claude to:
1. Write a short Kurdish Sorani poem on a given theme.
2. Translate it to Swedish, preserving poetic form.
3. Return structured JSON: `{ckb, swe, theme, form}`.

Expected yield: **500+ high-quality poetry pairs** in ~1 hour.

#### 1c. Pivot Translation via English (automated)
```bash
python data/collect_pivot.py --n_pairs 5000
```
Kurdish Sorani ↔ English has far more parallel data than Kurdish ↔ Swedish
directly. This script:
1. Downloads all available `ckb`↔`en` sentence pairs from OPUS (CCAligned,
   WikiMatrix, FLORES-200, TED2020, etc.).
2. Pivot-translates the English side to Swedish using the off-the-shelf
   `facebook/nllb-200-distilled-600M` model (no GPU required, but much faster
   with one).
3. Saves `data/raw/pivot_ckb_swe.jsonl` — picked up automatically by
   `preprocess.py`.

Expected yield: **1 000–20 000 additional ckb↔swe pairs** (labelled
`source=pivot-en/<corpus>`). Quality is lower than direct translations, so
these pairs supplement rather than replace synthetic or manual data.

```bash
# CPU-only quick test (500 pairs):
python data/collect_pivot.py --n_pairs 500 --device cpu

# GPU, all available pairs:
python data/collect_pivot.py

# Only download raw ckb↔en without translating:
python data/collect_pivot.py --download_only
```

#### 1d. Manual / Scraped Poetry (manual effort)
Suggested sources:
- **Kurdish Academy of Language** (kurdishacademy.org) — Sorani poems
- **Chwarçêwe magazine** — Kurdish literary magazine
- **Almasah** (almasah.net) — Kurdish poetry portal
- Ask Kurdish diaspora communities for bilingual poetry books

Even **50–100 manually curated poem pairs** will dramatically improve quality.

#### 1d. Preprocessing
```bash
python data/preprocess.py
```
- Remove duplicates, empty lines, lines that are identical in both languages
- Normalize Arabic/Kurdish Unicode (NFC normalization)
- Filter by length ratio (0.3–3.0)
- Split: 80% train / 10% val / 10% test
- Save as `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

Format of each line:
```json
{"ckb": "...", "swe": "...", "source": "synthetic|opus|manual"}
```

---

### Phase 2 — Model Selection

**Primary: `facebook/nllb-200-distilled-1.3B`**
- Best accuracy/VRAM trade-off for this hardware
- LoRA rank 16 adds ~5 M trainable parameters

**Optional upgrade: `facebook/nllb-200-3.3B`**
Requires 4-bit QLoRA (`bitsandbytes`). Fits in ~7 GB VRAM but trains 2× slower.
Use this only if the 1.3B model produces unsatisfactory results.

---

### Phase 3 — Fine-tuning

```bash
python training/train_nllb.py --config training/config.yaml
```

Key decisions encoded in `config.yaml`:
- **LoRA** on all attention projections (`q_proj`, `v_proj`, `k_proj`, `out_proj`) and FFN gates
- **Gradient checkpointing** enabled (trades compute for memory)
- **bf16** mixed precision (RTX 30-series supports bf16)
- **AdamW 8-bit** (`bitsandbytes`) — halves optimizer memory
- **Batch size 4, gradient accumulation 8** → effective batch 32
- **Linear warmup** for 100 steps, then cosine decay
- **Early stopping** on validation chrF score (patience 5)

Estimated training time on GTX 3060:
| Dataset size | Epochs | Time |
|---|---|---|
| 1 000 pairs | 10 | ~30 min |
| 5 000 pairs | 10 | ~2.5 hrs |
| 20 000 pairs | 5 | ~5 hrs |

---

### Phase 4 — Evaluation

```bash
python evaluation/evaluate.py --split test
```

Metrics:
- **chrF++** — best automatic metric for morphologically rich languages
- **BLEU** — standard, but less reliable for Kurdish
- **Back-translation consistency** — translate swe→ckb and check similarity to source

For poetry specifically, also do **human evaluation** on 20–30 poems:
- Faithfulness to meaning (1–5)
- Naturalness in Swedish (1–5)
- Preservation of poetic feel (1–5)

---

### Phase 5 — Inference

```bash
# Single poem
python evaluation/translate.py --text "ئەی گوڵ ئەی گوڵ ئەی گوڵی سوور"

# Batch file
python evaluation/translate.py --input poems.txt --output translations.txt
```

---

## Recommended Iteration Order

1. **Start small**: generate 200 synthetic pairs, train 3 epochs, check output quality
2. **Expand data**: add OPUS + more synthetic pairs
3. **Add manual poetry**: even 50 real translated poems help significantly
4. **Tune LoRA rank**: try rank 8, 16, 32 — rank 16 is a good default
5. **Try 3.3B model**: if 1.3B plateaus, upgrade with QLoRA

---

## Known Challenges

| Challenge | Mitigation |
|-----------|------------|
| Very few ckb↔swe parallel texts exist | Synthetic data + pivot via English |
| Kurdish Sorani script (Arabic-based, RTL) | NLLB handles this natively |
| Poetry requires style, not just meaning | Fine-tune on poetry-specific pairs |
| Dialectal variation in Sorani | Document which dialect your data uses |
| Swedish poetic register differs from prose | Include Swedish poetry examples in prompt |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data
python data/collect_opus.py
python data/collect_pivot.py --n_pairs 5000   # pivot ckb↔en → ckb↔swe
python data/generate_synthetic.py --n_poems 300

# 3. Preprocess
python data/preprocess.py

# 4. Train
python training/train_nllb.py --config training/config.yaml

# 5. Evaluate
python evaluation/evaluate.py --split test

# 6. Translate
python evaluation/translate.py --text "your kurdish poem here"
```
