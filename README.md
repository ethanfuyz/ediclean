# EdiClean v2

This project runs a full noisy-input robustness pipeline for small Transformer NLP models.

## What it does

1. Loads a downstream sentiment dataset. Default: **SST-2**.
2. Fine-tunes three downstream models:
   - Encoder-only: DistilBERT
   - Decoder-only: DistilGPT2
   - Encoder-decoder: T5-small
3. Converts downstream train/validation/test text into **human-like noisy** versions.
4. Evaluates the three downstream models on:
   - clean validation input
   - noisy validation input
5. Loads a separate non-sentiment clean-text dataset. Default: **WikiText-2**.
6. Converts WikiText into noisy-clean pairs using the same human-like noise generator.
7. Fine-tunes two neural cleaners:
   - Encoder-decoder cleaner: T5-small
   - Decoder-only cleaner: DistilGPT2
8. Cleans noisy downstream validation with:
   - encoder-decoder cleaner
   - decoder-only cleaner
   - pure algorithmic RuleClean
9. Evaluates cleaned validation input with the three downstream models.
10. Saves JSON results for paper tables.

## Run

```bash
pip install -r requirements.txt
python run_experiment.py --config configs/default.yaml
```

Results are saved to:

```text
outputs/results.json
outputs/predictions.json
outputs/noise_examples_downstream_validation.json
outputs/noise_examples_cleaner_dataset_validation.json
```

## Key config switches

Edit `configs/default.yaml`.

### Downstream dataset

Default: SST-2.

Alternatives are documented in comments, including IMDB and Yelp Polarity. For non-binary datasets, update prompts and label mappings.

### Cleaner dataset

Default: WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`). It is not a sentiment classification dataset and is used only to train noisy-to-clean cleaners.

Alternatives are listed in config comments, such as WikiText-103, AG News text, CNN/DailyMail articles, and Yelp Review Full text.

### Noise

Default noise type is `human_like`, which mixes:

- spelling / typo noise: character swap, delete, insert, keyboard-neighbor substitution, repeated characters
- punctuation deletion
- function-word deletion
- local word-order swap
- lowercasing

The generator protects negation words and common sentiment-bearing words from deletion to reduce label flips.

### Cleaner comparison

Default model-based cleaners:

- `ediclean_t5small_wikitext`: encoder-decoder cleaner
- `decoderclean_distilgpt2_wikitext`: decoder-only cleaner

RuleClean is a non-neural baseline using whitespace normalization, repeated-character reduction, and dictionary spell checking.

## Slurm

A sample Slurm script is included as `ediclean.sh`.
