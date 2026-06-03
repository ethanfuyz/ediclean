from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.ediclean.cleaners import RuleCleaner, build_cleaner
from src.ediclean.data import (
    build_noisy_clean_pairs,
    extract_texts,
    extract_texts_labels,
    load_cleaner_texts,
    load_text_classification_dataset,
)
from src.ediclean.eval import add_comparison_metrics, add_recovery_metrics, compute_classification_metrics
from src.ediclean.models import build_downstream_model
from src.ediclean.noise import add_noise_batch
from src.ediclean.utils import ensure_dir, load_config, load_json, save_json, set_seed, stable_hash


def maybe_cache_texts(cache_path: Path, generator_fn, use_cache: bool = True) -> List[str]:
    if use_cache and cache_path.exists():
        return load_json(cache_path)
    texts = generator_fn()
    save_json(texts, cache_path)
    return texts


def maybe_cache_pairs(cache_path: Path, generator_fn, use_cache: bool = True):
    if use_cache and cache_path.exists():
        data = load_json(cache_path)
        from datasets import Dataset
        return Dataset.from_dict(data)
    ds = generator_fn()
    save_json({"noisy_text": list(ds["noisy_text"]), "clean_text": list(ds["clean_text"])}, cache_path)
    return ds


def evaluate_model(model, texts, labels, label_names, clean_reference_metrics=None, noisy_reference_metrics=None):
    """Evaluate with both hard predictions and confidence/calibration metrics.

    All downstream model classes implement predict_proba(), returning probabilities ordered
    by sorted label ids. The hard prediction is the argmax of these probabilities.
    """
    probs = model.predict_proba(texts)
    labels_sorted = sorted(int(k) for k in label_names.keys())
    preds = [labels_sorted[int(max(range(len(p)), key=lambda i: p[i]))] for p in probs]
    metrics = compute_classification_metrics(labels, preds, label_names, y_proba=probs)
    invalid_count = sum(1 for p in preds if p not in label_names)
    metrics["invalid_prediction_count"] = invalid_count
    metrics["invalid_prediction_rate"] = invalid_count / max(len(preds), 1)
    if clean_reference_metrics is not None and noisy_reference_metrics is None:
        metrics = add_comparison_metrics(clean_reference_metrics, metrics)
    if clean_reference_metrics is not None and noisy_reference_metrics is not None:
        metrics = add_recovery_metrics(clean_reference_metrics, noisy_reference_metrics, metrics)
    return metrics, preds, probs



def format_placeholders(obj, **kwargs):
    """Recursively format string placeholders in config dictionaries/lists.

    Example run_name: "distilbert_{dataset_key}" becomes "distilbert_imdb".
    """
    if isinstance(obj, dict):
        return {k: format_placeholders(v, **kwargs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_placeholders(v, **kwargs) for v in obj]
    if isinstance(obj, str):
        try:
            return obj.format(**kwargs)
        except Exception:
            return obj
    return obj


def make_noisy_split(texts: List[str], noise_cfg: Dict, seed: int, cache_dir: Path, tag: str, dataset_cfg: Dict):
    noise_hash = stable_hash({"tag": tag, "noise": noise_cfg, "dataset": dataset_cfg, "seed": seed})
    cache_path = cache_dir / f"{tag}_noisy_{noise_hash}.json"
    noisy = maybe_cache_texts(cache_path, lambda: add_noise_batch(texts, noise_cfg, seed), use_cache=True)
    return noisy, cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    downstream_model_dir = ensure_dir(cfg["paths"].get("downstream_model_dir", cfg["paths"].get("model_dir", "models/downstream")))
    cleaner_model_dir = ensure_dir(cfg["paths"].get("cleaner_model_dir", "models/cleaners"))
    cache_dir = ensure_dir(cfg["paths"].get("cache_dir", "cache"))

    print("\n========== Step 1: Load downstream text-classification dataset ==========")
    train_ds, val_ds, test_ds, label_names = load_text_classification_dataset(cfg)
    resolved_ds_cfg = cfg.get("downstream_dataset_resolved", cfg["downstream_dataset"])
    text_field = resolved_ds_cfg["text_field"]
    label_field = resolved_ds_cfg["label_field"]

    train_clean_texts, train_labels = extract_texts_labels(train_ds, text_field, label_field)
    val_clean_texts, val_labels = extract_texts_labels(val_ds, text_field, label_field)
    test_clean_texts = extract_texts(test_ds, text_field) if test_ds is not None else []

    print(f"Resolved downstream dataset: {resolved_ds_cfg}")
    print(f"Downstream train examples: {len(train_clean_texts)}")
    print(f"Downstream validation examples: {len(val_clean_texts)}")
    print(f"Downstream test examples: {len(test_clean_texts)}")
    print(f"Validation label counts: {dict((x, val_labels.count(x)) for x in sorted(set(val_labels)))}")
    dataset_key = resolved_ds_cfg.get("dataset_key") or resolved_ds_cfg.get("provider", "dataset").replace("/", "_")
    cfg["models"] = format_placeholders(cfg["models"], dataset_key=dataset_key)
    cfg["cleaner_models"] = format_placeholders(cfg.get("cleaner_models", {}), dataset_key=dataset_key)

    print("\n========== Step 2: Generate human-like noisy downstream splits ==========")
    noise_cfg = cfg["noise"]
    train_noisy_texts, train_noisy_cache = make_noisy_split(train_clean_texts, noise_cfg, seed + 1, cache_dir, "downstream_train", resolved_ds_cfg)
    val_noisy_texts, val_noisy_cache = make_noisy_split(val_clean_texts, noise_cfg, seed + 2, cache_dir, "downstream_validation", resolved_ds_cfg)
    test_noisy_cache = None
    if test_clean_texts:
        _, test_noisy_cache = make_noisy_split(test_clean_texts, noise_cfg, seed + 3, cache_dir, "downstream_test", resolved_ds_cfg)

    noise_examples = [
        {"clean": c, "noisy": n, "label": int(y)}
        for c, n, y in zip(val_clean_texts[:50], val_noisy_texts[:50], val_labels[:50])
    ]
    save_json(noise_examples, out_dir / "noise_examples_downstream_validation.json")

    print("\n========== Step 3: Train/load downstream sentiment models ==========")
    trained_downstream = []
    rows = []
    predictions = {}

    for key, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            continue

        print(f"\n----- Downstream model: {model_cfg['run_name']} ({model_cfg['architecture']}) -----")
        model = build_downstream_model(model_cfg, cfg["downstream_training"], text_field, label_field, downstream_model_dir)
        model.train_or_load(train_ds, label_names)

        print(f"Evaluating clean validation: {model_cfg['run_name']}")
        clean_metrics, clean_preds, clean_probs = evaluate_model(model, val_clean_texts, val_labels, label_names)

        print(f"Evaluating noisy validation: {model_cfg['run_name']}")
        noisy_metrics, noisy_preds, noisy_probs = evaluate_model(model, val_noisy_texts, val_labels, label_names, clean_reference_metrics=clean_metrics)

        row = {
            "model_name": model_cfg["run_name"],
            "architecture": model_cfg["architecture"],
            "base_model_id": model_cfg["model_id"],
            "non_noise": clean_metrics,
            "noise": noisy_metrics,
        }
        rows.append(row)
        predictions[model_cfg["run_name"]] = {"clean": clean_preds, "noisy": noisy_preds, "clean_probs": clean_probs, "noisy_probs": noisy_probs}
        trained_downstream.append({"cfg": model_cfg, "model": model, "row": row})

    print("\n========== Step 4: Prepare noisy-clean pairs for cleaner training ==========")
    cleaner_train_texts, cleaner_val_texts, cleaner_test_texts, cleaner_source_meta = load_cleaner_texts(
        cfg,
        downstream_train_texts=train_clean_texts,
        downstream_val_texts=val_clean_texts,
    )
    print(f"Cleaner source: {cleaner_source_meta}")
    print(f"Cleaner train clean texts: {len(cleaner_train_texts)}")
    print(f"Cleaner validation clean texts: {len(cleaner_val_texts)}")
    print(f"Cleaner test clean texts: {len(cleaner_test_texts)}")

    cleaner_noise_cfg = cfg.get("cleaner_noise", noise_cfg)
    cleaner_pair_hash = stable_hash({
        "cleaner_dataset": cfg["cleaner_dataset"],
        "cleaner_source_meta": cleaner_source_meta,
        "cleaner_noise": cleaner_noise_cfg,
        "seed": seed,
    })
    cleaner_train_pairs_cache = cache_dir / f"cleaner_train_pairs_{cleaner_pair_hash}.json"
    cleaner_val_pairs_cache = cache_dir / f"cleaner_val_pairs_{cleaner_pair_hash}.json"

    cleaner_train_pairs = maybe_cache_pairs(
        cleaner_train_pairs_cache,
        lambda: build_noisy_clean_pairs(cleaner_train_texts, cleaner_noise_cfg, seed + 100),
        use_cache=True,
    )
    cleaner_val_pairs = maybe_cache_pairs(
        cleaner_val_pairs_cache,
        lambda: build_noisy_clean_pairs(cleaner_val_texts, cleaner_noise_cfg, seed + 101),
        use_cache=True,
    )

    save_json(
        [
            {"noisy": n, "clean": c}
            for n, c in zip(list(cleaner_val_pairs["noisy_text"])[:50], list(cleaner_val_pairs["clean_text"])[:50])
        ],
        out_dir / "noise_examples_cleaner_dataset_validation.json",
    )

    print("\n========== Step 5: Train/load neural cleaner models ==========")
    cleaner_outputs: Dict[str, List[str]] = {}
    cleaner_metadata: Dict[str, Dict] = {}

    for key, cleaner_cfg in cfg.get("cleaner_models", {}).items():
        if not cleaner_cfg.get("enabled", True):
            continue
        print(f"\n----- Cleaner model: {cleaner_cfg['run_name']} ({cleaner_cfg['architecture']}) -----")
        cleaner = build_cleaner(cleaner_cfg, cfg["cleaner_training"], cleaner_model_dir)
        cleaner.train_or_load(cleaner_train_pairs)

        clean_hash = stable_hash({"cleaner_cfg": cleaner_cfg, "val_noisy_cache": str(val_noisy_cache)})
        cleaned_cache = cache_dir / f"downstream_validation_cleaned_by_{cleaner_cfg['run_name']}_{clean_hash}.json"
        cleaned_texts = maybe_cache_texts(
            cleaned_cache,
            lambda cleaner=cleaner: cleaner.clean(val_noisy_texts),
            use_cache=bool(cleaner_cfg.get("use_cache", True)),
        )
        cleaner_outputs[cleaner_cfg["run_name"]] = cleaned_texts
        cleaner_metadata[cleaner_cfg["run_name"]] = {
            "architecture": cleaner_cfg["architecture"],
            "base_model_id": cleaner_cfg["model_id"],
            "trained_on": "cleaner_dataset noisy-clean pairs",
            "cache_file": str(cleaned_cache),
        }

    print("\n========== Step 6: Algorithmic cleaning baseline ==========")
    if cfg.get("algorithmic_cleaner", {}).get("enabled", True):
        alg_cfg = cfg["algorithmic_cleaner"]
        alg_hash = stable_hash({"algorithmic_cleaner": alg_cfg, "val_noisy_cache": str(val_noisy_cache)})
        alg_cache = cache_dir / f"downstream_validation_algorithmic_{alg_hash}.json"
        alg_texts = maybe_cache_texts(
            alg_cache,
            lambda: RuleCleaner(alg_cfg).clean(val_noisy_texts),
            use_cache=bool(alg_cfg.get("use_cache", True)),
        )
        cleaner_outputs[alg_cfg.get("name", "ruleclean")] = alg_texts
        cleaner_metadata[alg_cfg.get("name", "ruleclean")] = {
            "architecture": "algorithmic",
            "base_model_id": "none",
            "trained_on": "not trained",
            "cache_file": str(alg_cache),
        }

    print("\n========== Step 7: Evaluate cleaned validation inputs ==========")
    for item in trained_downstream:
        model_cfg = item["cfg"]
        model = item["model"]
        row = item["row"]
        clean_metrics = row["non_noise"]
        noisy_metrics = row["noise"]

        for cleaner_name, cleaned_texts in cleaner_outputs.items():
            print(f"Evaluating {model_cfg['run_name']} on {cleaner_name}-cleaned validation")
            metrics, preds, probs = evaluate_model(
                model,
                cleaned_texts,
                val_labels,
                label_names,
                clean_reference_metrics=clean_metrics,
                noisy_reference_metrics=noisy_metrics,
            )
            row[cleaner_name] = metrics
            predictions[model_cfg["run_name"]][cleaner_name] = preds
            predictions[model_cfg["run_name"]][cleaner_name + "_probs"] = probs

    result = {
        "project_name": cfg.get("project_name", "ediclean"),
        "downstream_dataset": resolved_ds_cfg,
        "cleaner_dataset": cfg["cleaner_dataset"],
        "cleaner_source_meta": cleaner_source_meta,
        "noise_config_downstream": noise_cfg,
        "noise_config_cleaner_training": cleaner_noise_cfg,
        "downstream_training": cfg["downstream_training"],
        "cleaner_training": cfg["cleaner_training"],
        "downstream_train_examples": len(train_clean_texts),
        "downstream_validation_examples": len(val_clean_texts),
        "downstream_validation_label_counts": {str(x): val_labels.count(x) for x in sorted(set(val_labels))},
        "cleaner_train_examples": len(cleaner_train_pairs),
        "cleaner_validation_examples": len(cleaner_val_pairs),
        "cleaner_models": cleaner_metadata,
        "algorithmic_cleaner_config": cfg.get("algorithmic_cleaner", {}),
        "table_rows": rows,
        "cache_files": {
            "downstream_train_noisy": str(train_noisy_cache),
            "downstream_validation_noisy": str(val_noisy_cache),
            "downstream_test_noisy": str(test_noisy_cache) if test_noisy_cache else None,
            "cleaner_train_pairs": str(cleaner_train_pairs_cache),
            "cleaner_validation_pairs": str(cleaner_val_pairs_cache),
        },
    }

    results_file = Path(cfg["paths"]["results_file"])
    save_json(result, results_file)
    save_json(predictions, Path(cfg["paths"].get("predictions_file", out_dir / "predictions.json")))
    print(f"\nSaved results to: {results_file}")
    print(f"Saved predictions to: {cfg['paths'].get('predictions_file', out_dir / 'predictions.json')}")


if __name__ == "__main__":
    main()
