from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from .noise import add_noise_batch


DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    # Short sentence-level binary sentiment. Fastest and most stable for quick tests.
    "sst2": {
        "provider": "glue",
        "name": "sst2",
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "text_field": "sentence",
        "label_field": "label",
        "label_names": {0: "negative", 1: "positive"},
        "task_type": "sentiment",
    },
    # Long movie reviews. Has train/test only; use test as validation or split train manually.
    "imdb": {
        "provider": "imdb",
        "name": None,
        "train_split": "train",
        "validation_split": "test",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_names": {0: "negative", 1: "positive"},
        "task_type": "sentiment",
    },
    # Binary review sentiment; larger than SST-2 and shorter than IMDB on average.
    "yelp_polarity": {
        "provider": "yelp_polarity",
        "name": None,
        "train_split": "train",
        "validation_split": "test",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_names": {0: "negative", 1: "positive"},
        "task_type": "sentiment",
    },
    # Sentence-level movie-review snippets. Small and easy to run.
    "rotten_tomatoes": {
        "provider": "rotten_tomatoes",
        "name": None,
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_names": {0: "negative", 1: "positive"},
        "task_type": "sentiment",
    },
    # Large binary product-review sentiment. Use max_train_samples first.
    "amazon_polarity": {
        "provider": "amazon_polarity",
        "name": None,
        "train_split": "train",
        "validation_split": "test",
        "test_split": "test",
        "text_field": "content",
        "label_field": "label",
        "label_names": {0: "negative", 1: "positive"},
        "task_type": "sentiment",
    },
}


def _merge_preset(ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dataset_key preset with explicit config values.

    Explicit values in downstream_dataset override preset values. This lets you switch
    datasets by changing only `dataset_key`, while still allowing custom HF datasets.
    """
    key = ds_cfg.get("dataset_key")
    if key:
        if key not in DATASET_PRESETS:
            raise ValueError(f"Unknown dataset_key={key}. Available: {sorted(DATASET_PRESETS)}")
        merged = dict(DATASET_PRESETS[key])
        merged.update({k: v for k, v in ds_cfg.items() if v != "auto"})
        return merged
    return dict(ds_cfg)


def _load_hf_dataset(ds_cfg: Dict[str, Any]) -> DatasetDict:
    provider = ds_cfg.get("provider")
    name = ds_cfg.get("name")
    if not provider:
        raise ValueError("Dataset config must contain provider, or set dataset_key to a preset.")

    if provider == "glue":
        # GLUE moved under nyu-mll/glue in newer datasets/hub versions.
        return load_dataset("nyu-mll/glue", name)
    if name is not None:
        return load_dataset(provider, name)
    return load_dataset(provider)


def _maybe_shuffle(ds: Dataset, ds_cfg: Dict[str, Any], seed: int) -> Dataset:
    if bool(ds_cfg.get("shuffle_before_select", True)):
        return ds.shuffle(seed=seed)
    return ds


def _balanced_select(ds: Dataset, label_field: str, max_samples: Any, seed: int) -> Dataset:
    """Select a balanced subset across labels.

    This fixes datasets such as IMDB where the first N test examples may contain only one
    label. If max_samples is None, we still return a shuffled dataset without downsampling.
    """
    if max_samples is None:
        return ds.shuffle(seed=seed)

    max_samples = min(int(max_samples), len(ds))
    labels = sorted(set(int(x) for x in ds[label_field]))
    if not labels:
        return ds.select(range(max_samples))

    per_label = max(1, max_samples // len(labels))
    # Pure-Python deterministic shuffle of original indices. This avoids relying on
    # internal datasets.Dataset index-table representations.
    import random
    selected = []
    rng = random.Random(seed)
    for label in labels:
        idxs = [i for i, y in enumerate(ds[label_field]) if int(y) == label]
        rng.shuffle(idxs)
        selected.extend(idxs[: min(per_label, len(idxs))])

    # Fill any remainder while keeping mixed labels.
    if len(selected) < max_samples:
        remaining = [i for i in range(len(ds)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])

    rng.shuffle(selected)
    selected = selected[:max_samples]
    return ds.select(selected)


def _select_split(ds: Dataset, ds_cfg: Dict[str, Any], max_key: str, label_field: str, seed: int, *, split_name: str) -> Dataset:
    max_samples = ds_cfg.get(max_key)
    if bool(ds_cfg.get("balanced_sampling", True)) and label_field in ds.column_names:
        return _balanced_select(ds, label_field, max_samples, seed)
    ds = _maybe_shuffle(ds, ds_cfg, seed)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


def _make_validation_from_train(train_ds: Dataset, ds_cfg: Dict[str, Any], seed: int) -> Tuple[Dataset, Dataset]:
    val_fraction = float(ds_cfg.get("validation_fraction", 0.1))
    train_ds = train_ds.shuffle(seed=seed)
    n_val = max(1, int(len(train_ds) * val_fraction))
    val_ds = train_ds.select(range(n_val))
    new_train_ds = train_ds.select(range(n_val, len(train_ds)))
    return new_train_ds, val_ds


def load_text_classification_dataset(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset | None, Dict[int, str]]:
    """Load and preprocess a text classification dataset.

    Supported presets can be selected with `downstream_dataset.dataset_key`:
    sst2, imdb, yelp_polarity, rotten_tomatoes, amazon_polarity.

    The function handles:
    - HF dataset loading
    - split selection / optional validation split creation
    - shuffle before selecting subsets
    - balanced subset selection to avoid one-class validation bugs
    - label-name normalization for downstream prompts and metrics
    """
    seed = int(cfg.get("seed", 42))
    ds_cfg = _merge_preset(cfg["downstream_dataset"])
    dataset = _load_hf_dataset(ds_cfg)

    train_split = ds_cfg.get("train_split", "train")
    val_split = ds_cfg.get("validation_split", "validation")
    test_split = ds_cfg.get("test_split", "test")
    label_field = ds_cfg.get("label_field", "label")

    if train_split not in dataset:
        raise ValueError(f"Train split {train_split!r} not found. Available splits: {list(dataset.keys())}")
    train_ds = dataset[train_split]

    if val_split in dataset:
        val_ds = dataset[val_split]
    elif bool(ds_cfg.get("create_validation_from_train", False)):
        train_ds, val_ds = _make_validation_from_train(train_ds, ds_cfg, seed)
    else:
        raise ValueError(
            f"Validation split {val_split!r} not found. Available splits: {list(dataset.keys())}. "
            f"Set create_validation_from_train: true or choose an existing validation_split."
        )

    test_ds = dataset[test_split] if test_split in dataset else None

    train_ds = _select_split(train_ds, ds_cfg, "max_train_samples", label_field, seed + 11, split_name="train")
    val_ds = _select_split(val_ds, ds_cfg, "max_eval_samples", label_field, seed + 22, split_name="validation")
    if test_ds is not None:
        test_ds = _select_split(test_ds, ds_cfg, "max_test_samples", label_field, seed + 33, split_name="test")

    label_names_raw = ds_cfg.get("label_names", {0: "negative", 1: "positive"})
    label_names = {int(k): str(v) for k, v in label_names_raw.items()}

    # Save resolved config back for downstream cache hashing and report clarity.
    cfg["downstream_dataset_resolved"] = ds_cfg
    return train_ds, val_ds, test_ds, label_names


def extract_texts_labels(ds: Dataset, text_field: str, label_field: str) -> Tuple[List[str], List[int]]:
    texts = [str(x) for x in ds[text_field]]
    labels = [int(x) for x in ds[label_field]]
    return texts, labels


def extract_texts(ds: Dataset, text_field: str) -> List[str]:
    return [str(x) for x in ds[text_field]]


def _limit_texts(texts: List[str], max_samples: Any) -> List[str]:
    if max_samples is None:
        return texts
    return texts[: min(int(max_samples), len(texts))]


def load_cleaner_texts(
    cfg: Dict[str, Any],
    downstream_train_texts: List[str],
    downstream_val_texts: List[str],
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    """Return clean texts used to construct noisy-clean pairs for cleaner training.

    source options:
    - downstream_train: use the current downstream train texts; labels are ignored.
      This is the recommended setting when you want the cleaner to match the downstream
      dataset's text style without using validation/test labels.
    - hf_corpus: use a separate unlabeled HF corpus such as WikiText.
    - downstream_validation: debugging only; leaks validation text and should not be used
      for final experiments.
    """
    seed = int(cfg.get("seed", 42))
    ds_cfg = cfg["cleaner_dataset"]
    source = ds_cfg.get("source", "downstream_train")

    if source == "downstream_train":
        texts = [str(t).strip() for t in downstream_train_texts if str(t).strip()]
        # Shuffle before limiting so IMDB/Yelp ordered labels do not bias cleaner style.
        import random
        rng = random.Random(seed + 700)
        rng.shuffle(texts)
        texts = _limit_texts(texts, ds_cfg.get("max_train_samples"))
        if not texts:
            raise ValueError("No downstream train texts available for cleaner training.")

        val_frac = float(ds_cfg.get("validation_fraction", 0.05))
        n_val = max(1, int(len(texts) * val_frac)) if len(texts) > 20 else max(1, len(texts) // 5)
        n_val = min(n_val, len(texts) - 1) if len(texts) > 1 else 0
        train_texts = texts[:-n_val] if n_val > 0 else texts
        val_texts = texts[-n_val:] if n_val > 0 else []
        val_texts = _limit_texts(val_texts, ds_cfg.get("max_eval_samples"))
        meta = {
            "source": "downstream_train",
            "description": "Cleaner pairs are generated from downstream training text only; labels are ignored.",
            "original_downstream_train_examples": len(downstream_train_texts),
        }
        return train_texts, val_texts, [], meta

    if source == "downstream_validation":
        texts = [str(t).strip() for t in downstream_val_texts if str(t).strip()]
        texts = _limit_texts(texts, ds_cfg.get("max_train_samples"))
        meta = {"source": "downstream_validation", "warning": "Validation leakage; use only for debugging."}
        return texts, [], [], meta

    if source == "hf_corpus":
        ds_cfg_merged = dict(ds_cfg)
        dataset = _load_hf_dataset(ds_cfg_merged)
        text_field = ds_cfg_merged.get("text_field", "text")
        min_chars = int(ds_cfg_merged.get("min_chars", 20))

        def split_texts(split_name: str, max_samples_key: str, offset: int) -> List[str]:
            ds = dataset[split_name].shuffle(seed=seed + offset)
            texts = [str(x).strip() for x in ds[text_field]]
            texts = [t for t in texts if len(t) >= min_chars and not t.startswith("=")]
            return _limit_texts(texts, ds_cfg_merged.get(max_samples_key))

        train_texts = split_texts(ds_cfg_merged.get("train_split", "train"), "max_train_samples", 801)
        val_texts = split_texts(ds_cfg_merged.get("validation_split", "validation"), "max_eval_samples", 802)
        test_texts = split_texts(ds_cfg_merged.get("test_split", "test"), "max_test_samples", 803)
        meta = {"source": "hf_corpus", "provider": ds_cfg_merged.get("provider"), "name": ds_cfg_merged.get("name")}
        return train_texts, val_texts, test_texts, meta

    raise ValueError(f"Unknown cleaner_dataset.source: {source}")


def build_noisy_clean_pairs(clean_texts: List[str], noise_cfg: Dict[str, Any], seed: int) -> Dataset:
    noisy = add_noise_batch(clean_texts, noise_cfg, seed)
    return Dataset.from_dict({"noisy_text": noisy, "clean_text": clean_texts})
