from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from .utils import get_device, model_exists


def _label_name(label: int, label_names: Dict[int, str]) -> str:
    return label_names[int(label)]


def _softmax_np(x: List[float]) -> List[float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return (exp / np.maximum(exp.sum(), 1e-12)).tolist()


class BaseDownstreamModel:
    def train_or_load(self, train_ds: Dataset, label_names: Dict[int, str]) -> None:
        raise NotImplementedError

    def predict(self, texts: List[str]) -> List[int]:
        probs = self.predict_proba(texts)
        return [int(np.argmax(p)) for p in probs]

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class EncoderOnlyClassifier(BaseDownstreamModel):
    def __init__(self, model_cfg: Dict[str, Any], train_cfg: Dict[str, Any], text_field: str, label_field: str, model_dir: Path):
        self.cfg = model_cfg
        self.train_cfg = train_cfg
        self.text_field = text_field
        self.label_field = label_field
        self.path = model_dir / model_cfg["run_name"]
        self.max_length = int(model_cfg.get("max_length", 128))
        self.batch_size = int(model_cfg.get("eval_batch_size", 32))
        self.device = get_device()
        self.tokenizer = None
        self.model = None

    def train_or_load(self, train_ds: Dataset, label_names: Dict[int, str]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_id"] if not model_exists(self.path) else self.path)
        if model_exists(self.path):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path).to(self.device)
            return

        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg["model_id"], num_labels=len(label_names)).to(self.device)

        def tok(batch):
            return self.tokenizer(batch[self.text_field], truncation=True, max_length=self.max_length)

        ds = train_ds.map(tok, batched=True)
        ds = ds.rename_column(self.label_field, "labels") if self.label_field != "labels" else ds
        cols = ["input_ids", "attention_mask", "labels"]
        ds.set_format(type="torch", columns=[c for c in cols if c in ds.column_names])

        args = TrainingArguments(
            output_dir=str(self.path),
            num_train_epochs=self.train_cfg["num_train_epochs"],
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            learning_rate=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.0),
            logging_steps=self.train_cfg.get("logging_steps", 50),
            save_total_limit=self.train_cfg.get("save_total_limit", 1),
            fp16=bool(self.train_cfg.get("fp16", False)),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 1),
            report_to="none",
            save_strategy="epoch",
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(self.tokenizer))
        trainer.train()
        trainer.save_model(str(self.path))
        self.tokenizer.save_pretrained(str(self.path))

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        self.model.eval()
        probs: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=self.cfg["run_name"] + ":predict_proba"):
            enc = self.tokenizer(
                texts[i:i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            batch_probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            probs.extend(batch_probs)
        return [[float(x) for x in row] for row in probs]


class DecoderOnlyClassifier(BaseDownstreamModel):
    def __init__(self, model_cfg: Dict[str, Any], train_cfg: Dict[str, Any], text_field: str, label_field: str, model_dir: Path):
        self.cfg = model_cfg
        self.train_cfg = train_cfg
        self.text_field = text_field
        self.label_field = label_field
        self.path = model_dir / model_cfg["run_name"]
        self.max_length = int(model_cfg.get("max_length", 160))
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.label_texts = {int(k): v for k, v in model_cfg["label_texts"].items()}

    def train_or_load(self, train_ds: Dataset, label_names: Dict[int, str]) -> None:
        load_from = self.path if model_exists(self.path) else self.cfg["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model_exists(self.path):
            self.model = AutoModelForCausalLM.from_pretrained(self.path).to(self.device)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            return

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg["model_id"]).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        prompt_tmpl = self.cfg["prompt_template"]
        train_tmpl = self.cfg["train_template"]

        def tok(ex):
            label_word = _label_name(ex[self.label_field], label_names)
            full = train_tmpl.format(text=ex[self.text_field], label=label_word)
            prompt = prompt_tmpl.format(text=ex[self.text_field])
            full_ids = self.tokenizer(full, truncation=True, max_length=self.max_length)["input_ids"]
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
            labels = full_ids.copy()
            labels[:len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
            return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

        ds = train_ds.map(tok, remove_columns=train_ds.column_names)

        def collate(features):
            max_len = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in features:
                pad = max_len - len(f["input_ids"])
                batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
                batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
                batch["labels"].append(f["labels"] + [-100] * pad)
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

        args = TrainingArguments(
            output_dir=str(self.path),
            num_train_epochs=self.train_cfg["num_train_epochs"],
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            learning_rate=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.0),
            logging_steps=self.train_cfg.get("logging_steps", 50),
            save_total_limit=self.train_cfg.get("save_total_limit", 1),
            fp16=bool(self.train_cfg.get("fp16", False)),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 1),
            report_to="none",
            save_strategy="epoch",
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=ds, data_collator=collate)
        trainer.train()
        trainer.save_model(str(self.path))
        self.tokenizer.save_pretrained(str(self.path))

    @torch.no_grad()
    def _label_logprob(self, prompt: str, label_text: str) -> float:
        full = prompt + label_text
        full_ids = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=self.max_length).input_ids.to(self.device)
        outputs = self.model(full_ids)
        logits = outputs.logits[:, :-1, :]
        target = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        label_len = len(self.tokenizer(label_text, add_special_tokens=False).input_ids)
        if label_len <= 0:
            return -1e9
        label_len = min(label_len, target.shape[1])
        lp = log_probs[:, -label_len:, :].gather(-1, target[:, -label_len:].unsqueeze(-1)).squeeze(-1).sum()
        # Average by label length to avoid tiny preference for shorter verbalizers.
        return float((lp / max(label_len, 1)).cpu())

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        self.model.eval()
        rows: List[List[float]] = []
        labels = sorted(self.label_texts.keys())
        for text in tqdm(texts, desc=self.cfg["run_name"] + ":predict_proba"):
            prompt = self.cfg["prompt_template"].format(text=text)
            scores = [self._label_logprob(prompt, self.label_texts[label]) for label in labels]
            rows.append(_softmax_np(scores))
        return rows


class EncoderDecoderClassifier(BaseDownstreamModel):
    def __init__(self, model_cfg: Dict[str, Any], train_cfg: Dict[str, Any], text_field: str, label_field: str, model_dir: Path):
        self.cfg = model_cfg
        self.train_cfg = train_cfg
        self.text_field = text_field
        self.label_field = label_field
        self.path = model_dir / model_cfg["run_name"]
        self.max_source_length = int(model_cfg.get("max_source_length", 128))
        self.max_target_length = int(model_cfg.get("max_target_length", 8))
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.label_texts = {int(k): v for k, v in model_cfg["label_texts"].items()}

    def train_or_load(self, train_ds: Dataset, label_names: Dict[int, str]) -> None:
        load_from = self.path if model_exists(self.path) else self.cfg["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        if model_exists(self.path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(self.device)
            return

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg["model_id"]).to(self.device)
        prefix = self.cfg.get("source_prefix", "")

        def tok(batch):
            sources = [prefix + str(x) for x in batch[self.text_field]]
            targets = [self.label_texts[int(y)] for y in batch[self.label_field]]
            model_inputs = self.tokenizer(sources, max_length=self.max_source_length, truncation=True)
            labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = train_ds.map(tok, batched=True, remove_columns=train_ds.column_names)
        args = Seq2SeqTrainingArguments(
            output_dir=str(self.path),
            num_train_epochs=self.train_cfg["num_train_epochs"],
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            learning_rate=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_ratio=self.train_cfg.get("warmup_ratio", 0.0),
            logging_steps=self.train_cfg.get("logging_steps", 50),
            save_total_limit=self.train_cfg.get("save_total_limit", 1),
            fp16=bool(self.train_cfg.get("fp16", False)),
            gradient_accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 1),
            report_to="none",
            save_strategy="epoch",
            predict_with_generate=True,
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, self.model),
        )
        trainer.train()
        trainer.save_model(str(self.path))
        self.tokenizer.save_pretrained(str(self.path))

    @torch.no_grad()
    def _label_logprob(self, source_text: str, label_text: str) -> float:
        prefix = self.cfg.get("source_prefix", "")
        enc = self.tokenizer(
            prefix + source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)
        labels = self.tokenizer(
            text_target=label_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids.to(self.device)

        out = self.model(**enc, labels=labels)
        logits = out.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = labels.ne(self.tokenizer.pad_token_id)
        gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        total = (gathered * mask).sum()
        length = mask.sum().clamp(min=1)
        return float((total / length).cpu())

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        self.model.eval()
        rows: List[List[float]] = []
        labels = sorted(self.label_texts.keys())
        for text in tqdm(texts, desc=self.cfg["run_name"] + ":predict_proba"):
            scores = [self._label_logprob(text, self.label_texts[label]) for label in labels]
            rows.append(_softmax_np(scores))
        return rows


def build_downstream_model(model_cfg: Dict[str, Any], train_cfg: Dict[str, Any], text_field: str, label_field: str, model_dir: Path) -> BaseDownstreamModel:
    arch = model_cfg["architecture"]
    if arch == "encoder_only":
        return EncoderOnlyClassifier(model_cfg, train_cfg, text_field, label_field, model_dir)
    if arch == "decoder_only":
        return DecoderOnlyClassifier(model_cfg, train_cfg, text_field, label_field, model_dir)
    if arch == "encoder_decoder":
        return EncoderDecoderClassifier(model_cfg, train_cfg, text_field, label_field, model_dir)
    raise ValueError(f"Unknown architecture: {arch}")
