from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from datasets import Dataset
from spellchecker import SpellChecker
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from .utils import get_device, model_exists


class RuleCleaner:
    """Non-neural text cleaner: normalization + dictionary spell correction.

    This intentionally avoids LLMs or neural generation. It is a lightweight baseline for
    surface-level noise.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.spell = SpellChecker() if cfg.get("spellcheck", True) else None
        self.min_token_length = int(cfg.get("min_token_length", 4))
        self.preserve_case = bool(cfg.get("preserve_case", True))

    @staticmethod
    def reduce_repeated_chars_token(token: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", token)

    def _correct_token(self, token: str) -> str:
        if not self.spell or len(token) < self.min_token_length or not token.isalpha():
            return token
        lower = token.lower()
        if lower in self.spell:
            return token
        corrected = self.spell.correction(lower)
        if not corrected:
            return token
        if self.preserve_case and token[:1].isupper():
            corrected = corrected.capitalize()
        return corrected

    def clean_one(self, text: str) -> str:
        if self.cfg.get("normalize_whitespace", True):
            text = re.sub(r"\s+", " ", text).strip()
        if self.cfg.get("lowercase", False):
            text = text.lower()

        chunks = re.findall(r"\w+|[^\w\s]+|\s+", text, flags=re.UNICODE)
        cleaned = []
        for ch in chunks:
            if ch.isalpha():
                tok = ch
                if self.cfg.get("reduce_repeated_chars", True):
                    tok = self.reduce_repeated_chars_token(tok)
                tok = self._correct_token(tok)
                cleaned.append(tok)
            else:
                cleaned.append(ch)
        out = "".join(cleaned)
        if self.cfg.get("normalize_whitespace", True):
            out = re.sub(r"\s+", " ", out).strip()
        return out

    def clean(self, texts: Iterable[str]) -> List[str]:
        return [self.clean_one(t) for t in tqdm(list(texts), desc="RuleClean")]


class EncoderDecoderCleaner:
    """Fine-tunable encoder-decoder cleaner, e.g. T5-small or BART-base.

    Trains on pairs: noisy_text -> clean_text.
    """

    def __init__(self, cfg: Dict[str, Any], train_cfg: Dict[str, Any], model_dir: Path):
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.path = model_dir / cfg["run_name"]
        self.device = get_device()
        self.model_id = cfg["model_id"]
        self.prefix = cfg.get("source_prefix", "")
        self.max_source_length = int(cfg.get("max_source_length", 128))
        self.max_target_length = int(cfg.get("max_target_length", 128))
        self.batch_size = int(cfg.get("clean_batch_size", 16))
        self.num_beams = int(cfg.get("num_beams", 4))
        self.tokenizer = None
        self.model = None

    def train_or_load(self, train_pairs: Dataset) -> None:
        load_from = self.path if model_exists(self.path) else self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        if model_exists(self.path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(self.device)
            self.model.eval()
            return

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device)

        def tok(batch):
            sources = [self.prefix + str(x) for x in batch["noisy_text"]]
            targets = [str(x) for x in batch["clean_text"]]
            model_inputs = self.tokenizer(sources, max_length=self.max_source_length, truncation=True)
            labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = train_pairs.map(tok, batched=True, remove_columns=train_pairs.column_names)
        args = Seq2SeqTrainingArguments(
            output_dir=str(self.path),
            num_train_epochs=self.train_cfg["num_train_epochs"],
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            learning_rate=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg.get("weight_decay", 0.0),
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
        self.model.eval()

    @torch.no_grad()
    def clean(self, texts: Iterable[str]) -> List[str]:
        texts = list(texts)
        outputs: List[str] = []
        self.model.eval()
        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"EdiClean:{self.cfg['run_name']}"):
            batch = [self.prefix + t for t in texts[start:start + self.batch_size]]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
            ).to(self.device)
            generated = self.model.generate(
                **enc,
                max_new_tokens=self.max_target_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs


class DecoderOnlyCleaner:
    """Fine-tunable decoder-only cleaner, e.g. DistilGPT2 or GPT2.

    Trains on prompt+noisy input, but masks the prompt/input part so loss is only on the
    clean output continuation.
    """

    def __init__(self, cfg: Dict[str, Any], train_cfg: Dict[str, Any], model_dir: Path):
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.path = model_dir / cfg["run_name"]
        self.device = get_device()
        self.model_id = cfg["model_id"]
        self.max_length = int(cfg.get("max_length", 192))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 96))
        self.batch_size = int(cfg.get("clean_batch_size", 8))
        self.prompt_template = cfg.get("prompt_template", "Correct the noisy sentence.\nInput: {noisy}\nOutput:")
        self.train_template = cfg.get("train_template", "Correct the noisy sentence.\nInput: {noisy}\nOutput: {clean}")
        self.tokenizer = None
        self.model = None

    def train_or_load(self, train_pairs: Dataset) -> None:
        load_from = self.path if model_exists(self.path) else self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_exists(self.path):
            self.model = AutoModelForCausalLM.from_pretrained(self.path).to(self.device)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.eval()
            return

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        def tok(ex):
            noisy = str(ex["noisy_text"])
            clean = str(ex["clean_text"])
            prompt = self.prompt_template.format(noisy=noisy)
            full = self.train_template.format(noisy=noisy, clean=clean)
            full_ids = self.tokenizer(full, truncation=True, max_length=self.max_length)["input_ids"]
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
            labels = full_ids.copy()
            labels[: len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
            return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

        ds = train_pairs.map(tok, remove_columns=train_pairs.column_names)

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
            weight_decay=self.train_cfg.get("weight_decay", 0.0),
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
        self.model.eval()

    def _extract_output(self, decoded: str, prompt: str) -> str:
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()
        marker = "Output:"
        if marker in decoded:
            return decoded.split(marker, 1)[1].strip()
        return decoded.strip()

    @torch.no_grad()
    def clean(self, texts: Iterable[str]) -> List[str]:
        texts = list(texts)
        outputs: List[str] = []
        self.model.eval()
        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"DecoderClean:{self.cfg['run_name']}"):
            batch_texts = texts[start:start + self.batch_size]
            prompts = [self.prompt_template.format(noisy=t) for t in batch_texts]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            generated = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=int(self.cfg.get("num_beams", 1)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend([self._extract_output(d, p) for d, p in zip(decoded, prompts)])
        return outputs


def build_cleaner(cleaner_cfg: Dict[str, Any], train_cfg: Dict[str, Any], model_dir: Path):
    arch = cleaner_cfg["architecture"]
    if arch == "encoder_decoder_cleaner":
        return EncoderDecoderCleaner(cleaner_cfg, train_cfg, model_dir)
    if arch == "decoder_only_cleaner":
        return DecoderOnlyCleaner(cleaner_cfg, train_cfg, model_dir)
    raise ValueError(f"Unknown cleaner architecture: {arch}")
