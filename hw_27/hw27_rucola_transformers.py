# -*- coding: utf-8 -*-
"""
ДЗ 27. Почувствуй мощь трансформеров в бою (RuCoLA)

Блоки:
  0) Импорты и настройки
  1) Загрузка данных и train/val split
  2) Fine-tune RuBERT
  3) RuGPT3 zero-/few-shot (затравки × число примеров)
  4) Fine-tune RuT5
  5) Сравнение метрик

Запуск:
  python hw27_rucola_transformers.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =============================================================================
# Блок 0. Пути, модели, устройство
# =============================================================================
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
MAX_LEN = 128
BERT_MODEL = "DeepPavlov/rubert-base-cased"
GPT_MODEL = "ai-forever/rugpt3small_based_on_gpt2"  # RuGPT3 small/base, влезает в 4GB
T5_MODEL = "ai-forever/ruT5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_all_seeds(seed: int = SEED) -> None:
    """Фиксируем seed во всех генераторах — чтобы прогон был воспроизводим."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Блок 1. Данные RuCoLA
# =============================================================================
def load_splits(val_size: float = 0.1):
    """
    train = часть in_domain_train.csv
    val   = оставшаяся часть in_domain_train.csv (стратификация по метке)
    test  = in_domain_dev.csv  (как просит условие ДЗ)
    """
    train_full = pd.read_csv(DATA / "in_domain_train.csv")
    test = pd.read_csv(DATA / "in_domain_dev.csv")
    train, val = train_test_split(
        train_full,
        test_size=val_size,
        random_state=SEED,
        stratify=train_full["acceptable"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(
        f"train={len(train)}, val={len(val)}, test={len(test)} | "
        f"доля acceptable в train={train.acceptable.mean():.3f}"
    )
    return train, val, test


def calc_metrics(y_true, y_pred) -> dict:
    """Accuracy + MCC — стандартные метрики RuCoLA."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


# =============================================================================
# Блок 2. Fine-tune RuBERT (классификация на 2 класса)
# =============================================================================
def run_rubert(train_df, val_df, test_df) -> dict:
    """Дообучает RuBERT и считает метрики на test."""
    print("\n=== Блок 2. Fine-tune RuBERT ===")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)

    def tokenize_batch(batch):
        # Текст -> input_ids / attention_mask; метка -> labels
        enc = tokenizer(
            batch["sentence"], truncation=True, max_length=MAX_LEN, padding=False
        )
        enc["labels"] = batch["acceptable"]
        return enc

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[["sentence", "acceptable"]], preserve_index=False
            ),
            "val": Dataset.from_pandas(
                val_df[["sentence", "acceptable"]], preserve_index=False
            ),
            "test": Dataset.from_pandas(
                test_df[["sentence", "acceptable"]], preserve_index=False
            ),
        }
    ).map(tokenize_batch, batched=True, remove_columns=["sentence", "acceptable"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return calc_metrics(labels, preds)

    args = TrainingArguments(
        output_dir=str(OUT / "rubert_ckpt"),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",  # выбираем чекпоинт по MCC на val
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(ds["test"])
    result = {
        "model": BERT_MODEL,
        "val": trainer.evaluate(ds["val"]),
        "test_accuracy": float(test_metrics["eval_accuracy"]),
        "test_mcc": float(test_metrics["eval_mcc"]),
    }
    print("RuBERT test:", result["test_accuracy"], result["test_mcc"])
    trainer.save_model(str(OUT / "rubert_best"))
    tokenizer.save_pretrained(str(OUT / "rubert_best"))
    return result


# =============================================================================
# Блок 3. RuGPT3 zero-/few-shot
# =============================================================================
# Три варианта затравок (пункт «а» задания)
PROMPTS = {
    "yes_no": {
        "instruction": (
            "Определи, является ли предложение грамматически правильным "
            "и приемлемым на русском языке. Ответь только да или нет.\n"
        ),
        "example_fn": lambda s, y: (
            f"Предложение: {s}\nОтвет: {'да' if y == 1 else 'нет'}\n"
        ),
        "query_fn": lambda s: f"Предложение: {s}\nОтвет:",
        "pos": "да",
        "neg": "нет",
    },
    "acceptable": {
        "instruction": (
            "Задача: классификация лингвистической приемлемости. "
            "Метка acceptable или unacceptable.\n"
        ),
        "example_fn": lambda s, y: (
            f"Текст: {s}\nМетка: {'acceptable' if y == 1 else 'unacceptable'}\n"
        ),
        "query_fn": lambda s: f"Текст: {s}\nМетка:",
        "pos": "acceptable",
        "neg": "unacceptable",
    },
    "correct_incorrect": {
        "instruction": (
            "Проверь корректность русского предложения. "
            "Ответ: корректно или некорректно.\n"
        ),
        "example_fn": lambda s, y: (
            f"Пример: {s}\nРезультат: {'корректно' if y == 1 else 'некорректно'}\n"
        ),
        "query_fn": lambda s: f"Пример: {s}\nРезультат:",
        "pos": "корректно",
        "neg": "некорректно",
    },
}


def build_prompt(prompt_cfg, shot_examples, sentence: str) -> str:
    """Собирает полный текст: инструкция + примеры + запрос."""
    parts = [prompt_cfg["instruction"]]
    for s, y in shot_examples:
        parts.append(prompt_cfg["example_fn"](s, y))
    parts.append(prompt_cfg["query_fn"](sentence))
    return "".join(parts)


@torch.no_grad()
def gpt_score_continuation(model, tokenizer, prompt: str, continuation: str) -> float:
    """
    Суммарный log-probability продолжения при условии prompt.
    Чем выше значение — тем модель «охотнее» генерирует эту метку.
    """
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    cont_ids = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # loss только на токенах ответа
    outputs = model(input_ids=input_ids, labels=labels)
    n_tok = cont_ids.shape[1]
    return -outputs.loss.item() * n_tok


def select_shots(train_df, k: int, seed: int = SEED):
    """k сбалансированных few-shot примеров (примерно поровну классов)."""
    if k == 0:
        return []
    n_pos = max(1, k // 2)
    n_neg = k - n_pos
    pos = train_df[train_df.acceptable == 1].sample(n=n_pos, random_state=seed)
    neg = train_df[train_df.acceptable == 0].sample(n=n_neg, random_state=seed)
    shots = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    return list(zip(shots.sentence.tolist(), shots.acceptable.tolist()))


def run_rugpt_fewshot(train_df, test_df, max_test: int | None = None) -> list:
    """
    Перебор затравок × числа shots (0,1,2,4).
    Модель не обучаем — только inference.
    """
    print("\n=== Блок 3. RuGPT3 few-/zero-shot ===")
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(GPT_MODEL)
    model.to(DEVICE)
    model.eval()

    eval_df = test_df
    if max_test is not None and len(test_df) > max_test:
        eval_df = test_df.sample(n=max_test, random_state=SEED).reset_index(drop=True)
        print(f"Оценка GPT на подвыборке n={len(eval_df)}")

    results = []
    for prompt_name, cfg in PROMPTS.items():
        for k in [0, 1, 2, 4]:  # пункт «б» задания
            shots = select_shots(train_df, k)
            preds = []
            for sentence in eval_df.sentence.tolist():
                prompt = build_prompt(cfg, shots, sentence)
                lp_pos = gpt_score_continuation(
                    model, tokenizer, prompt, " " + cfg["pos"]
                )
                lp_neg = gpt_score_continuation(
                    model, tokenizer, prompt, " " + cfg["neg"]
                )
                preds.append(1 if lp_pos >= lp_neg else 0)
            m = calc_metrics(eval_df.acceptable.tolist(), preds)
            row = {
                "method": "rugpt3_fewshot",
                "model": GPT_MODEL,
                "prompt": prompt_name,
                "n_shots": k,
                "n_eval": len(eval_df),
                **m,
            }
            results.append(row)
            print(
                f"prompt={prompt_name:20s} shots={k} "
                f"acc={m['accuracy']:.4f} mcc={m['mcc']:.4f}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# =============================================================================
# Блок 4. Fine-tune RuT5 (seq2seq: yes / no)
# =============================================================================
POS_LABEL = "yes"
NEG_LABEL = "no"


def run_rut5(train_df, val_df, test_df) -> dict:
    """
    Дообучает RuT5 генерировать yes/no.
    Схема как в baselines/finetune_t5.py репозитория RuCoLA.
    """
    print("\n=== Блок 4. Fine-tune RuT5 ===")
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)

    def preprocess(examples):
        # Вход с префиксом задачи; выход — короткая метка
        inputs = ["приемлемо ли предложение: " + s for s in examples["sentence"]]
        model_inputs = tokenizer(
            inputs, truncation=True, max_length=MAX_LEN, padding=False
        )
        targets = [
            POS_LABEL if y == 1 else NEG_LABEL for y in examples["acceptable"]
        ]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, max_length=8, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[["sentence", "acceptable"]], preserve_index=False
            ),
            "val": Dataset.from_pandas(
                val_df[["sentence", "acceptable"]], preserve_index=False
            ),
            "test": Dataset.from_pandas(
                test_df[["sentence", "acceptable"]], preserve_index=False
            ),
        }
    ).map(preprocess, batched=True, remove_columns=["sentence", "acceptable"])

    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)

    def compute_metrics(p):
        preds = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
        labels = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        int_preds = [1 if x.strip().lower() == POS_LABEL else 0 for x in preds]
        int_labels = [1 if x.strip().lower() == POS_LABEL else 0 for x in label_str]
        return calc_metrics(int_labels, int_preds)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "rut5_ckpt"),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.0,
        predict_with_generate=True,
        generation_max_length=8,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_out = trainer.predict(ds["test"], max_length=8)
    result = {
        "model": T5_MODEL,
        "test_accuracy": float(test_out.metrics["test_accuracy"]),
        "test_mcc": float(test_out.metrics["test_mcc"]),
        "val": trainer.evaluate(ds["val"]),
    }
    print("RuT5 test:", result["test_accuracy"], result["test_mcc"])
    trainer.save_model(str(OUT / "rut5_best"))
    tokenizer.save_pretrained(str(OUT / "rut5_best"))
    return result


# =============================================================================
# Блок 5. Запуск всего пайплайна и сравнение
# =============================================================================
def main() -> None:
    set_all_seeds()
    print("Устройство:", DEVICE)
    if DEVICE == "cuda":
        mem = torch.cuda.get_device_properties(0).total_memory // 2**20
        print("GPU:", torch.cuda.get_device_name(0), f"({mem} МБ)")

    train_df, val_df, test_df = load_splits()
    results = {"rubert": None, "rugpt3": None, "rut5": None}

    # --- RuBERT ---
    results["rubert"] = run_rubert(train_df, val_df, test_df)
    (OUT / "rubert_metrics.json").write_text(
        json.dumps(results["rubert"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- RuGPT3 ---
    results["rugpt3"] = run_rugpt_fewshot(train_df, test_df, max_test=None)
    pd.DataFrame(results["rugpt3"]).to_csv(
        OUT / "rugpt3_fewshot_metrics.csv", index=False
    )

    # --- RuT5 ---
    results["rut5"] = run_rut5(train_df, val_df, test_df)
    (OUT / "rut5_metrics.json").write_text(
        json.dumps(results["rut5"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Сводная таблица ---
    gpt_df = pd.DataFrame(results["rugpt3"])
    best_gpt = gpt_df.sort_values(["mcc", "accuracy"], ascending=False).iloc[0]
    rows = [
        {
            "подход": "RuBERT fine-tune",
            "детали": BERT_MODEL,
            "accuracy": results["rubert"]["test_accuracy"],
            "mcc": results["rubert"]["test_mcc"],
        },
        {
            "подход": "RuGPT3 few/zero-shot (лучший)",
            "детали": f"{best_gpt.prompt}, shots={int(best_gpt.n_shots)}",
            "accuracy": float(best_gpt.accuracy),
            "mcc": float(best_gpt.mcc),
        },
        {
            "подход": "RuT5 fine-tune",
            "детали": T5_MODEL,
            "accuracy": results["rut5"]["test_accuracy"],
            "mcc": results["rut5"]["test_mcc"],
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "comparison_summary.csv", index=False)
    print("\n=== Сравнение ===")
    print(summary.to_string(index=False))

    (OUT / "all_results.json").write_text(
        json.dumps(
            {
                "rubert": results["rubert"],
                "rugpt3": results["rugpt3"],
                "rut5": results["rut5"],
                "summary": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
