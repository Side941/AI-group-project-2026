"""
mentalroberta_baseline.py
=========================
External baseline for the dissertation comparison: MentalRoBERTa
(`mental/mental-roberta-base`) fine-tuned as a 3-class classifier and evaluated
on the SAME 450-post held-out set used for Phase 9 of the RAG pipeline.

IMPORTANT — why fine-tuning is required
---------------------------------------
`mental/mental-roberta-base` is a masked language model (pipeline tag:
fill-mask). It is RoBERTa-base further pretrained on mental-health Reddit
posts. It has NO classification head and cannot emit suicidal/depression/normal
labels out of the box: loading it with `AutoModelForSequenceClassification`
attaches a RANDOMLY INITIALISED head, which scores at chance (~0.33) until it
is trained. This script therefore fine-tunes it on labelled data drawn from the
same corpus as the project's own splits.

Leakage control
---------------
Training and validation rows are drawn from the source corpus with every
evaluation post removed, matched on normalised text (not row id), so no post
seen in training reappears at evaluation. The 450 evaluation posts are used
only for the final prediction pass.

Comparability
-------------
The output CSV uses the same schema as the project's phase result files
(`true_label`, `predicted`, `correct`), so the comparison notebook can load
both without special-casing.

Usage
-----
    pip install "transformers>=4.44" "torch>=2.2" scikit-learn datasets
    huggingface-cli login          # model is gated; a read token is required

    python analysis/mentalroberta_baseline.py                    # full run
    python analysis/mentalroberta_baseline.py --smoke            # 200 rows, 1 epoch
    python analysis/mentalroberta_baseline.py --train-size 12000 --epochs 3

Fine-tuning RoBERTa-base on CPU is slow. On an Apple-silicon Mac the script
uses the MPS backend automatically. If that is still too slow, run this file in
Google Colab with a T4 GPU and copy the output CSV back into results/.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_NAME = "mental/mental-roberta-base"
LABELS = ["suicidal", "depression", "normal"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

EVAL_CSV = ROOT / "datasets" / "processed" / "multiclass_eval.csv"
RAW_DIR = ROOT / "datasets" / "raw"
OUT_CSV = ROOT / "results" / "mentalroberta_eval_predictions.csv"
MODEL_DIR = ROOT / "models" / "mentalroberta-finetuned"

# Matches the project's own dataset builder (see multiclass_eval.meta.json).
MIN_CHARS, MAX_CHARS = 40, 2000
EXCLUDE_LABELS = {"anxiety"}


def norm(text: str) -> str:
    """Normalised form used for leakage matching (whitespace/case insensitive)."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def load_source_corpus() -> pd.DataFrame:
    """Load the labelled corpus, preferring local raw CSVs, else the Hub."""
    frames = []
    for name in ("mental_health_train.csv", "mental_health_test.csv"):
        p = RAW_DIR / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        print(f"Loaded {len(frames)} local raw file(s) from {RAW_DIR}")
        return pd.concat(frames, ignore_index=True)

    print("Local raw CSVs not found — downloading from the Hub …")
    from datasets import load_dataset
    ds = load_dataset("ourafla/Mental-Health_Text-Classification_Dataset")
    return pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)


def prepare_training_data(eval_df: pd.DataFrame, train_size: int,
                          val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_source_corpus()

    # Normalise column names across possible source schemas.
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("statement") or cols.get("post")
    label_col = cols.get("label") or cols.get("status") or cols.get("category")
    if not text_col or not label_col:
        sys.exit(f"Could not identify text/label columns in {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    n0 = len(df)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[~df["label"].isin(EXCLUDE_LABELS)]
    df = df[df["label"].isin(LABELS)]
    df["text"] = df["text"].astype(str)
    lengths = df["text"].str.len()
    df = df[(lengths >= MIN_CHARS) & (lengths <= MAX_CHARS)]
    df["_hash"] = df["text"].map(text_hash)
    df = df.drop_duplicates("_hash")
    print(f"Corpus: {n0} raw -> {len(df)} after label/length filtering and dedupe")

    # Leakage control: drop anything matching an evaluation post.
    eval_hashes = set(eval_df["text"].map(text_hash))
    before = len(df)
    df = df[~df["_hash"].isin(eval_hashes)]
    print(f"Removed {before - len(df)} rows matching evaluation posts "
          f"({len(eval_hashes)} eval texts); {len(df)} remain")

    # Balanced stratified sample.
    per_train, per_val = train_size // 3, val_size // 3
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in LABELS:
        pool = df[df["label"] == label].sample(frac=1.0, random_state=seed)
        need = per_train + per_val
        if len(pool) < need:
            sys.exit(f"Only {len(pool)} '{label}' rows available, need {need}. "
                     f"Lower --train-size.")
        train_parts.append(pool.iloc[:per_train])
        val_parts.append(pool.iloc[per_train:need])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"Train: {len(train)} ({train.label.value_counts().to_dict()})")
    print(f"Val:   {len(val)} ({val.label.value_counts().to_dict()})")
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=9000)
    ap.add_argument("--val-size", type=int, default=1500)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                    default="auto",
                    help="training device; use cpu if the mps preflight fails")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the gradient-flow check (not recommended)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline end to end")
    args = ap.parse_args()

    if args.smoke:
        args.train_size, args.val_size, args.epochs = 300, 150, 1.0

    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)
    from transformers import __version__ as transformers_version

    print(f"transformers {transformers_version}")

    set_seed(args.seed)

    if not EVAL_CSV.exists():
        sys.exit(f"Evaluation set not found at {EVAL_CSV}")
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"Evaluation set: {len(eval_df)} posts "
          f"({eval_df.label.value_counts().to_dict()})")

    train_df, val_df = prepare_training_data(
        eval_df, args.train_size, args.val_size, args.seed)

    if args.device != "auto":
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    def preflight(model, tokenizer, frame: pd.DataFrame, device: str,
                  steps: int = 30) -> None:
        """
        Verify that gradients actually update the model ON THIS DEVICE before
        committing to a long run.

        Trains a copy on a handful of examples and requires both the loss to
        fall and the classifier weights to move. A silent no-op training loop
        (seen with some transformers/MPS combinations, where the reported
        grad_norm is negative and eval metrics are byte-identical across
        epochs) is caught here in under a minute instead of after hours.
        """
        import copy

        print(f"\n=== Preflight: gradient-flow check on {device} ===")
        probe = copy.deepcopy(model).to(device)
        probe.train()

        sample = frame.groupby("label", group_keys=False).head(6)
        enc = tokenizer(list(sample["text"]), truncation=True, padding=True,
                        max_length=128, return_tensors="pt").to(device)
        labels = torch.tensor([LABEL2ID[l] for l in sample["label"]]).to(device)

        before = probe.classifier.out_proj.weight.detach().clone()
        opt = torch.optim.AdamW(probe.parameters(), lr=5e-5)

        losses, grad_norms = [], []
        for _ in range(steps):
            opt.zero_grad()
            out = probe(**enc, labels=labels)
            out.loss.backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum()
                                for p in probe.parameters() if p.grad is not None))
            grad_norms.append(float(gn))
            opt.step()
            losses.append(float(out.loss))

        delta = float((probe.classifier.out_proj.weight.detach() - before).abs().max())
        print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  "
              f"(chance = {np.log(len(LABELS)):.4f})")
        print(f"  grad norm: first {grad_norms[0]:.6f}, last {grad_norms[-1]:.6f}")
        print(f"  max classifier weight change: {delta:.6e}")

        problems = []
        if not all(g > 0 for g in grad_norms):
            problems.append("gradient norm is zero or negative (gradients not flowing)")
        if delta < 1e-8:
            problems.append("classifier weights did not change (optimiser not stepping)")
        if losses[-1] > losses[0] * 0.85:
            problems.append(f"loss did not fall over {steps} steps on {len(sample)} "
                            f"examples (the model cannot even overfit a tiny batch)")

        if problems:
            print("\n  PREFLIGHT FAILED:")
            for p_ in problems:
                print(f"    - {p_}")
            print(f"\n  Training on '{device}' with transformers "
                  f"{transformers_version} would produce an untrained model.")
            print("  Options, in order of preference:")
            print("    1. Run this script in Google Colab on a GPU runtime.")
            print("    2. Re-run locally with --device cpu (slower but correct).")
            print("    3. Pin a known-good stack: "
                  "pip install 'transformers>=4.44,<5' 'torch>=2.2'")
            sys.exit(1)

        print("  Preflight passed: gradients flow and the loss decreases.\n")

    class DS(torch.utils.data.Dataset):
        def __init__(self, frame: pd.DataFrame, with_labels: bool = True):
            self.enc = tokenizer(list(frame["text"]), truncation=True,
                                 padding="max_length", max_length=args.max_length)
            self.labels = ([LABEL2ID[l] for l in frame["label"]]
                           if with_labels else None)

        def __len__(self) -> int:
            return len(self.enc["input_ids"])

        def __getitem__(self, i: int) -> dict:
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[i])
            return item

    def metrics(pred) -> dict:
        y_pred = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(pred.label_ids, y_pred),
                "macro_f1": f1_score(pred.label_ids, y_pred, average="macro")}

    # TrainingArguments has changed across transformers major versions: some
    # keywords were renamed (evaluation_strategy -> eval_strategy) and others
    # removed. Filter the requested settings against the installed signature so
    # this script runs on whichever version is present, and report what was
    # dropped rather than failing.
    import inspect

    supported = set(inspect.signature(TrainingArguments.__init__).parameters)

    wanted = {
        "output_dir": str(MODEL_DIR / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "learning_rate": args.lr,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "logging_steps": 50,
        "seed": args.seed,
        "report_to": [],
    }

    # Older versions spell the evaluation schedule differently.
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")

    accepted = {k: v for k, v in wanted.items() if k in supported}
    dropped = sorted(set(wanted) - set(accepted))
    if dropped:
        print(f"[info] transformers {transformers_version}: ignoring "
              f"unsupported TrainingArguments {dropped}")

    # load_best_model_at_end requires save and eval schedules to match; if either
    # was dropped, disable it rather than let Trainer raise.
    if accepted.get("load_best_model_at_end") and not (
        {"eval_strategy", "evaluation_strategy"} & set(accepted)
        and "save_strategy" in accepted
    ):
        accepted.pop("load_best_model_at_end", None)
        accepted.pop("metric_for_best_model", None)
        print("[info] disabled load_best_model_at_end (eval/save strategy "
              "unavailable on this version)")

    targs = TrainingArguments(**accepted)

    if not args.skip_preflight:
        preflight(model, tokenizer, train_df, device)

    trainer = Trainer(model=model, args=targs, train_dataset=DS(train_df),
                      eval_dataset=DS(val_df), compute_metrics=metrics)

    print("\n=== Fine-tuning ===")
    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation:", val_metrics)

    # A fine-tuned in-domain encoder should comfortably beat chance. If it does
    # not, the run failed regardless of what the loss log printed, and the
    # predictions must not be reported as a baseline.
    val_acc = val_metrics.get("eval_accuracy", 0.0)
    if val_acc < 0.50:
        print(f"\n  WARNING: validation accuracy {val_acc:.4f} is at or near "
              f"chance ({1 / len(LABELS):.3f}).")
        print("  The model did not learn. Do NOT report these predictions as a")
        print("  baseline. Re-run on a GPU runtime or with --device cpu.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")

    print("\n=== Predicting on the held-out evaluation set ===")
    logits = trainer.predict(DS(eval_df, with_labels=False)).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = probs.argmax(-1)

    out = pd.DataFrame({
        "row_id": eval_df["row_id"],
        "post_index": range(len(eval_df)),
        "config": "MentalRoBERTa (fine-tuned)",
        "true_label": eval_df["label"],
        "predicted": [ID2LABEL[i] for i in pred_ids],
        "confidence": probs.max(-1).round(4),
    })
    for label in LABELS:
        out[f"p_{label}"] = probs[:, LABEL2ID[label]].round(4)
    out["correct"] = (out["predicted"] == out["true_label"]).astype(int)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nAccuracy: {out.correct.mean():.4f}")
    print(f"Macro-F1: {f1_score(out.true_label, out.predicted, average='macro'):.4f}")
    print()
    print(classification_report(out.true_label, out.predicted, digits=3))
    print(f"Predictions saved to {OUT_CSV}")
    print("\nNext: notebooks/05_external_comparison.ipynb for the paired comparison.")


if __name__ == "__main__":
    main()
