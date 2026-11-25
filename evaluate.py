"""Evaluate the final ensemble (and each member) on the held-out splits.

    python evaluate.py                       # all splits, writes results/ (tables + figures)
    python evaluate.py --splits test_external

Writes:
  results/final_metrics.csv          accuracy (+ 95% Wilson interval), macro precision / recall / F1, bin accuracy with the rules in bins.json, per-class recall
  results/final_per_class_<split>.csv  precision, recall, F1, support per class
  results/confusion_<split>.png      confusion matrix (counts)
  results/roc_<split>.png            one-vs-rest ROC curves with AUC
  results/sample_outputs.png         photos with true class, predicted class and bin
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_curve)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from binsort import BinRules, WasteClassifier  # noqa: E402

DATA = ROOT / "data" / "trashnet_custom_v2"
OUT = ROOT / "results"
SPLITS = ["val", "val_bgshift", "test", "test_external", "test_bgshift"]
EXT = {".jpg", ".jpeg", ".png"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def list_split(split, classes):
    files, labels = [], []
    for i, c in enumerate(classes):
        for p in sorted((DATA / split / c).iterdir()):
            if p.suffix.lower() in EXT:
                files.append(p)
                labels.append(i)
    return files, np.array(labels)


def predict_all(clf, files, bs=32):
    out = []
    for i in range(0, len(files), bs):
        out.append(clf.predict_batch(files[i:i + bs]))
    return np.concatenate(out)


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(y, probs, classes, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y == i, probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sample_grid(files, y, probs, classes, rules, path, n=12, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        pred = int(probs[i].argmax())
        ax.imshow(Image.open(files[i]).convert("RGB"))
        ok = pred == y[i]
        ax.set_title(f"true: {classes[y[i]]}\npred: {classes[pred]} ({probs[i][pred]:.2f}) -> {rules.lookup(classes[pred])['bin']}",
                     fontsize=7.5, color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test", "test_external", "test_bgshift"], choices=SPLITS)
    ap.add_argument("--members", action="store_true", help="also report each ensemble member on its own")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    clf = WasteClassifier()
    classes = clf.classes
    rules = BinRules()
    rows = []
    variants = [("ENS(E1+E2)", None)]
    if args.members:
        variants += [(n, [n]) for n, _ in clf.members]
    for name, members in variants:
        c = clf if members is None else WasteClassifier(members=members)
        for split in args.splits:
            files, y = list_split(split, classes)
            probs = predict_all(c, files)
            pred = probs.argmax(1)
            acc = accuracy_score(y, pred)
            p, r, f, sup = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
            lo, hi = wilson(int((pred == y).sum()), len(y))
            bin_of = np.array([rules.lookup(c)["bin"] for c in classes])   # class -> bin name with the rules in bins.json
            bin_acc = float((bin_of[y] == bin_of[pred]).mean())
            row = dict(model=name, split=split, n=len(y), accuracy=acc, acc_ci_low=lo, acc_ci_high=hi,
                       macro_precision=p.mean(), macro_recall=r.mean(), macro_f1=f1_score(y, pred, average="macro"),
                       bin_accuracy=bin_acc)
            row.update({f"recall_{k}": v for k, v in zip(classes, r)})
            rows.append(row)
            print(f"{name:20s} {split:14s} n={len(y):4d} acc={acc:.3f} [{lo:.3f},{hi:.3f}] macroF1={row['macro_f1']:.3f} "
                  f"worst-class recall={r.min():.2f} ({classes[int(r.argmin())]}) bin accuracy={bin_acc:.3f}")
            if members is None:
                pd.DataFrame(dict(precision=p, recall=r, f1=f, support=sup), index=classes).round(4).to_csv(
                    OUT / f"final_per_class_{split}.csv")
                plot_confusion(confusion_matrix(y, pred, labels=range(len(classes))), classes,
                               f"{name} on {split} (n={len(y)})", OUT / f"confusion_{split}.png")
                plot_roc(y, probs, classes, f"{name} on {split}: one-vs-rest ROC", OUT / f"roc_{split}.png")
                if split == "test_external":
                    sample_grid(files, y, probs, classes, rules, OUT / "sample_outputs.png")
    pd.DataFrame(rows).round(4).to_csv(OUT / "final_metrics.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
