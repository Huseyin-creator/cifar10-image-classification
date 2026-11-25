"""CIFAR-10 üzerinde ResNet-18 fine-tune + test değerlendirmesi.

Kullanım:  python train.py --epochs 10
Çıktılar:  outputs/ (best_model.pt, history.json, metrics.json, grafikler)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from common import (CLASSES, CLASSES_TR, INPUT_SIZE, build_model, eval_tf, get_device,
                    train_tf, upsample)

OUT = "outputs"


def get_loaders(batch_size, val_size, seed, workers):
    train_full = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR10("data", train=True, download=False, transform=eval_tf)
    test = datasets.CIFAR10("data", train=False, download=True, transform=eval_tf)

    # 50.000 eğitim görüntüsü -> 45.000 eğitim + 5.000 doğrulama (sabit seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_eval, val_idx)   # doğrulamada augmentation yok

    kw = dict(num_workers=workers, persistent_workers=workers > 0)
    return (DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, 256, shuffle=False, **kw),
            DataLoader(test, 256, shuffle=False, **kw),
            len(train_ds), len(val_ds), len(test))


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = upsample(x.to(device)), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(upsample(x.to(device))).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def measure_inference(model, device, runs=200):
    """Tek görüntü (batch=1) için ortalama tahmin süresi (ms)."""
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(20):            # ısınma
        model(upsample(x))
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(upsample(x))
    sync()
    return (time.perf_counter() - t0) / runs * 1000


def plot_history(h):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], "o-", label="Eğitim")
    ax[0].plot(ep, h["val_loss"], "o-", label="Doğrulama")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    ax[1].plot(ep, h["train_acc"], "o-", label="Eğitim")
    ax[1].plot(ep, h["val_acc"], "o-", label="Doğrulama")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_curves.png", dpi=150)
    plt.close(fig)


def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES_TR, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES_TR)
    ax.set(xlabel="Tahmin", ylabel="Gerçek", title="Confusion Matrix (Test Kümesi)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Cihaz: {device}")

    train_dl, val_dl, test_dl, n_tr, n_val, n_te = get_loaders(
        args.batch_size, args.val_size, args.seed, args.workers)
    print(f"Eğitim/Doğrulama/Test: {n_tr}/{n_val}/{n_te}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)

    h = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"]}
    best_acc, best_ep = 0.0, 0
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer, scheduler)
        vl, va = run_epoch(model, val_dl, criterion, device)
        dt = time.perf_counter() - t0
        for k, v in zip(h, [tl, ta, vl, va, dt]):
            h[k].append(v)
        flag = ""
        if va > best_acc:
            best_acc, best_ep = va, ep
            torch.save(model.state_dict(), f"{OUT}/best_model.pt")
            flag = "  *kaydedildi"
        print(f"Epoch {ep:2d}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {dt:.0f}s{flag}", flush=True)
    train_time = time.perf_counter() - t_start
    json.dump(h, open(f"{OUT}/history.json", "w"), indent=2)
    plot_history(h)

    # ---- Test değerlendirmesi (en iyi doğrulama modeliyle) ----
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", map_location=device))
    t0 = time.perf_counter()
    y, yp = collect_preds(model, test_dl, device)
    test_time = time.perf_counter() - t0
    prec, rec, f1, _ = precision_recall_fscore_support(y, yp, average="macro")
    cm = confusion_matrix(y, yp)
    plot_cm(cm)
    report = classification_report(y, yp, target_names=CLASSES, digits=4, output_dict=True)
    print(classification_report(y, yp, target_names=CLASSES, digits=4))

    inf_dev = measure_inference(model, device)
    inf_cpu = measure_inference(model.to("cpu"), torch.device("cpu"), runs=50)

    metrics = {
        "device": str(device),
        "hyperparameters": vars(args) | {"input_size": INPUT_SIZE, "optimizer": "AdamW",
                                         "scheduler": "OneCycleLR", "loss": "CrossEntropyLoss"},
        "split": {"train": n_tr, "val": n_val, "test": n_te},
        "best_val_acc": best_acc, "best_epoch": best_ep,
        "test_accuracy": accuracy_score(y, yp),
        "test_precision_macro": prec, "test_recall_macro": rec, "test_f1_macro": f1,
        "per_class": {c: report[c] for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "training_time_sec": train_time,
        "avg_epoch_time_sec": float(np.mean(h["epoch_time"])),
        "test_set_inference_time_sec": test_time,
        "test_set_throughput_img_per_sec": n_te / test_time,
        "inference_ms_per_image_batch1": {str(device): inf_dev, "cpu": inf_cpu},
    }
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
    print(f"\nTest Acc {metrics['test_accuracy']:.4f} | P {prec:.4f} R {rec:.4f} F1 {f1:.4f}")
    print(f"Eğitim süresi {train_time/60:.1f} dk | tahmin {inf_dev:.2f} ms ({device}), "
          f"{inf_cpu:.2f} ms (cpu)")


if __name__ == "__main__":
    main()
