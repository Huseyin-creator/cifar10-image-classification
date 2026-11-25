"""Telefonla çekilen fotoğrafları modelle test eder (ödev madde 8).

Fotoğrafları real_photos/ klasörüne gerçek sınıf adıyla başlayacak şekilde koy:
    real_photos/cat_1.jpg, real_photos/automobile_sokak.jpg, real_photos/dog.heic ...
(sınıf adları: airplane automobile bird cat deer dog frog horse ship truck)

Çalıştırma:  python predict_real.py
Çıktılar:    outputs/real_photos_results.png, outputs/real_photos_results.csv
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from common import CLASSES, CLASSES_TR, load_trained, predict

DIR = "real_photos"
EXT = (".jpg", ".jpeg", ".png", ".heic", ".webp")

try:  # iPhone HEIC desteği (opsiyonel)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def true_label(fname):
    stem = os.path.splitext(fname)[0].lower()
    for c in CLASSES:
        if stem.startswith(c):
            return c
    return None


def main():
    files = sorted(f for f in os.listdir(DIR) if f.lower().endswith(EXT))
    if not files:
        sys.exit(f"{DIR}/ klasöründe fotoğraf yok.")
    model = load_trained()
    rows = []
    for f in files:
        img = Image.open(os.path.join(DIR, f))
        probs = predict(model, img)
        top = probs.topk(3)
        pred = CLASSES[top.indices[0]]
        gt = true_label(f)
        rows.append(dict(file=f, true=gt, pred=pred, conf=float(top.values[0]),
                         correct=(gt == pred) if gt else None,
                         top3=", ".join(f"{CLASSES_TR[i]} %{p*100:.1f}"
                                        for p, i in zip(top.values, top.indices)),
                         img=ImageOps.exif_transpose(img).convert("RGB")))

    n = len(rows)
    cols = min(n, 5)
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(3.6 * cols, 4.3 * r), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows):
        ax.imshow(row["img"])
        gt_tr = CLASSES_TR[CLASSES.index(row["true"])] if row["true"] else "?"
        color = "green" if row["correct"] else ("red" if row["correct"] is False else "black")
        ax.set_title(f"Gerçek: {gt_tr}\nTahmin: {CLASSES_TR[CLASSES.index(row['pred'])]} "
                     f"(%{row['conf']*100:.1f})", color=color, fontsize=11)
        ax.text(0.5, -0.06, row["top3"], transform=ax.transAxes, ha="center",
                va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/real_photos_results.png", dpi=150, bbox_inches="tight")

    # Modelin gördüğü girdi: ortadan kare kırpılmış ve 32x32'ye küçültülmüş hâl
    fig2, axes2 = plt.subplots(1, n, figsize=(2.4 * n, 2.8), squeeze=False)
    for ax, row in zip(axes2[0], rows):
        w, h = row["img"].size
        s = min(w, h)
        small = row["img"].crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s,
                                 (h - s) // 2 + s)).resize((32, 32), Image.BICUBIC)
        ax.imshow(small, interpolation="nearest")
        ax.set_title(row["file"], fontsize=8)
        ax.axis("off")
    fig2.suptitle("Modelin gördüğü girdi (32×32)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig("outputs/real_photos_model_input.png", dpi=150, bbox_inches="tight")

    with open("outputs/real_photos_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "true", "pred", "conf", "correct", "top3"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in w.fieldnames})

    ok = sum(1 for x in rows if x["correct"])
    for x in rows:
        print(f"{x['file']:30s} gerçek={x['true']!s:11s} tahmin={x['pred']:11s} "
              f"%{x['conf']*100:5.1f}  {'✓' if x['correct'] else '✗'}   [{x['top3']}]")
    print(f"\nDoğru: {ok}/{n}")


if __name__ == "__main__":
    main()
