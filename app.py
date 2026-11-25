"""Gradio arayüzü: kullanıcı fotoğraf yükler, model sınıf olasılıklarını gösterir.

Çalıştırma:  python app.py   ->  http://127.0.0.1:7860
"""
import time

import gradio as gr

from common import CLASSES, CLASSES_TR, load_trained, predict

model = load_trained()


def classify(img):
    if img is None:
        return {}, ""
    t0 = time.perf_counter()
    probs = predict(model, img)
    ms = (time.perf_counter() - t0) * 1000
    labels = {f"{CLASSES_TR[i]} ({CLASSES[i]})": float(p) for i, p in enumerate(probs)}
    return labels, f"Tahmin süresi: {ms:.1f} ms"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Fotoğraf yükle / çek"),
    outputs=[gr.Label(num_top_classes=5, label="Tahmin (ilk 5 sınıf)"),
             gr.Textbox(label="Süre")],
    title="CIFAR-10 Görüntü Sınıflandırma — ResNet-18",
    description=("Sınıflar: uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon. "
                 "Nesne fotoğrafın ortasında ve büyük olursa sonuç daha iyidir."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
