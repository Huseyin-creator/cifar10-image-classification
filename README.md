# CIFAR-10 Görüntü Sınıflandırma — ResNet-18 (Transfer Learning)

Computer Vision ödevi: açık kaynaklı CIFAR-10 veri seti üzerinde ImageNet ön-eğitimli
ResNet-18 modeli fine-tune edilmiş, Gradio ile çalışan bir uygulamaya dönüştürülmüştür.

| | |
|---|---|
| Görev | Görüntü sınıflandırma (10 sınıf) |
| Veri seti | CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html |
| Bölünme | 45.000 eğitim / 5.000 doğrulama / 10.000 test |
| Model | ResNet-18 (ImageNet ağırlıkları), son katman 10 çıkışlı |
| Giriş | 32×32 RGB → 96×96'ya bilinear büyütme |
| Uygulama | Gradio |

## Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

```bash
python train.py --epochs 10          # eğitim + test metrikleri -> outputs/
python app.py                        # Gradio arayüzü -> http://127.0.0.1:7860
python predict_real.py               # real_photos/ içindeki telefon fotoğraflarını test eder
```

Telefon fotoğrafları `real_photos/` klasörüne gerçek sınıf adıyla başlayan isimle konur
(ör. `cat_1.jpg`, `automobile_park.jpg`).

## Dosyalar

- `common.py` — sınıflar, model tanımı, ön-işleme
- `train.py` — eğitim, doğrulama, test değerlendirmesi, süre ölçümleri, grafikler
- `app.py` — Gradio uygulaması
- `predict_real.py` — gerçek fotoğraflarla test ve görsel karşılaştırma
- `outputs/` — metrikler (`metrics.json`), eğitim geçmişi, grafikler, model ağırlıkları

---
Hüseyin Melih ÖZBEK — 23120205052
