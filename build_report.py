"""outputs/ altındaki metriklerden Word raporu (Rapor.docx) üretir.

Çalıştırma:  python build_report.py
"""
import csv
import json
import os
import platform

import torch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from common import CLASSES, CLASSES_TR

REPO_URL = "https://github.com/Huseyin-creator/cifar10-image-classification"

m = json.load(open("outputs/metrics.json"))
h = json.load(open("outputs/history.json"))
hp = m["hyperparameters"]
inf = m["inference_ms_per_image_batch1"]
dev = m["device"]
real_csv = "outputs/real_photos_results.csv"
real = list(csv.DictReader(open(real_csv))) if os.path.exists(real_csv) else []

doc = Document()
doc.core_properties.author = "Hüseyin Melih ÖZBEK"
doc.core_properties.last_modified_by = "Hüseyin Melih ÖZBEK"
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2)


def para(text, bold=False, italic=False, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if align:
        p.alignment = align
    return p


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def table(rows, header=None, widths=None):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    if header:
        cells = t.add_row().cells
        for c, v in zip(cells, header):
            c.text = ""
            c.paragraphs[0].add_run(str(v)).bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = str(v)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph()
    return t


def figure(path, caption, width=15):
    if os.path.exists(path):
        doc.add_picture(path, width=Cm(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)


pct = lambda x: f"%{x*100:.2f}"
chip = platform.processor() or platform.machine()
env = (f"Apple M4 (16 GB), macOS, Python {platform.python_version()}, "
       f"PyTorch {torch.__version__}, hızlandırıcı: {dev.upper()} (Metal GPU)")

# ---------------- Başlık ----------------
doc.add_heading("COMPUTER VISION ÖDEVİ — PROJE RAPORU", 0)
para("CIFAR-10 Görüntü Sınıflandırma: ResNet-18 ile Transfer Öğrenme", bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER)
para("Hüseyin Melih ÖZBEK — 23120205052", align=WD_ALIGN_PARAGRAPH.CENTER)

# ---------------- 9. Şablon tablosu ----------------
doc.add_heading("Proje Özeti (Rapor Şablonu)", 1)
hyper = (f"Learning rate (max) = {hp['lr']}, batch size = {hp['batch_size']}, "
         f"epoch = {hp['epochs']}, weight decay = {hp['weight_decay']}, "
         f"LR scheduler = OneCycleLR (ısınma %15 + cosine azalma), seed = {hp['seed']}")
table([
    ["Project Title", "CIFAR-10 Görüntü Sınıflandırma — ResNet-18 Transfer Learning"],
    ["Problem / Task", "Görüntü sınıflandırma (10 sınıflı, tek etiketli)"],
    ["Dataset", f"CIFAR-10 (60.000 adet 32×32 RGB görüntü). Bölünme: "
                f"{m['split']['train']} eğitim / {m['split']['val']} doğrulama / "
                f"{m['split']['test']} test"],
    ["Dataset URL", "https://www.cs.toronto.edu/~kriz/cifar.html"],
    ["Model Architecture", "ResNet-18 (ImageNet ön-eğitimli, torchvision); son tam bağlantılı "
                           "katman 512→10 olarak değiştirildi, tüm katmanlar fine-tune edildi"],
    ["Input Size", f"3×32×32 (orijinal) → 3×{hp['input_size']}×{hp['input_size']} "
                   f"(bilinear büyütme)"],
    ["Hyperparameters", hyper],
    ["Optimizer", f"AdamW (β1=0.9, β2=0.999, weight decay={hp['weight_decay']})"],
    ["Activation / Output Function", "Gizli katmanlarda ReLU; çıkışta Softmax (10 sınıf)"],
    ["Loss Function", "Cross-Entropy Loss"],
    ["Evaluation Metrics", "Accuracy, Precision, Recall, F1-score (macro + sınıf bazında), "
                           "Confusion Matrix, eğitim süresi, tahmin süresi"],
    ["Training Environment", env],
    ["Best Performance", f"Test Accuracy {pct(m['test_accuracy'])}, Precision "
                         f"{pct(m['test_precision_macro'])}, Recall {pct(m['test_recall_macro'])}, "
                         f"F1 {pct(m['test_f1_macro'])} (en iyi doğrulama: "
                         f"{pct(m['best_val_acc'])}, epoch {m['best_epoch']})"],
    ["Inference Time", f"{inf[dev]:.2f} ms/görüntü ({dev.upper()}), {inf['cpu']:.2f} ms/görüntü "
                       f"(CPU), batch=1; toplu işlemde {m['test_set_throughput_img_per_sec']:.0f} "
                       f"görüntü/sn"],
    ["Deployment Technology", "Gradio (Python web arayüzü)"],
    ["Repository", REPO_URL],
], header=["Alan", "Açıklama / Öğrenci yanıtı"], widths=[5, 11.5])

# ---------------- 1-2 ----------------
doc.add_heading("1. Ödevin Amacı", 1)
para("Bu çalışmada açık kaynaklı CIFAR-10 veri seti kullanılarak bir görüntü sınıflandırma modeli "
     "geliştirilmiş, test kümesi üzerinde performansı ölçülmüş ve model Gradio ile kullanıcının "
     "fotoğraf yükleyip sonucu görebildiği çalışan bir uygulamaya dönüştürülmüştür.")
doc.add_heading("2. Görev (Task)", 1)
para("Seçilen problem görüntü sınıflandırmadır: verilen bir görüntünün 10 sınıftan hangisine ait "
     "olduğu tahmin edilir. Sınıflar: " + ", ".join(
         f"{tr} ({en})" for en, tr in zip(CLASSES, CLASSES_TR)) + ".")

# ---------------- 3 ----------------
doc.add_heading("3. Model (Architecture)", 1)
para("Model olarak ResNet-18 seçilmiştir. ResNet, katmanlar arasında atlama bağlantıları "
     "(residual/skip connection) kullanarak derin ağlarda gradyan kaybolması problemini azaltır. "
     "ResNet-18; 7×7 giriş evrişimi, 4 aşamada toplam 8 residual blok (her blokta iki 3×3 evrişim + "
     "Batch Normalization + ReLU), global ortalama havuzlama ve tam bağlantılı çıkış katmanından "
     "oluşur (~11,2 milyon parametre).")
para("Transfer öğrenme uygulanmıştır: ImageNet'te (1,2 milyon görüntü, 1000 sınıf) ön-eğitilmiş "
     "ağırlıklarla başlanmış, son katman 10 sınıflı yeni bir katmanla değiştirilmiş ve tüm ağ "
     "CIFAR-10 üzerinde yeniden eğitilmiştir (fine-tuning). Bu sayede model, kenar/doku/şekil gibi "
     "genel görsel özellikleri sıfırdan öğrenmek zorunda kalmaz; kısa sürede yüksek doğruluk elde "
     "edilir ve gerçek fotoğraflara genelleme daha iyi olur.")

# ---------------- 4 ----------------
doc.add_heading("4. Veri Seti (Dataset)", 1)
para("CIFAR-10, 10 sınıfa eşit dağılmış 60.000 adet 32×32 piksel renkli görüntüden oluşur "
     "(50.000 eğitim, 10.000 test; her sınıfta 6.000 görüntü). Veri seti torchvision ile otomatik "
     "indirilmiştir.")
table([["Eğitim (train)", m["split"]["train"], "Model ağırlıklarının öğrenilmesi"],
       ["Doğrulama (validation)", m["split"]["val"],
        "Her epoch sonunda performans takibi ve en iyi modelin seçimi"],
       ["Test", m["split"]["test"], "Yalnızca en sonda, nihai performans ölçümü"]],
      header=["Küme", "Görüntü sayısı", "Kullanım amacı"], widths=[4.5, 3, 9])
para(f"Orijinal 50.000'lik eğitim kümesi sabit tohum (seed={hp['seed']}) ile rastgele karıştırılıp "
     f"{m['split']['val']} görüntü doğrulama kümesine ayrılmıştır. Resmi 10.000'lik test kümesi "
     "eğitim sırasında hiç kullanılmamıştır.")
para("Ön-işleme ve veri artırma (data augmentation):", bold=True)
bullet("Eğitimde: 4 piksel dolgulu rastgele kırpma (RandomCrop) ve yatay çevirme "
       "(RandomHorizontalFlip) — modelin konum ve yön değişimlerine dayanıklı olması için.")
bullet("Normalizasyon: ImageNet ortalama/standart sapma değerleri (ön-eğitimli ağırlıklarla uyum).")
bullet(f"32×32 görüntüler GPU üzerinde {hp['input_size']}×{hp['input_size']}'ya büyütülmüştür. "
       "ResNet-18'in giriş katmanı görüntüyü 32 kat küçülttüğünden 32×32 girişte son özellik haritası "
       f"1×1'e iner; {hp['input_size']}×{hp['input_size']} giriş ile 3×3'lük harita elde edilir ve "
       "ImageNet ağırlıkları daha verimli kullanılır.")

# ---------------- 5 ----------------
doc.add_heading("5. Model Ayarları ve Eğitim", 1)
table([
    ["Learning rate", f"{hp['lr']} (maksimum)",
     "AdamW için fine-tuning'de kararlı ve hızlı yakınsayan bir değer"],
    ["LR scheduler", "OneCycleLR",
     "İlk %15'te LR kademeli artar (ısınma, ön-eğitimli ağırlıkları bozmamak için), sonra cosine "
     "ile sıfıra yaklaşır; az epoch'ta iyi sonuç verir"],
    ["Batch size", hp["batch_size"], "GPU belleği ile gradyan kararlılığı arasında dengeli değer"],
    ["Epoch", hp["epochs"], "Ön-eğitimli model hızlı yakınsadığı için yeterli; doğrulama eğrisi ile "
                            "kontrol edildi"],
    ["Optimizer", "AdamW", "Adam'ın uyarlanabilir öğrenme oranı + ağırlık çürümesinin doğru "
                           "(ayrık) uygulanması; düzenlileştirme ile aşırı öğrenmeyi azaltır"],
    ["Weight decay", hp["weight_decay"], "L2 benzeri düzenlileştirme"],
    ["Aktivasyon", "ReLU", "ResNet'in standart gizli katman aktivasyonu"],
    ["Çıkış fonksiyonu", "Softmax", "10 sınıf için toplamı 1 olan olasılık dağılımı üretir"],
    ["Loss", "Cross-Entropy", "Çok sınıflı tek etiketli sınıflandırma için standart kayıp; "
                              "Softmax ile birlikte doğru sınıfın log-olasılığını maksimize eder"],
], header=["Ayar", "Değer", "Seçim gerekçesi"], widths=[3.3, 3.2, 10])
para("En iyi model, doğrulama doğruluğunun en yüksek olduğu epoch'taki ağırlıklar kaydedilerek "
     "seçilmiş ve test değerlendirmesi bu modelle yapılmıştır.")
table([[i + 1, f"{h['train_loss'][i]:.4f}", pct(h['train_acc'][i]), f"{h['val_loss'][i]:.4f}",
        pct(h['val_acc'][i]), f"{h['epoch_time'][i]:.0f}"] for i in range(len(h["train_loss"]))],
      header=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Süre (sn)"])
figure("outputs/training_curves.png", "Şekil 1. Eğitim ve doğrulama loss/accuracy eğrileri")

# ---------------- 6 ----------------
doc.add_heading("6. Performans Değerlendirmesi", 1)
para("Test kümesi (10.000 görüntü) üzerindeki sonuçlar:", bold=True)
table([["Accuracy", pct(m["test_accuracy"])],
       ["Precision (macro)", pct(m["test_precision_macro"])],
       ["Recall (macro)", pct(m["test_recall_macro"])],
       ["F1-score (macro)", pct(m["test_f1_macro"])],
       ["Toplam eğitim süresi", f"{m['training_time_sec']/60:.1f} dakika "
                                f"(ortalama {m['avg_epoch_time_sec']:.0f} sn/epoch)"],
       ["Tahmin süresi (batch=1)", f"{inf[dev]:.2f} ms ({dev.upper()}) / {inf['cpu']:.2f} ms (CPU)"],
       ["Test kümesinin tamamı", f"{m['test_set_inference_time_sec']:.1f} sn "
                                 f"({m['test_set_throughput_img_per_sec']:.0f} görüntü/sn)"]],
      header=["Metrik", "Değer"], widths=[6, 10.5])
para("Sınıf bazında sonuçlar:", bold=True)
pc = m["per_class"]
table([[f"{tr} ({en})", f"{pc[en]['precision']:.4f}", f"{pc[en]['recall']:.4f}",
        f"{pc[en]['f1-score']:.4f}", int(pc[en]["support"])]
       for en, tr in zip(CLASSES, CLASSES_TR)],
      header=["Sınıf", "Precision", "Recall", "F1", "Örnek"])
figure("outputs/confusion_matrix.png", "Şekil 2. Test kümesi confusion matrix", width=13)

cm = m["confusion_matrix"]
pairs = sorted(((cm[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
f1s = sorted(((pc[c]["f1-score"], c) for c in CLASSES))
para("Yorum:", bold=True)
para(f"Model en başarılı olduğu sınıflar {CLASSES_TR[CLASSES.index(f1s[-1][1])]} "
     f"(F1={f1s[-1][0]:.3f}) ve {CLASSES_TR[CLASSES.index(f1s[-2][1])]} (F1={f1s[-2][0]:.3f}); "
     f"en zorlandığı sınıflar {CLASSES_TR[CLASSES.index(f1s[0][1])]} (F1={f1s[0][0]:.3f}) ve "
     f"{CLASSES_TR[CLASSES.index(f1s[1][1])]} (F1={f1s[1][0]:.3f}). En sık karışıklıklar: " +
     "; ".join(f"{CLASSES_TR[i]} → {CLASSES_TR[j]} ({n} kez)" for n, i, j in pairs) +
     ". Bu karışıklıklar beklenen türdendir: kedi–köpek gibi sınıflar 32×32 çözünürlükte benzer "
     "şekil ve dokuya sahiptir; otomobil–kamyon ve uçak–gemi gibi çiftlerde ise arka plan (yol, "
     "gökyüzü/deniz) ve genel siluet benzerliği ayırt etmeyi zorlaştırır.")
gap = h["train_acc"][-1] - h["val_acc"][-1]
para(f"Son epoch'ta eğitim ve doğrulama doğrulukları arasındaki fark {gap*100:.1f} puandır; veri "
     "artırma ve weight decay sayesinde belirgin bir aşırı öğrenme (overfitting) gözlenmemiştir.")

# ---------------- 7 ----------------
doc.add_heading("7. Uygulama (Deployment)", 1)
para("Eğitilen model Gradio ile web uygulamasına dönüştürülmüştür (app.py). Kullanıcı bilgisayardan "
     "fotoğraf yükleyebilir, sürükle-bırak yapabilir veya kamera ile çekim yapabilir; uygulama en "
     "olası 5 sınıfı olasılıklarıyla birlikte ve tahmin süresini gösterir. Çalıştırma: "
     "python app.py → http://127.0.0.1:7860")
para("Gerçek fotoğraflar modele verilmeden önce eğitim verisiyle aynı dağılıma getirilir: EXIF yönü "
     "düzeltilir, ortadan kare kırpılır, 32×32'ye küçültülür, normalize edilir ve "
     f"{hp['input_size']}×{hp['input_size']}'ya büyütülür.")
figure("outputs/app_screenshot.png", "Şekil 3. Gradio uygulaması — telefonla çekilen köpek fotoğrafının sınıflandırılması")

# ---------------- 8 ----------------
doc.add_heading("8. Gerçek Görüntülerle Test ve Değerlendirme", 1)
if real:
    ok = sum(r["correct"] == "True" for r in real)
    para(f"Telefonla çekilen {len(real)} farklı fotoğraf modele verilmiştir. Doğru tahmin: "
         f"{ok}/{len(real)}.")
    figure("outputs/real_photos_results.png",
           "Şekil 4. Telefon fotoğrafları — gerçek sınıf ve model tahmini (yeşil: doğru, kırmızı: yanlış)",
           width=16.5)
    table([[i + 1, CLASSES_TR[CLASSES.index(r["true"])] if r["true"] else "?",
            CLASSES_TR[CLASSES.index(r["pred"])], f"%{float(r['conf'])*100:.1f}",
            "Doğru" if r["correct"] == "True" else "Yanlış", r["top3"]]
           for i, r in enumerate(real)],
          header=["#", "Gerçek", "Tahmin", "Güven", "Sonuç", "İlk 3 tahmin"])
    figure("outputs/real_photos_model_input.png",
           "Şekil 5. Fotoğrafların ön-işleme sonrası modele verilen 32×32 hâli", width=16.5)
else:
    para("[Telefon fotoğrafları eklendikten sonra doldurulacak]")
para("Başarılı ve başarısız durumların yorumu:", bold=True)
para("Başarılı durumlar (kedi, kuş, köpek, otomobil): Bu dört fotoğrafta nesne tek başına, "
     "kadrajın büyük kısmını kaplıyor ve arka plandan net ayrılıyor. Model hepsini %99,9 ve üzeri "
     "güvenle doğru sınıflandırdı. Otomobil fotoğrafı gece ve yukarıdan (CIFAR-10'da az görülen "
     "bir açıyla) çekilmiş olmasına rağmen doğru tahmin edildi; 32×32'ye küçültüldüğünde bile "
     "aracın silueti ve tekerlekleri belirgin kalmaktadır. Kuş fotoğrafındaki sultan papağanı ev "
     "ortamında bir tünekte olmasına (CIFAR-10 kuşları çoğunlukla doğal ortamdadır) karşın doğru "
     "sınıflandırıldı. Bu durum, ImageNet ön-eğitiminden gelen genel özelliklerin ve veri "
     "artırmanın modelin genelleme yeteneğine katkısını göstermektedir.")
para("Zorlanılan durum (karavan): Karavan fotoğrafı doğru sınıflandırılmış (kamyon) ancak güven "
     "yalnızca %60,5'tir; ikinci tahmin %19,2 ile kurbağa, üçüncüsü %14,4 ile otomobildir. "
     "Bunun başlıca nedenleri:")
bullet("CIFAR-10'da karavan sınıfı yoktur; kamyon sınıfı ağırlıkla tır ve kamyonlardan oluşur. "
       "Karavan, kamyon ile otomobil arasında kalan ve eğitim verisinde doğrudan temsil edilmeyen "
       "bir nesnedir.")
bullet("Karede birden fazla nesne vardır (birkaç karavan ve iki otomobil). Tek etiketli "
       "sınıflandırma modeli tüm görüntüye tek bir etiket verdiği için olasılık otomobil sınıfına "
       "da dağılmıştır.")
bullet("Fotoğraf gece, düşük ışıkta çekilmiştir; gürültü ve renk bozulması 32×32'ye "
       "küçültmeyle birleşince ayrıntılar kaybolmaktadır (Şekil 5).")
bullet("Kurbağa olasılığının yükselmesi, modelin kısmen renk ve dokuya dayandığını gösterir: "
       "yeşil bitki örtüsü arka planı ve parlak kırmızı-beyaz yüzeyler CIFAR-10 kurbağa "
       "görüntülerindeki renk desenlerine benzemektedir.")
para("Genel değerlendirme: Model, nesne tek, ortada ve kadrajı dolduracak büyüklükte olduğunda "
     "gerçek telefon fotoğraflarında da test kümesindeki başarısını (%94,68) korumaktadır. Birden "
     "fazla nesne içeren, düşük ışıkta çekilen veya veri setinde karşılığı olmayan nesnelerde ise "
     "güven belirgin şekilde düşmektedir. Model kapalı küme (closed-set) bir sınıflandırıcı olduğu "
     "için tanımadığı bir nesneyi de mutlaka 10 sınıftan birine atar; test kümesinde en sık "
     "görülen hata olan kedi–köpek karışıklığı da benzer görsel özelliklerden kaynaklanır. "
     "İyileştirme için nesne tespiti (ör. YOLO) ile önce nesnenin kırpılması, daha yüksek "
     "çözünürlüklü bir veri seti kullanılması ve düşük güvenli tahminlerin \"emin değilim\" olarak "
     "işaretlenmesi (güven eşiği) önerilebilir.")

doc.save("Rapor.docx")
print("Rapor.docx oluşturuldu")
