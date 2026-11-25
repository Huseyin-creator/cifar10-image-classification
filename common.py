"""Ortak tanımlar: sınıflar, model, ön-işleme."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
CLASSES_TR = ["uçak", "otomobil", "kuş", "kedi", "geyik",
              "köpek", "kurbağa", "at", "gemi", "kamyon"]

CIFAR_SIZE = 32      # veri setinin orijinal çözünürlüğü
INPUT_SIZE = 96      # modele verilen giriş boyutu (32 -> 96 upsample)
MEAN = (0.485, 0.456, 0.406)   # ImageNet istatistikleri (ön-eğitimli ağırlıklarla uyumlu)
STD = (0.229, 0.224, 0.225)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(pretrained=True):
    """ImageNet ön-eğitimli ResNet-18; son katman 10 sınıfa değiştirildi."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def upsample(x):
    """32x32 batch'i GPU üzerinde modele giriş boyutuna büyütür."""
    return F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)


# Eğitim / test dönüşümleri 32x32 düzeyinde yapılır, büyütme GPU'da (upsample)
train_tf = transforms.Compose([
    transforms.RandomCrop(CIFAR_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def preprocess_photo(img: Image.Image) -> torch.Tensor:
    """Gerçek (telefon) fotoğrafını eğitim verisiyle aynı dağılıma getirir:
    EXIF yönü düzeltilir, ortadan kare kırpılır, 32x32'ye küçültülür, normalize edilir.
    Dönen tensör: (1, 3, 32, 32). Modelden önce upsample() uygulanır."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((CIFAR_SIZE, CIFAR_SIZE), Image.BICUBIC)
    return eval_tf(img).unsqueeze(0)


def load_trained(path="outputs/best_model.pt", device=None):
    device = device or get_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img: Image.Image, device=None):
    device = device or next(model.parameters()).device
    x = upsample(preprocess_photo(img).to(device))
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    return probs
