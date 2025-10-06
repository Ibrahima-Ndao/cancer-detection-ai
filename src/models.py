# src/models.py
# ------------------------------------------------------------
# Fichier "modèles" du projet.
# - Contient : ResNet18/50, VGG16, EfficientNet-B0, DenseNet121
# - Et un modèle perso avec attention : IbraCancerModel
# - Une fabrique build_model(name=...) pour construire le bon réseau
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# 1) BLOCS D'ATTENTION (CBAM-like)
# =========================
# Idée : apprendre à "peser" (attend) les canaux (features) et/ou
# les positions spatiales importantes pour mieux détecter les motifs tumoraux.

class ChannelAttention(nn.Module):
    """
    Attention par canal :
      - On résume chaque carte de feature par 2 stats : moyenne et max (Global Pool)
      - On passe par un MLP 1x1 (Conv 1x1) pour produire des poids par canal
      - On applique une sigmoid pour obtenir des coefficients entre 0 et 1
    """
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        # 2 agrégations globales : moyenne et max -> 2 tenseurs [B, c, 1, 1]
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        # "goulot" (réduction) -> expansion : Conv 1x1 pour simuler un petit MLP
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # On calcule séparément via avg et via max, puis on additionne
        avg_out = self.fc(self.avg(x))
        max_out = self.fc(self.max(x))
        out = avg_out + max_out
        return self.sigmoid(out)  # poids canal [B, c, 1, 1]


class SpatialAttention(nn.Module):
    """
    Attention spatiale :
      - On condense les canaux par moyenne et max -> 2 cartes [B, 1, H, W]
      - On concatène en 2 canaux -> Conv pour produire 1 carte de poids
      - Sigmoid pour [0,1]
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)        # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # [B,1,H,W]
        x_cat = torch.cat([avg_out, max_out], dim=1)        # [B,2,H,W]
        out = self.conv(x_cat)                              # [B,1,H,W]
        return self.sigmoid(out)                            # poids spatial


class AttentionBlock(nn.Module):
    """
    Bloc combinant ChannelAttention puis SpatialAttention.
    On multiplie l'entrée par les poids appris.
    """
    def __init__(self, c: int):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =========================
# 2) MODÈLE PERSONNALISÉ : IbraCancerModel
# =========================
# - Conçu pour des patches 32/64/96 px
# - 3 blocs conv + attention + pooling
# - Tête linéaire -> 1 logit
# - Appelle-le avec --model ibracancermodel
class IbraCancerModel(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # -> [B,32,H,W]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            AttentionBlock(32),                           # attention canal+spatial
            nn.MaxPool2d(2),                              # /2 -> [B,32,H/2,W/2]

            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [B,64,H/2,W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            nn.MaxPool2d(2),                              # /2 -> [B,64,H/4,W/4]

            # Bloc 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> [B,128,H/4,W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AttentionBlock(128),

            # Pooling global pour condenser en [B,128,1,1]
            nn.AdaptiveAvgPool2d(1),
        )

        # Tête de classification binaire -> 1 logit
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B,128]
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # num_classes=1 -> logit binaire
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # sortie [B, 1]


# =========================
# 3) BACKBONES TORCHVISION (pré-entraînés optionnels)
# =========================
# Astuce : si tu mets pretrained=True, utilise de préférence img_size=224,
# car ces poids sont appris sur ImageNet (224x224).
def _replace_fc_linear(m: nn.Module, in_features: int, num_classes: int, dropout: float):
    """Remplace proprement la tête fully-connected pour sortie binaire."""
    m.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

def _replace_vgg_classifier(m: nn.Module, in_features: int, num_classes: int):
    m.classifier[-1] = nn.Linear(in_features, num_classes)

def _replace_efficientnet_classifier(m: nn.Module, in_features: int, num_classes: int):
    m.classifier[-1] = nn.Linear(in_features, num_classes)

def _replace_densenet_classifier(m: nn.Module, in_features: int, num_classes: int):
    m.classifier = nn.Linear(in_features, num_classes)


# =========================
# 4) FABRIQUE DE MODÈLES
# =========================
def build_model(
    name: str = "resnet18",
    num_classes: int = 1,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Construit et retourne le modèle demandé.

    Paramètres
    ----------
    name : str
        "resnet18" | "resnet50" | "vgg16" | "efficientnet_b0" | "densenet121" | "ibracancermodel"
    num_classes : int
        Pour binaire, laisser 1 (logit). Si >1, passer à BCEWithLogits multi-label ou CrossEntropy selon le cas.
    pretrained : bool
        Si True (quand supporté), charge des poids ImageNet (penser à img_size=224).
    dropout : float
        Dropout dans la tête finale (utile pour la régularisation).
    """
    n = name.lower()

    if n == "ibracancermodel":
        # Ton modèle perso avec attention (léger & adapté aux petits patches)
        return IbraCancerModel(num_classes=num_classes, dropout=dropout)

    if n == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        _replace_fc_linear(m, m.fc.in_features, num_classes, dropout)
        return m

    if n == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        _replace_fc_linear(m, m.fc.in_features, num_classes, dropout)
        return m

    if n == "vgg16":
        m = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if pretrained else None
        )
        _replace_vgg_classifier(m, m.classifier[-1].in_features, num_classes)
        return m

    if n == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        _replace_efficientnet_classifier(m, m.classifier[-1].in_features, num_classes)
        return m

    if n == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        _replace_densenet_classifier(m, m.classifier.in_features, num_classes)
        return m

    raise ValueError(
        f"Unknown model: {name}. "
        "Use one of: resnet18, resnet50, vgg16, efficientnet_b0, densenet121, ibracancermodel"
    )
