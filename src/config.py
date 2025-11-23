# src/config.py

from pathlib import Path
import torch

# Detecta GPU ou CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminho raiz do projeto (MUM-XAI/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Pasta de modelos
MODELS_DIR = ROOT_DIR / "models"

# Caminho completo para os modelos .pth
RESNET_CKPT_PATH = MODELS_DIR / "resnet_fold5.pth"
SWIN_CKPT_PATH   = MODELS_DIR / "swin_fold2.pth"

# Classes
CLASS_NAMES = ["uninfected", "infected"]
