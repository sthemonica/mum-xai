import io
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.swin_transformer import swin_t
from captum.attr import LayerGradCam
from .config import DEVICE

def load_resnet_model(ckpt_path: str):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    model.to(DEVICE)
    model.eval()

    gradcam_layer = model.layer4[-1]
    gradcam_obj = LayerGradCam(model, gradcam_layer)

    return model, gradcam_obj


def load_swin_model(ckpt_path: str):
    swin = swin_t(weights=None)
    num_features = swin.head.in_features
    swin.head = nn.Linear(num_features, 2)

    state = torch.load(ckpt_path, map_location=DEVICE)
    swin.load_state_dict(state)

    swin.to(DEVICE)
    swin.eval()

    swin_layer = swin.features[0][0]
    gradcam_obj = LayerGradCam(swin, swin_layer)

    return swin, gradcam_obj


def load_custom_model_from_pth(pth_file, base_model: str):
    state = torch.load(io.BytesIO(pth_file.read()), map_location=DEVICE)

    if base_model == "ResNet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(state)

        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        model.to(DEVICE)
        model.eval()
        gradcam_layer = model.layer4[-1]
        gradcam_obj = LayerGradCam(model, gradcam_layer)

    elif base_model == "Swin-T":
        model = swin_t(weights=None)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        swin_layer = model.features[0][0]
        gradcam_obj = LayerGradCam(model, swin_layer)

    else:
        raise ValueError("Modelo base não suportado para custom .pth")

    return model, gradcam_obj