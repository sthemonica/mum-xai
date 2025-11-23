import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import IntegratedGradients, Occlusion, LayerAttribution

def explain_ig(model, x, target=None, n_steps=50, baseline=None):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    if baseline is None:
        baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    attr = attributions.abs().sum(dim=1, keepdim=True)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr.detach().cpu()  # (1,1,H,W)


def explain_occlusion(model, x, target=None, window=15, stride=8, baseline=0):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    occ = Occlusion(model)
    attributions = occ.attribute(
        x,
        strides=(3, stride, stride),
        sliding_window_shapes=(3, window, window),
        baselines=baseline,
        target=target,
    )

    attr = attributions.mean(dim=1, keepdim=True)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr.detach().cpu()  # (1,1,H,W)


def explain_gradcam(model, gradcam_obj, x, target=None):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    attributions = gradcam_obj.attribute(x, target=target)
    upsampled = LayerAttribution.interpolate(attributions, x.shape[2:])
    attr = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    return attr.detach().cpu()  # (1,1,H,W)


def run_xai_heatmap(model, gradcam_obj, x, method="GradCAM", target=None):
    """
    x: tensor [1,3,224,224] (já normalizado)
    return:
      - heatmap (224,224) np.float32 entre 0-1
      - target_class (int)
      - prob_target (float)
    """
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        if target is None:
            target = int(probs.argmax(dim=1).item())
        prob_target = float(probs[0, target].item())

    if method == "GradCAM":
        attr = explain_gradcam(model, gradcam_obj, x, target=target)
    elif method == "Integrated Gradients":
        attr = explain_ig(model, x, target=target, baseline=torch.zeros_like(x))
    elif method == "Occlusion":
        attr = explain_occlusion(model, x, target=target)
    else:
        attr = explain_gradcam(model, gradcam_obj, x, target=target)

    hmap = attr.squeeze().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    return hmap.astype(np.float32), target, prob_target
