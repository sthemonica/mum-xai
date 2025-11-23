import numpy as np
from PIL import Image
from skimage import measure
from cellpose import models as cp_models, core as cp_core

CELLPOSE_MODEL = None


def get_cellpose_model():
    global CELLPOSE_MODEL
    if CELLPOSE_MODEL is None:
        gpu = cp_core.use_gpu()
        CELLPOSE_MODEL = cp_models.CellposeModel(
            gpu=gpu,
            model_type="cyto2",
        )
        print(f"Cellpose inicializado (gpu={gpu})")
    return CELLPOSE_MODEL


def run_cellpose_and_crop(image_pil, min_area=80, max_cells=150):
    model = get_cellpose_model()
    img_rgb = np.array(image_pil)

    channels = [0, 0]

    masks, flows, styles = model.eval(
        img_rgb,
        channels=channels,
        batch_size=8,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"tile_norm_blocksize": 0},
    )

    regions = measure.regionprops(masks)
    crops = []

    for reg in regions:
        if reg.area < min_area:
            continue
        y0, x0, y1, x1 = reg.bbox
        crop = image_pil.crop((x0, y0, x1, y1))
        crop_224 = crop.resize((224, 224))

        crops.append(
            {
                "crop": crop_224,
                "bbox": (y0, x0, y1, x1),
                "label": reg.label,
            }
        )

        if len(crops) >= max_cells:
            break

    if masks.max() > 0:
        mask_viz = (masks.astype(np.float32) / masks.max() * 255)
    else:
        mask_viz = masks.astype(np.float32)
    mask_viz = mask_viz.astype(np.uint8)
    mask_viz_pil = Image.fromarray(mask_viz)

    return crops, mask_viz_pil
