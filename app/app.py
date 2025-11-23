import io
import sys
import time
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
# adding the root path to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DEVICE, RESNET_CKPT_PATH, SWIN_CKPT_PATH, CLASS_NAMES
from src.models import (
    load_resnet_model,
    load_swin_model,
    load_custom_model_from_pth,
)
from src.segmentation import run_cellpose_and_crop
from src.reconstruction import reconstruct_full_heatmap
from src.xai_utils import run_xai_heatmap

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="🔬MUM-XAI", layout="wide")

# =========================================================
# ESTILO / CSS
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }
    .pipeline-step {
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.8rem;
    }
    .pipeline-step h3 {
        margin-bottom: 0.3rem;
    }
    .gallery-title {
        text-align: center;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .img-name {
        font-family: monospace;
        font-size: 0.85rem;
        color: #555;
    }
    .info-bar {
        background-color: #ffe0e0;
        padding: 0.4rem 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid #f5b3b3;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .result-box {
        background-color: #d6e9ff;
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
        border: 1px solid #a9c9ff;
        margin-top: 0.8rem;
        font-weight: 600;
        text-align: center;
    }
    /* caixas de diagnóstico verde/vermelho */
    .diagnosis-box {
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
        margin-top: 0.8rem;
        font-weight: 600;
        text-align: center;
        color: #ffffff;
    }
    .diagnosis-ok {
        background-color: #4CAF50;
        border: 1px solid #2E7D32;
    }
    .diagnosis-alert {
        background-color: #E53935;
        border: 1px solid #B71C1C;
    }
    /* NOVO: texto de tempos por imagem */
    .runtime-text {
        font-size: 0.85rem;
        margin-top: 0.4rem;
        color: #333333;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background-color:#ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #ececec;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    ">
        <h1 style="color:#222; text-align:center; margin-bottom:0.4rem;">
            🔬MUM-XAI: Malaria Under Microscope Explainable AI
        </h1>

        MUM-XAI is a complete analysis pipeline for malaria detection in microscope images. The system automatically 
        segments blood cells, classifies each one using deep-learning models, and applies explainability methods to 
        reveal why a cell was predicted as infected or healthy. By uploading your images, you can generate clear 
        visualizations—such as masks, heatmaps, and highlighted infected cells—and download organized reports to 
        support research, teaching, or clinical workflow exploration.
    </div>
    """,
    unsafe_allow_html=True,
)



# =========================================================
# TRANSFORMS
# =========================================================
normalize_imagenet = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

transform_gray = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize_imagenet,
    ]
)


def load_image_pil(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")


def tensor_from_pil_gray(pil_img):
    x = transform_gray(pil_img)
    return x.unsqueeze(0).to(DEVICE)  # [1,3,224,224]

# =========================================================
# SESSION STATE FOR GALERY RESULTS
# =========================================================
if "images_info" not in st.session_state:
    st.session_state.images_info = []  # lista de dicts com resultados por imagem
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
# chave para resetar o file_uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
# NOVO: tempo total da pipeline
if "total_runtime" not in st.session_state:
    st.session_state.total_runtime = None


# =========================================================
# SIDEBAR – PIPELINE CONFIGURATIONS
# =========================================================
st.sidebar.header("⚙️ Pipeline Configurations")

# clear all and make new upload
if st.sidebar.button("🗑️ Remove images and insert new ones"):
    st.session_state.images_info = []
    st.session_state.df_final = None
    st.session_state.current_idx = 0
    st.session_state.uploader_key += 1  # new reset
    st.session_state.total_runtime = None
    st.rerun()

uploaded_imgs = st.sidebar.file_uploader(
    "Send until 10 imagens (BMP/JPG/PNG)",
    type=["bmp", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploader_key}",
)

# limiting to 10 images
if uploaded_imgs and len(uploaded_imgs) > 10:
    st.sidebar.warning("You send more than 10 images. Only the first 10 will be processed.")
    uploaded_imgs = uploaded_imgs[:10]

st.sidebar.markdown("---")
st.sidebar.subheader(" Choose the AI model")

model_option = st.sidebar.selectbox(
    "Choose the model checkpoint",
    ["ResNet50", "Swin-T"],
)

use_custom_model = st.sidebar.checkbox("Send a new model .pth file")
custom_model_file = None
base_model_for_custom = None

if use_custom_model:
    custom_model_file = st.sidebar.file_uploader(
        "Send the .pth file",
        type=["pth"],
    )
    base_model_for_custom = st.sidebar.selectbox(
        "Base model .pth",
        ["ResNet50", "Swin-T"],
    )

st.sidebar.markdown("---")
st.sidebar.subheader("XAI method")

xai_method = st.sidebar.selectbox(
    "Select the XAI method",
    ["GradCAM", "Integrated Gradients", "Occlusion"],
)

st.sidebar.markdown("---")
run_pipeline = st.sidebar.button("Run the pipeline")


# =========================================================
# PIPELINE EXECUTION (One time run and store in session_state)
# =========================================================
if run_pipeline and uploaded_imgs:
    if use_custom_model and custom_model_file is None:
        st.error("You check the option fot model with .pth file, but don't send it.")
    else:
        # clean the preview results
        st.session_state.images_info = []
        st.session_state.df_final = None
        st.session_state.current_idx = 0
        st.session_state.total_runtime = None

        total_imgs = len(uploaded_imgs)
        geral_progress = st.progress(0.0)
        geral_status = st.empty()

        # Load the model just 1 time
        if use_custom_model:
            model, gradcam_obj = load_custom_model_from_pth(
                custom_model_file, base_model_for_custom
            )
        else:
            if model_option == "ResNet50":
                model, gradcam_obj = load_resnet_model(RESNET_CKPT_PATH)
            else:
                model, gradcam_obj = load_swin_model(SWIN_CKPT_PATH)

        df_lista = []

        # timer
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_total_start = time.time()

        for idx_img, uploaded_img in enumerate(uploaded_imgs):
            nome_img = uploaded_img.name
            geral_status.markdown(
                f" **Processing image {idx_img+1}/{total_imgs}:** `{nome_img}`"
            )

            original_img = load_image_pil(uploaded_img)

            # --- begin the time of image ---
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_img_start = time.time()

            # 1) CELLPOSE
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_cellpose_start = time.time()
            cells, mask_viz = run_cellpose_and_crop(original_img)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_cellpose = time.time() - t_cellpose_start

            # 2) portraits + XAI
            max_cells_xai = 40
            cells_subset = cells[:max_cells_xai]

            cells_with_heatmaps = []
            rows = []

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_xai_start = time.time()

            for i, cell in enumerate(cells_subset):
                heatmap, target_class, prob_target = run_xai_heatmap(
                    model,
                    gradcam_obj,
                    tensor_from_pil_gray(cell["crop"]),
                    method=xai_method,
                )

                class_name = CLASS_NAMES[target_class]

                cells_with_heatmaps.append(
                    {
                        "bbox": cell["bbox"],
                        "heatmap": heatmap,
                        "target_class": target_class,
                        "class_name": class_name,
                        "prob": prob_target,
                        "index": i,
                    }
                )

                y0, x0, y1, x1 = cell["bbox"]
                rows.append(
                    {
                        "imagem": nome_img,
                        "cell_index": i,
                        "y0": y0,
                        "x0": x0,
                        "y1": y1,
                        "x1": x1,
                        "pred_class_id": target_class,
                        "pred_class_name": class_name,
                        "pred_prob": prob_target,
                    }
                )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_xai = time.time() - t_xai_start

            df_img = pd.DataFrame(rows)
            df_lista.append(df_img)

            # 3) reconstruction
            t_recon_start = time.time()
            overlay_img = reconstruct_full_heatmap(original_img, cells_with_heatmaps)
            t_recon = time.time() - t_recon_start

            # 4) diagnosis
            t_diag_start = time.time()
            has_infected = df_img["pred_class_id"].eq(1).any()

            if has_infected:
                diagnosis_text = (
                    f"⚠️ The image {nome_img} was classified as "
                    f"CONTAMINATED WITH MALARIA. At least one cell was "
                    f"identified as infected."
                )
                img_diag = original_img.convert("RGB").copy()
                draw = ImageDraw.Draw(img_diag)
                for row in df_img.itertuples():
                    if row.pred_class_id == 1:
                        draw.ellipse(
                            (row.x0, row.y0, row.x1, row.y1),
                            outline="#26FF00",
                            width=10,
                        )
            else:
                diagnosis_text = (
                    f"✅ The image {nome_img} was classified as "
                    f"UNCONTAMINATED. No cells were identified as altered."
                )
                img_diag = original_img

            t_diag = time.time() - t_diag_start

            # --- TOTAL TIMER FOR IMAGE ---
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_img_total = time.time() - t_img_start

            # store all info in session_state
            st.session_state.images_info.append(
                {
                    "name": nome_img,
                    "original": original_img,
                    "mask_viz": mask_viz,
                    "overlay": overlay_img,
                    "diagnosis_text": diagnosis_text,
                    "diagnosis_img": img_diag,
                    "df_img": df_img,
                    "has_infected": bool(has_infected),
                    "runtime_sec": t_img_total,
                    "runtime_cellpose": t_cellpose,
                    "runtime_xai": t_xai,
                    "runtime_recon": t_recon,
                    "runtime_diag": t_diag,
                }
            )

            geral_progress.progress((idx_img + 1) / total_imgs)

        # total pipeline timer
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        st.session_state.total_runtime = time.time() - t_total_start

        # final csv
        if df_lista:
            st.session_state.df_final = pd.concat(df_lista, ignore_index=True)

        geral_status.markdown("✅ Pipeline finished! Check the results below.")


# =========================================================
# GALERY – IMAGE NAVIGATOR
# =========================================================
images_info = st.session_state.images_info

if images_info:
    n_imgs = len(images_info)
    idx = st.session_state.current_idx
    idx = max(0, min(idx, n_imgs - 1)) 
    st.session_state.current_idx = idx
    info = images_info[idx]

    # General informations bar (model, XAI, etc.)
    if use_custom_model and base_model_for_custom:
        modelo_str = f"Model: Custom {base_model_for_custom}"
    else:
        modelo_str = f"Model: {model_option}"

    info_bar_text = (
        f"{modelo_str} • XAI method: {xai_method} • "
        f"Processed images: {n_imgs}"
    )

    # Total time and average time per image.
    total_rt = st.session_state.get("total_runtime", None)
    if total_rt is not None and n_imgs > 0:
        media = total_rt / n_imgs
        info_bar_text += f" • Total time: {total_rt:.1f}s • Mean/img: {media:.2f}s"

    st.markdown(
        f'<div class="info-bar">{info_bar_text}</div>',
        unsafe_allow_html=True,
    )

    # Bar with image name + navigation (↔)
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

    with col_nav1:
        prev_disabled = (idx == 0)
        if st.button("⬅️ Previous", use_container_width=True, disabled=prev_disabled):
            if st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()

    with col_nav3:
        next_disabled = (idx == n_imgs - 1)
        if st.button("Next ➡️", use_container_width=True, disabled=next_disabled):
            if st.session_state.current_idx < n_imgs - 1:
                st.session_state.current_idx += 1
                st.rerun()

    with col_nav2:
        st.markdown(
            f"<div class='gallery-title'>Image {idx+1} of {n_imgs}<br>"
            f"<span class='img-name'>{info['name']}</span></div>",
            unsafe_allow_html=True,
        )

    # Main layout: thumbnail column + 2x2 image grid
    col_thumbs, col_main = st.columns([1, 3])

    # ---- Miniatures in 2xN grid on the left. ----
    with col_thumbs:
        st.markdown("#### 📂 Images")

        n = len(images_info)
        for start in range(0, n, 2):
            c1, c2 = st.columns(2)

            i = start
            with c1:
                im_info = images_info[i]
                st.image(
                    im_info["original"],
                    use_container_width=True,
                    caption=f"{i+1}",
                )
                if st.button(f"Select {i+1}", key=f"thumb_btn_{i}"):
                    st.session_state.current_idx = i
                    st.rerun()

            j = start + 1
            if j < n:
                with c2:
                    im_info2 = images_info[j]
                    st.image(
                        im_info2["original"],
                        use_container_width=True,
                        caption=f"{j+1}",
                    )
                    if st.button(f"Select {j+1}", key=f"thumb_btn_{j}"):
                        st.session_state.current_idx = j
                        st.rerun()

    # ---- Right column: 2x2 (original, cellpose, XAI, highlighted cells) ----
    with col_main:
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            st.markdown("#### 🖼️ Original image")
            st.image(info["original"], use_container_width=True)

        with row1_col2:
            st.markdown("#### 🧬 Cellpose Mask")
            st.image(info["mask_viz"], use_container_width=True)

        with row2_col1:
            st.markdown("#### 🌡️ Image with XAI")
            st.image(info["overlay"], use_container_width=True)

        with row2_col2:
            st.markdown("#### 🔴 Highlighted cells")
            st.image(
                info["diagnosis_img"],
                caption="Cells labeled according to model classification. Green circles indicate infected cells.",
                use_container_width=True,
            )

        # Download buttons (cellpose, XAI, highlighted cells)
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        # Cellpose
        buf_mask = io.BytesIO()
        info["mask_viz"].save(buf_mask, format="PNG")
        buf_mask.seek(0)
        with col_dl1:
            st.download_button(
                label="⬇️ Download Cellpose image (PNG)",
                data=buf_mask,
                file_name=f"cellpose_{info['name']}.png",
                mime="image/png",
            )

        # XAI
        buf_xai = io.BytesIO()
        info["overlay"].save(buf_xai, format="PNG")
        buf_xai.seek(0)
        with col_dl2:
            st.download_button(
                label="⬇️ Download XAI image (PNG)",
                data=buf_xai,
                file_name=f"xai_overlay_{info['name']}.png",
                mime="image/png",
            )

        # Highlighted cells
        buf_diag = io.BytesIO()
        info["diagnosis_img"].save(buf_diag, format="PNG")
        buf_diag.seek(0)
        with col_dl3:
            st.download_button(
                label="⬇️ Download highlighted cells (PNG)",
                data=buf_diag,
                file_name=f"cells_highlight_{info['name']}.png",
                mime="image/png",
            )

        # green/red result box
        has_infected_flag = info.get("has_infected", False)
        css_class = "diagnosis-alert" if has_infected_flag else "diagnosis-ok"
        st.markdown(
            f'<div class="diagnosis-box {css_class}">{info["diagnosis_text"]}</div>',
            unsafe_allow_html=True,
        )

        # current image times
        rt_total = info.get("runtime_sec", None)
        rt_cellpose = info.get("runtime_cellpose", None)
        rt_xai = info.get("runtime_xai", None)
        rt_recon = info.get("runtime_recon", None)
        rt_diag = info.get("runtime_diag", None)

        if rt_total is not None:
            st.markdown(
                f"""
                <div class="runtime-text">
                ⏱ <b>Times of this image</b><br>
                • Total: {rt_total:.2f}s<br>
                • Cellpose: {rt_cellpose:.2f}s • XAI: {rt_xai:.2f}s<br>
                • Reconstruction: {rt_recon:.2f}s • Diagnosis: {rt_diag:.2f}s
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Table and CSV of this image
    st.markdown("#### 📊 Results by cell (current image)")
    st.dataframe(info["df_img"], use_container_width=True)

    csv_img_bytes = info["df_img"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download CSV image {info['name']}",
        data=csv_img_bytes,
        file_name=f"cell_results_{info['name']}.csv",
        mime="text/csv",
    )

# =========================================================
# CSV FINAL CONSOLIDATED (ALL IMAGES)
# =========================================================
if st.session_state.df_final is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>📊 Consolidated CSV (all images)</h3></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(st.session_state.df_final, use_container_width=True)

    st.download_button(
        "⬇️ Download consolidated CSV (all images)",
        st.session_state.df_final.to_csv(index=False).encode("utf-8"),
        file_name="resultados_multiplos_imagens.csv",
        mime="text/csv",
    )
