#!/usr/bin/env python3
"""
TAIR: Text-Aware Image Restoration with Diffusion Models (ICLR 2026)
Interactive & Batch Gradio Web Application
Supports Single-Image Restoration and Batch / Folder Processing with ZIP Download.
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
import gradio as gr

# Patch Gradio 4.43.0 ASGI schema bug where boolean additionalProperties crashes json_schema_to_python_type
try:
    import gradio_client.utils as gcu
    _orig_schema_conv = gcu._json_schema_to_python_type
    def _safe_schema_conv(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_schema_conv(schema, defs)
    gcu._json_schema_to_python_type = _safe_schema_conv
except Exception:
    pass

# Ensure repository root and testr are on Python path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "testr"))

from terediff.utils.common import instantiate_from_config, text_to_image
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
import initialize

# Global pipeline cache
PIPELINE = None
VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


def is_valid_image_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() in VALID_IMAGE_EXTS:
        return True
    try:
        with Image.open(p) as img:
            img.verify()
            return True
    except Exception:
        return False


def load_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("[*] Loading TAIR TeReDiff pipeline and model weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = REPO_ROOT / "configs" / "val" / "val_terediff.yaml"
    config_testr_path = REPO_ROOT / "testr" / "configs" / "TESTR" / "TESTR_R_50_Polygon.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))

    class DummyArgs:
        config = str(config_path)
        config_testr = str(config_testr_path)

    # Properly scoped DummyAccelerator class accepting device
    class DummyAccelerator:
        def __init__(self, dev):
            self.device = dev
            self.is_main_process = True

        def prepare(self, *m):
            return m[0] if len(m) == 1 else m

        def unwrap_model(self, m):
            return m

    dummy_acc = DummyAccelerator(device)
    args = DummyArgs()

    models, _ = initialize.load_model(dummy_acc, device, args, cfg)
    pure_cldm = models["cldm"]

    diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    for m in models.values():
        if isinstance(m, torch.nn.Module):
            m.eval()

    preprocess_lq = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])

    gen = torch.Generator(device)
    gen.manual_seed(25)

    PIPELINE = {
        "cfg": cfg,
        "models": models,
        "pure_cldm": pure_cldm,
        "diffusion": diffusion,
        "sampler": sampler,
        "preprocess_lq": preprocess_lq,
        "device": device,
        "gen": gen
    }
    print("[+] TAIR TeReDiff pipeline successfully loaded and ready for inference!")
    return PIPELINE


def restore_single_image(input_image, prompt_style="CAPTION", steps=50, cfg_scale=1.0, score_threshold=0.5):
    if input_image is None:
        return None, None, "Please upload or select an input image."

    weights_path = REPO_ROOT / "weights" / "terediff_stage3.pt"
    if not weights_path.exists():
        msg = (
            "⚠️ Model weights not found at ./weights/terediff_stage3.pt!\n\n"
            "Please run `bash download_weights.sh` in the terminal to automatically "
            "download and verify all model checkpoints."
        )
        return None, None, msg

    pipe = load_pipeline()
    cfg = pipe["cfg"]
    models = pipe["models"]
    pure_cldm = pipe["pure_cldm"]
    sampler = pipe["sampler"]
    device = pipe["device"]
    preprocess_lq = pipe["preprocess_lq"]
    gen = pipe["gen"]

    cfg.exp_args.prompt_style = prompt_style
    models['testr'].test_score_threshold = float(score_threshold)
    ts_model = models['testr']

    pil_img = input_image.convert("RGB")
    val_lq = preprocess_lq(pil_img).unsqueeze(0).to(device)
    val_bs, _, val_H, val_W = val_lq.shape
    val_prompt = [""]

    with torch.no_grad():
        val_clean = models['swinir'](val_lq)
        val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)
        pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)

        val_z, val_ts_results = sampler.val_sample(
            model=models['cldm'],
            device=device,
            steps=int(steps),
            x_size=(val_bs, 4, int(val_H / 8), int(val_W / 8)),
            cond=val_cond,
            uncond=None,
            cfg_scale=float(cfg_scale),
            x_T=pure_noise,
            progress=True,
            cfg=cfg,
            pure_cldm=pure_cldm,
            ts_model=ts_model,
            val_prompt=val_prompt
        )

        restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)
        restored_pil = TF.to_pil_image(restored_img.squeeze().cpu())

        # Build prediction report
        val_prompt_text = val_prompt[0] if val_prompt else ""
        lines = [f"** using OCR prompt w/ {prompt_style} style **\n\ninitial input prompt:\n"]
        width = 80
        for i in range(0, len(val_prompt_text), width):
            lines.append(val_prompt_text[i:i + width] + "\n")
        lines.append("\n")

        all_detected = []
        for ts_result in val_ts_results:
            timestep = ts_result.get('timestep', 0)
            pred_texts = ts_result.get('pred_texts', [])
            all_detected.extend(pred_texts)
            lines.append(f"timestep: {timestep:<4} /  pred_texts: {', '.join(pred_texts)}\n")

        pred_text_img = text_to_image(lines)
        unique_texts = list(dict.fromkeys(all_detected))
        summary_text = f"Detected {len(unique_texts)} scene text instance(s):\n" + "\n".join(f"• {t}" for t in unique_texts) if unique_texts else "No text detected."

    return restored_pil, pred_text_img, summary_text


def restore_batch_images(files, use_server_dir=False, prompt_style="CAPTION", steps=50, cfg_scale=1.0, score_threshold=0.5, progress=gr.Progress()):
    weights_path = REPO_ROOT / "weights" / "terediff_stage3.pt"
    if not weights_path.exists():
        return [], "⚠️ Model weights not found at ./weights/terediff_stage3.pt! Please run bash download_weights.sh.", None

    # Each element: (full_path_str, relative_path_str)
    tasks = []
    if use_server_dir:
        server_dir = REPO_ROOT / "uploaded_stuff"
        if server_dir.exists():
            for p in sorted(server_dir.rglob("*")):
                if is_valid_image_file(p):
                    rel = p.relative_to(server_dir)
                    tasks.append((str(p), str(rel)))
    elif files:
        for f in files:
            path_str = f if isinstance(f, str) else getattr(f, "name", str(f))
            p = Path(path_str)
            if p.is_dir():
                for sub in sorted(p.rglob("*")):
                    if is_valid_image_file(sub):
                        rel = sub.relative_to(p)
                        tasks.append((str(sub), str(rel)))
            elif is_valid_image_file(p):
                tasks.append((str(p), p.name))

    if not tasks:
        return [], "⚠️ No valid images found to restore. Please upload images/folder or check ./uploaded_stuff.", None

    total = len(tasks)
    gallery_items = []
    timestamp = int(time.time())
    out_dir = REPO_ROOT / "results" / f"batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = REPO_ROOT / "results" / f"restored_batch_{timestamp}.zip"
    results_root = REPO_ROOT / "results"

    pipe = load_pipeline()
    for idx, (img_path, rel_path) in enumerate(tasks):
        progress((idx + 1) / total, desc=f"Restoring image {idx+1}/{total} ({rel_path})...")
        try:
            with Image.open(img_path) as raw:
                img_pil = raw.convert("RGB")
            
            restored, _, text_summary = restore_single_image(img_pil, prompt_style, steps, cfg_scale, score_threshold)
            if restored:
                rel_p = Path(rel_path)
                # Ensure target file has image extension if original lacked one
                if not rel_p.suffix:
                    rel_p = rel_p.with_suffix(".png")

                # 1. Save in timestamped batch folder preserving directory hierarchy
                out_file = out_dir / rel_p
                out_file.parent.mkdir(parents=True, exist_ok=True)
                restored.save(out_file)

                # 2. Mirror into main ./results/ folder preserving exact subfolders & original filename
                mirror_file = results_root / rel_p
                mirror_file.parent.mkdir(parents=True, exist_ok=True)
                restored.save(mirror_file)

                gallery_items.append((restored, f"{rel_path}"))
        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")

    # Build ZIP archive preserving subfolder hierarchy
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zipf.write(f, arcname=str(f.relative_to(out_dir)))

    summary_msg = (
        f"🎉 **Batch Restoration Complete!**\n\n"
        f"• **Total Images Processed**: {len(gallery_items)} / {total}\n"
        f"• **Subfolder Structure & Original Filenames Preserved** ✅\n"
        f"• **Mirrored to Results**: `{results_root}`\n"
        f"• **Batch Package Ready**: Download ZIP contains full original subfolder tree."
    )
    return gallery_items, summary_msg, str(zip_path)


def create_demo():
    demo_images = []
    lq_dir = REPO_ROOT / "assets" / "demo_imgs" / "lq"
    if lq_dir.exists():
        for f in sorted(lq_dir.glob("*.jpg")) + sorted(lq_dir.glob("*.png")):
            demo_images.append(str(f))

    custom_css = """
    .gradio-container { max-width: 1300px !important; margin: 0 auto; }
    .header-banner { text-align: center; margin-bottom: 1.5rem; }
    """

    with gr.Blocks(title="TAIR: Text-Aware Image Restoration", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["header-banner"]):
            gr.Markdown(
                """
                # 🌟 TAIR: Text-Aware Image Restoration with Diffusion Models
                ### **ICLR 2026** — Accelerated for NVIDIA Blackwell GPUs (`sm_100`/`sm_120`)
                Restore heavily degraded scene images while accurately preserving and super-resolving embedded text.
                """
            )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: Single Image Restoration
            # ==============================================================
            with gr.TabItem("🖼️ Single Image Restoration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(type="pil", label="Low-Quality (Degraded) Input Image")
                        
                        with gr.Accordion("⚙️ Advanced Restoration Parameters", open=False):
                            prompt_style = gr.Dropdown(
                                choices=["CAPTION", "TAG"],
                                value="CAPTION",
                                label="Prompting Style",
                                info="Style of OCR text condition injected into the diffusion prior"
                            )
                            steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                step=5,
                                value=50,
                                label="Sampling Steps",
                                info="Number of DDPM sampling iterations (default: 50)"
                            )
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                step=0.5,
                                value=1.0,
                                label="CFG Scale",
                                info="Classifier-Free Guidance strength"
                            )
                            score_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                step=0.05,
                                value=0.5,
                                label="Text Spotting Score Threshold",
                                info="Confidence threshold for TESTR polygon detector"
                            )

                        restore_btn = gr.Button("🚀 Restore Image & Spot Text", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        output_restored = gr.Image(type="pil", label="✨ High-Quality Restored Output")
                        output_spotted = gr.Image(type="pil", label="🔍 Detected Scene Text & Prediction Map")
                        detected_text_box = gr.Textbox(label="📝 Spotted Text Strings", lines=3)

                if demo_images:
                    gr.Examples(
                        examples=demo_images[:4],
                        inputs=input_img,
                        label="Sample Demo Images (Degraded Inputs)"
                    )

                restore_btn.click(
                    fn=restore_single_image,
                    inputs=[input_img, prompt_style, steps, cfg_scale, score_threshold],
                    outputs=[output_restored, output_spotted, detected_text_box],
                    api_name=False
                )

            # ==============================================================
            # TAB 2: Batch & Folder Restoration
            # ==============================================================
            with gr.TabItem("📁 Batch & Folder Restoration"):
                gr.Markdown(
                    """
                    ### 📁 Bulk Image & Folder Super-Resolution
                    Upload a folder of images or multiple files, or restore images uploaded directly to `./uploaded_stuff` on your NVMe storage.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            file_count="multiple",
                            label="Select or Drag Images / Folder"
                        )
                        use_server_folder = gr.Checkbox(
                            label="📁 Process All Images Already in Server NVMe Folder (./uploaded_stuff)",
                            value=False,
                            info="Processes all images in ./uploaded_stuff and subdirectories (e.g. text_images/)"
                        )

                        with gr.Accordion("⚙️ Batch Restoration Parameters", open=False):
                            batch_prompt_style = gr.Dropdown(
                                choices=["CAPTION", "TAG"],
                                value="CAPTION",
                                label="Prompting Style"
                            )
                            batch_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                step=5,
                                value=50,
                                label="Sampling Steps"
                            )
                            batch_cfg = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                step=0.5,
                                value=1.0,
                                label="CFG Scale"
                            )
                            batch_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                step=0.05,
                                value=0.5,
                                label="Text Spotting Score Threshold"
                            )

                        batch_run_btn = gr.Button("⚡ Restore All Images in Batch", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        batch_status = gr.Markdown("⏳ Waiting for batch job...")
                        batch_zip_download = gr.File(label="📦 Download All Restored Images (ZIP Package)")

                with gr.Row():
                    batch_gallery = gr.Gallery(
                        label="🖼️ Restored Image Gallery",
                        columns=3,
                        rows=2,
                        height="auto",
                        preview=True
                    )

                batch_run_btn.click(
                    fn=restore_batch_images,
                    inputs=[batch_files, use_server_folder, batch_prompt_style, batch_steps, batch_cfg, batch_threshold],
                    outputs=[batch_gallery, batch_status, batch_zip_download],
                    api_name=False
                )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = create_demo()
    print(f"[*] Launching TAIR Gradio Web UI on 0.0.0.0:{port}...")
    app.launch(server_name="0.0.0.0", server_port=port, share=False, show_api=False, inbrowser=False)
