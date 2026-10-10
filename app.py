#!/usr/bin/env python3
"""
TAIR: Text-Aware Image Restoration with Diffusion Models (ICLR 2026)
Interactive Gradio Web Application
Exposes model restoration, text spotting, and diffusion sampling in a browser UI.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
import gradio as gr

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

    class DummyAccelerator:
        is_main_process = True
        device = device

        def prepare(self, m):
            return m

        def unwrap_model(self, m):
            return m

    dummy_acc = DummyAccelerator()
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


def restore_image(input_image, prompt_style="CAPTION", steps=50, cfg_scale=1.0, score_threshold=0.5):
    if input_image is None:
        return None, None, "Please upload or select an input image."

    # Check for weights before attempting inference
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


def create_demo():
    demo_images = []
    lq_dir = REPO_ROOT / "assets" / "demo_imgs" / "lq"
    if lq_dir.exists():
        for f in sorted(lq_dir.glob("*.jpg")) + sorted(lq_dir.glob("*.png")):
            demo_images.append(str(f))

    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto; }
    .header-banner { text-align: center; margin-bottom: 1.5rem; }
    """

    with gr.Blocks(title="TAIR: Text-Aware Image Restoration", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["header-banner"]):
            gr.Markdown(
                """
                # 🌟 TAIR: Text-Aware Image Restoration with Diffusion Models
                ### **ICLR 2026**
                Restore heavily degraded scene images while accurately preserving and super-resolving embedded text.
                """
            )

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
            fn=restore_image,
            inputs=[input_img, prompt_style, steps, cfg_scale, score_threshold],
            outputs=[output_restored, output_spotted, detected_text_box]
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app = create_demo()
    print(f"[*] Launching TAIR Gradio Web UI on 0.0.0.0:{port}...")
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
