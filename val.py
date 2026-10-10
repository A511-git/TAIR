import os
import argparse
import wandb
import pyiqa
import numpy as np
from PIL import Image 
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from accelerate.utils import set_seed
from terediff.utils.common import instantiate_from_config, text_to_image
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
import initialize


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)

    # Allow CLI overrides for input and output directories
    if getattr(args, 'input_dir', None):
        cfg.dataset.lq_img_path = args.input_dir
    if getattr(args, 'output_dir', None):
        cfg.exp_args.save_val_img_dir = args.output_dir
    if getattr(args, 'gt_dir', None):
        cfg.dataset.gt_img_path = args.gt_dir
    
    # setup logging tool
    if cfg.log_args.log_tool == 'wandb':
        wandb.login(key=cfg.log_args.wandb_key)
        wandb.init(project=cfg.log_args.wandb_proj_name, 
                name='VAL_terediff_stage3_DEMO',
                config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
        )

    
    # load demo images from dataset folders (recursively scanning all subdirectories)
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    if not os.path.exists(cfg.dataset.lq_img_path):
        raise FileNotFoundError(f"Input image directory does not exist: {cfg.dataset.lq_img_path}")

    def collect_images_recursively(dir_path):
        collected = []
        for root, _, files in os.walk(dir_path):
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.lower().endswith(valid_exts):
                    collected.append(p)
                else:
                    try:
                        with Image.open(p) as test_img:
                            test_img.verify()
                            collected.append(p)
                    except Exception:
                        pass
        return sorted(collected)

    lq_imgs_path = collect_images_recursively(cfg.dataset.lq_img_path)
    
    gt_dir = getattr(cfg.dataset, 'gt_img_path', None)
    has_gt = bool(gt_dir) and os.path.exists(str(gt_dir))
    gt_imgs_path = collect_images_recursively(gt_dir) if has_gt else []
    
    eval_mode = has_gt and len(gt_imgs_path) == len(lq_imgs_path)
    if not eval_mode and has_gt:
        print(f"[*] Note: GT image count ({len(gt_imgs_path)}) != LQ image count ({len(lq_imgs_path)}). Running in pure restoration inference mode.")

                    
    # load models
    models, _ = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup model accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)
    metric_dists = pyiqa.create_metric('dists', device=device)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_musiq = pyiqa.create_metric('musiq', device=device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=device)


    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]
    tot_val_dists=[]
    tot_val_niqe=[]
    tot_val_musiq=[]
    tot_val_maniqa=[]
    tot_val_clipiqa=[]
    

    # set seed for identical generation for validation sampling noise
    gen.manual_seed(25)
    
    
    # put model on eval
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()


    # For val_gt (range [-1, 1])
    preprocess_gt = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For val_lq (range [0, 1])
    preprocess_lq = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    if eval_mode:
        img_iterator = zip(gt_imgs_path, lq_imgs_path)
        total_items = len(gt_imgs_path)
    else:
        img_iterator = [(None, lq_path) for lq_path in lq_imgs_path]
        total_items = len(lq_imgs_path)

    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(img_iterator, desc='val/restore', total=total_items)):
        lq_id = os.path.splitext(os.path.basename(lq_img_path))[0]
        lq_img = Image.open(lq_img_path).convert("RGB")
        val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)
        val_bs, _, val_H, val_W = val_lq.shape

        if gt_img_path is not None:
            gt_id = os.path.splitext(os.path.basename(gt_img_path))[0]
            assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
            gt_img = Image.open(gt_img_path).convert("RGB")
            val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)
        else:
            gt_id = lq_id
            val_gt = None
        
        val_prompt = [""]
            
        with torch.no_grad():
            val_clean = models['swinir'](val_lq)   
            val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

            M=1
            pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
            
            models['testr'].test_score_threshold = 0.5   
            ts_model = models['testr']

            # sampling
            val_z, val_ts_results = sampler.val_sample(    
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T = pure_noise,
                progress=accelerator.is_main_process,
                cfg=cfg, 
                pure_cldm=pure_cldm,
                ts_model = ts_model,
                val_prompt=val_prompt
            )
            
            # log val prompts
            val_prompt = val_prompt[0]
            lines = []
            lines.append(f"** using OCR prompt w/ {cfg.exp_args.prompt_style}style **\n")
            # Format prompt
            lines.append("initial input prompt:\n")
            width = 80
            for i in range(0, len(val_prompt), width):
                lines.append(val_prompt[i:i+width] + "\n")
            lines.append("\n")
            
            # Add prediction results
            for ts_result in val_ts_results:
                timestep = ts_result['timestep']
                pred_texts = ', '.join(ts_result['pred_texts'])
                lines.append(f"timestep: {timestep:<4} /  pred_texts: {pred_texts}\n")
            
            # Now convert the list of strings to image
            img_of_pred_text = text_to_image(lines)
            
            restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
            
            # save sampled images   
            if cfg.log_args.log_tool is None:
                # Mirror original subdirectory structure relative to input dataset directory
                rel_path = os.path.relpath(lq_img_path, cfg.dataset.lq_img_path)
                sub_dir = os.path.dirname(rel_path)
                orig_filename = os.path.basename(lq_img_path)

                target_dir = os.path.join(cfg.exp_args.save_val_img_dir, sub_dir)
                os.makedirs(target_dir, exist_ok=True)

                out_file_path = os.path.join(target_dir, orig_filename)
                if not os.path.splitext(out_file_path)[1]:
                    out_file_path += ".png"

                restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
                restored_img_pil.save(out_file_path)

                if getattr(args, 'save_pred_texts', False):
                    pred_dir = os.path.join(cfg.exp_args.save_val_img_dir, "pred_texts", sub_dir)
                    os.makedirs(pred_dir, exist_ok=True)
                    pred_filename = orig_filename if os.path.splitext(orig_filename)[1] else f"{orig_filename}.png"
                    img_of_pred_text.save(os.path.join(pred_dir, pred_filename))
            
            # log total psnr, ssim, lpips for val if ground truth is available
            if val_gt is not None:
                tot_val_psnr.append(torch.mean(metric_psnr(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_ssim.append(torch.mean(metric_ssim(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_lpips.append(torch.mean(metric_lpips(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_dists.append(torch.mean(metric_dists(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_niqe.append(torch.mean(metric_niqe(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_musiq.append(torch.mean(metric_musiq(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_maniqa.append(torch.mean(metric_maniqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_clipiqa.append(torch.mean(metric_clipiqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                
                # log sampling val imgs to wandb
                if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                    wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            })
                    
                    wandb.log({ f'sampling_val_FINAL_VIS/{gt_id}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                                f'sampling_val_FINAL_VIS/{gt_id}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                                f'sampling_val_FINAL_VIS/{gt_id}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                                f'sampling_val_FINAL_VIS/{gt_id}_val_sampled': wandb.Image(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                                f'sampling_val_FINAL_VIS/{gt_id}_val_prompts': wandb.Image(img_of_pred_text, caption='prompts used for sampling'),
                            })
                    wandb.log({f'sampling_val_FINAL_VIS/{gt_id}_val_all': wandb.Image(torch.concat([val_lq, val_clean, torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_gt], dim=2), caption='lq_clean_sample,gt')})

    # average using numpy if evaluation metrics were recorded
    if len(tot_val_psnr) > 0:
        tot_val_psnr = np.array(tot_val_psnr).mean()
        tot_val_ssim = np.array(tot_val_ssim).mean()
        tot_val_lpips = np.array(tot_val_lpips).mean()
        tot_val_dists = np.array(tot_val_dists).mean()
        tot_val_niqe = np.array(tot_val_niqe).mean()
        tot_val_musiq = np.array(tot_val_musiq).mean()
        tot_val_maniqa = np.array(tot_val_maniqa).mean()
        tot_val_clipiqa = np.array(tot_val_clipiqa).mean()

        print(f"\n[+] Validation Metrics Summary:")
        print(f"    - PSNR: {tot_val_psnr:.2f} dB | SSIM: {tot_val_ssim:.4f} | LPIPS: {tot_val_lpips:.4f}")
        print(f"    - DISTS: {tot_val_dists:.4f} | NIQE: {tot_val_niqe:.4f} | MUSIQ: {tot_val_musiq:.2f}")

        # log total val metrics 
        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
            wandb.log({
                f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
                f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
                f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
                f'sampling_val_METRIC/tot_val_dists': tot_val_dists,
                f'sampling_val_METRIC/tot_val_niqe': tot_val_niqe,
                f'sampling_val_METRIC/tot_val_musiq': tot_val_musiq,
                f'sampling_val_METRIC/tot_val_maniqa': tot_val_maniqa,
                f'sampling_val_METRIC/tot_val_clipiqa': tot_val_clipiqa,
            })
    else:
        print(f"\n[+] Inference Complete: All restored images saved to: {cfg.exp_args.save_val_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAIR TeReDiff Model Inference & Validation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML model configuration")
    parser.add_argument('--config_testr', type=str, default=None, help="Path to TESTR detector configuration")
    parser.add_argument('--input_dir', '--lq_dir', type=str, default=None, help="Directory containing low-quality images (overrides config)")
    parser.add_argument('--output_dir', '--save_dir', type=str, default=None, help="Directory to save restored images (overrides config)")
    parser.add_argument('--gt_dir', type=str, default=None, help="Optional directory containing ground-truth images")
    parser.add_argument('--save_pred_texts', action="store_true", default=False, help="Save OCR text prediction visualization images in a separate folder")
    args = parser.parse_args()
    main(args)
