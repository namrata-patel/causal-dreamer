# src/inference/controlnet_seq_generate.py
# ControlNet + Sequential generation baseline (no your adapter touched)
#
# Output structure (matches your eval assumptions):
#   outputs/controlnet_seq/<NUM_FRAMES>/<row_id>/
#       00.png, 01.png, ..., grid.png, meta.json
#
# Usage:
#   python src/inference/controlnet_seq_generate.py \
#       --prompt_file data/test_prompts.txt \
#       --out_method controlnet_seq \
#       --num_frames 5 \
#       --control_type canny \
#       --device cuda \
#       --max_samples None
#
# Notes:
# - Frame0 is text2img.
# - Frames 1..T-1 are img2img with ControlNet conditioning computed from the previous frame.
# - This is a simple, defensible baseline: "ControlNet + sequential (prev-frame conditioned)".
#
# Dependencies:
#   pip install diffusers transformers accelerate safetensors opencv-python pillow torch

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)

# ----------------------------
# Robust project root
# ----------------------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "outputs").exists() and (p / "data").exists():
            return p
        p = p.parent
    # fallback: just return the folder containing this file
    return start.resolve().parent

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_prompt_list(prompt_file: Path) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise ValueError(f"Prompt file empty: {prompt_file}")
    return prompts

def make_grid_row(pil_images: List[Image.Image], pad=8, bg=(255, 255, 255)) -> Image.Image:
    if len(pil_images) == 0:
        raise ValueError("No images for grid.")
    widths, heights = zip(*[im.size for im in pil_images])
    total_w = sum(widths) + pad * (len(pil_images) - 1)
    max_h = max(heights)
    grid = Image.new("RGB", (total_w, max_h), bg)
    x = 0
    for im in pil_images:
        y = (max_h - im.size[1]) // 2
        grid.paste(im, (x, y))
        x += im.size[0] + pad
    return grid

# ----------------------------
# Control preprocessing
# ----------------------------
def _to_np_rgb(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))

def make_canny_control(pil: Image.Image, low=100, high=200) -> Image.Image:
    import cv2
    img = _to_np_rgb(pil)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges, edges, edges], axis=-1)  # 3ch
    return Image.fromarray(edges)

def make_soft_canny_control(pil: Image.Image, low=50, high=150, blur=3) -> Image.Image:
    """Slightly smoother canny (less brittle)."""
    import cv2
    img = _to_np_rgb(pil)
    if blur and blur > 0:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)

def make_depth_control(pil: Image.Image, device: str) -> Image.Image:
    """
    Uses diffusers depth estimator if available in your env.
    Falls back to a canny-like control if not.
    """
    try:
        from transformers import pipeline as hf_pipeline
        depth_pipe = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0 if device.startswith("cuda") else -1)
        out = depth_pipe(pil.convert("RGB"))
        depth = out["depth"]
        depth = depth.resize(pil.size)
        # normalize to 0..255
        d = np.array(depth).astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d = (d * 255.0).clip(0, 255).astype(np.uint8)
        d = np.stack([d, d, d], axis=-1)
        return Image.fromarray(d)
    except Exception:
        # fallback
        return make_soft_canny_control(pil)

def control_image_from_prev(prev_frame: Image.Image, control_type: str, device: str) -> Image.Image:
    if control_type == "canny":
        return make_canny_control(prev_frame)
    if control_type == "soft_canny":
        return make_soft_canny_control(prev_frame)
    if control_type == "depth":
        return make_depth_control(prev_frame, device=device)
    raise ValueError(f"Unknown control_type: {control_type}")

def controlnet_id(control_type: str) -> str:
    # common ControlNet checkpoints
    if control_type in ("canny", "soft_canny"):
        return "lllyasviel/sd-controlnet-canny"
    if control_type == "depth":
        return "lllyasviel/sd-controlnet-depth"
    raise ValueError(f"No ControlNet id for: {control_type}")

# ----------------------------
# Main generator
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--out_method", type=str, default="controlnet_seq")
    ap.add_argument("--num_frames", type=int, default=5)

    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--control_type", type=str, default="canny", choices=["canny", "soft_canny", "depth"])

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)

    # Sequential knobs
    ap.add_argument("--strength", type=float, default=0.55, help="img2img strength for frames 1..T-1 (0..1)")
    ap.add_argument("--controlnet_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    ap.add_argument("--eta", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    # roots
    if args.project_root is None:
        project_root = find_project_root(Path(__file__))
    else:
        project_root = Path(args.project_root).resolve()

    prompt_file = Path(args.prompt_file)
    if not prompt_file.is_absolute():
        prompt_file = (project_root / prompt_file).resolve()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_root = project_root / "outputs" / args.out_method / str(args.num_frames)
    ensure_dir(out_root)

    print("PROJECT_ROOT:", project_root)
    print("PROMPT_FILE:", prompt_file)
    print("OUT_ROOT:", out_root)
    print("DEVICE:", device)

    prompts = load_prompt_list(prompt_file)
    if args.max_samples is not None:
        prompts = prompts[: args.max_samples]

    # 1) Base SD for frame0
    print("Loading SD text2img...")
    sd = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,  # IMPORTANT: if you want to avoid black images from safety; set to None
        requires_safety_checker=False,
    ).to(device)
    sd.enable_attention_slicing()

    # 2) ControlNet img2img for frames 1..T-1
    print("Loading ControlNet img2img...")
    cnet = ControlNetModel.from_pretrained(
        controlnet_id(args.control_type),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)

    sd_cnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=cnet,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,  # same note
        requires_safety_checker=False,
    ).to(device)
    sd_cnet.enable_attention_slicing()

    # optional: share VAE/UNet weights memory if needed (diffusers already handles)
    # sd_cnet.unet = sd.unet  # only if you know what you're doing

    for row_id, base_prompt in enumerate(tqdm(prompts, desc=f"Generate {args.out_method}")):
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")

        # ---- build prompts per frame (same style you used) ----
        frame_prompts = [base_prompt] + [
            f"{base_prompt} Frame {t}/{args.num_frames}." for t in range(2, args.num_frames + 1)
        ]

        frames: List[Image.Image] = []
        frame_paths: List[str] = []

        # ---- frame0: text2img ----
        g0 = torch.Generator(device=device).manual_seed(args.seed + row_id * 1000 + 0)
        img0 = sd(
            prompt=frame_prompts[0],
            height=args.img_size,
            width=args.img_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=g0,
            eta=args.eta,
        ).images[0]

        p0 = sample_dir / "00.png"
        img0.save(p0)
        frames.append(img0)
        frame_paths.append(str(p0))

        # ---- frames 1..T-1: img2img with ControlNet(prev-frame control) ----
        prev = img0
        for t in range(1, args.num_frames):
            control = control_image_from_prev(prev, args.control_type, device=device)

            gt = torch.Generator(device=device).manual_seed(args.seed + row_id * 1000 + t)
            img_t = sd_cnet(
                prompt=frame_prompts[t],
                image=prev,                    # sequential init
                control_image=control,         # sequential structure
                strength=args.strength,        # how much to change from prev
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gt,
                controlnet_conditioning_scale=args.controlnet_scale,
                eta=args.eta,
            ).images[0]

            pt = sample_dir / f"{t:02d}.png"
            img_t.save(pt)
            frames.append(img_t)
            frame_paths.append(str(pt))
            prev = img_t

        # grid
        grid = make_grid_row(frames, pad=8)
        grid_path = sample_dir / "grid.png"
        grid.save(grid_path)

        meta = {
            "row_id": row_id,
            "base_prompt": base_prompt,
            "prompts": frame_prompts,
            "frame_paths": frame_paths,
            "grid_path": str(grid_path),
            "method": args.out_method,
            "num_frames": args.num_frames,
            "sd_model": args.base_model,
            "control_type": args.control_type,
            "controlnet_id": controlnet_id(args.control_type),
            "steps": args.steps,
            "guidance": args.guidance,
            "img2img_strength": args.strength,
            "controlnet_scale": args.controlnet_scale,
            "seed_rule": f"seed = {args.seed} + row_id*1000 + t",
            "img_size": args.img_size,
            "eta": args.eta,
        }
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Done. Outputs at:", out_root)


if __name__ == "__main__":
    main()
