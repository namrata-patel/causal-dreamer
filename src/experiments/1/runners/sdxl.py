import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# ---------------------------- 
# Robust project root
# ---------------------------- 
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "outputs").exists() and (p / "data").exists():
            return p
        p = p.parent
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
# Main generator for SDXL Sequential Generation
# ---------------------------- 
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--out_method", type=str, default="sdxl_seq")
    ap.add_argument("--num_frames", type=int, default=5)

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)

    # Sequential knobs
    ap.add_argument("--strength", type=float, default=0.55, help="img2img strength for frames 1..T-1 (0..1)")
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

    # 1) Base SDXL for frame0 (text2img)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load SDXL text2img pipeline
    sdxl_txt2img = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    sdxl_txt2img.enable_attention_slicing()

    # 2) Load SDXL img2img pipeline (shares components with txt2img)
    sdxl_img2img = StableDiffusionXLImg2ImgPipeline(
        vae=sdxl_txt2img.vae,
        text_encoder=sdxl_txt2img.text_encoder,
        text_encoder_2=sdxl_txt2img.text_encoder_2,
        tokenizer=sdxl_txt2img.tokenizer,
        tokenizer_2=sdxl_txt2img.tokenizer_2,
        unet=sdxl_txt2img.unet,
        scheduler=sdxl_txt2img.scheduler,
    ).to(device)
    sdxl_img2img.enable_attention_slicing()

    for row_id, base_prompt in enumerate(tqdm(prompts, desc=f"Generate {args.out_method}")):
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")

        # ---- build prompts per frame ----
        frame_prompts = [base_prompt] + [
            f"{base_prompt} Frame {t}/{args.num_frames}." for t in range(2, args.num_frames + 1)
        ]

        frames: List[Image.Image] = []
        frame_paths: List[str] = []

        # ---- frame0: text2img ----
        g0 = torch.Generator(device=device).manual_seed(args.seed + row_id * 1000 + 0)
        img0 = sdxl_txt2img(
            prompt=frame_prompts[0],
            height=args.img_size,
            width=args.img_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=g0,
        ).images[0]

        p0 = sample_dir / "00.png"
        img0.save(p0)
        frames.append(img0)
        frame_paths.append(str(p0))

        # ---- frames 1..T-1: img2img ----
        prev = img0
        for t in range(1, args.num_frames):
            g = torch.Generator(device=device).manual_seed(args.seed + row_id * 1000 + t)
            img_t = sdxl_img2img(
                prompt=frame_prompts[t],
                image=prev,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=g,
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
            "sd_model": model_id,
            "steps": args.steps,
            "guidance": args.guidance,
            "img2img_strength": args.strength,
            "seed_rule": f"seed = {args.seed} + row_id*1000 + t",
            "img_size": args.img_size,
        }
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Done. Outputs at:", out_root)


if __name__ == "__main__":
    main()