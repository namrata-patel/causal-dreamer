"""
ControlNet Baseline for Counterfactual Generation
Generates 8 counterfactual variations using ControlNet with different control images

Instead of temporal sequences, this generates diverse outcomes by:
1. Using different effects from CSV as prompts
2. Applying ControlNet with varied conditioning
3. Maintaining visual consistency through control signals

Output structure: outputs/controlnet_counterfactual/8/<row_id>/
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

# ----------------------------
# Utility Functions
# ----------------------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "data").exists():
            return p
        p = p.parent
    return start.resolve().parent

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_cause_effects(csv_file: Path):
    """Load cause-effect pairs from CSV"""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cause = row['cause']
            effects = [row[f'effect{i}'] for i in range(1, 9)]
            data.append({'cause': cause, 'effects': effects})
    return data

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
# Control Image Generation
# ----------------------------
def _to_np_rgb(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"))

def make_canny_control(pil: Image.Image, low=100, high=200) -> Image.Image:
    """Generate canny edge control image"""
    import cv2
    img = _to_np_rgb(pil)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)

def make_soft_canny_control(pil: Image.Image, low=50, high=150, blur=3) -> Image.Image:
    """Softer canny edges for less rigid control"""
    import cv2
    img = _to_np_rgb(pil)
    if blur and blur > 0:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)

def generate_varied_control_image(base_image: Image.Image, variation_idx: int, control_type: str) -> Image.Image:
    """
    Generate varied control images for diversity
    Different canny thresholds create different structural guidance
    """
    if control_type in ("canny", "soft_canny"):
        # Vary canny thresholds for diversity
        low_vals = [50, 70, 90, 110, 60, 80, 100, 120]
        high_vals = [150, 170, 190, 210, 160, 180, 200, 220]
        
        low = low_vals[variation_idx % len(low_vals)]
        high = high_vals[variation_idx % len(high_vals)]
        
        return make_soft_canny_control(base_image, low=low, high=high, blur=3)
    
    # For other control types, just return standard control
    return make_canny_control(base_image)

def controlnet_id(control_type: str) -> str:
    """Get ControlNet checkpoint ID"""
    if control_type in ("canny", "soft_canny"):
        return "lllyasviel/sd-controlnet-canny"
    if control_type == "depth":
        return "lllyasviel/sd-controlnet-depth"
    raise ValueError(f"Unknown control_type: {control_type}")

# ----------------------------
# Main Generator
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--csv_file", type=str, default="data/test_prompts_with_effects.csv")
    ap.add_argument("--out_method", type=str, default="controlnet_counterfactual")
    ap.add_argument("--num_effects", type=int, default=8)

    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--control_type", type=str, default="soft_canny", choices=["canny", "soft_canny", "depth"])

    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--controlnet_scale", type=float, default=0.8)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    # Setup paths
    if args.project_root is None:
        project_root = find_project_root(Path(__file__))
    else:
        project_root = Path(args.project_root).resolve()

    csv_file = Path(args.csv_file)
    if not csv_file.is_absolute():
        csv_file = (project_root / csv_file).resolve()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = project_root / "outputs" / args.out_method / str(args.num_effects)
    ensure_dir(out_root)

    print("="*60)
    print("CONTROLNET COUNTERFACTUAL BASELINE")
    print("="*60)
    print("PROJECT_ROOT:", project_root)
    print("CSV_FILE:", csv_file)
    print("OUT_ROOT:", out_root)
    print("DEVICE:", device)
    print("CONTROL_TYPE:", args.control_type)
    print()

    # Load data
    print("Loading cause-effect data...")
    data = load_cause_effects(csv_file)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} cause-effect sets")
    print()

    # Load pipelines
    print("Loading Stable Diffusion (for base image)...")
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    sd_pipe.enable_attention_slicing()
    
    print("Loading ControlNet pipeline (for variations)...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id(args.control_type),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)

    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    controlnet_pipe.enable_attention_slicing()
    print()

    # Generate for each cause
    for row_id, item in enumerate(tqdm(data, desc="Generating")):
        cause = item['cause']
        effects = item['effects']
        
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")
        
        print(f"\n[{row_id+1}/{len(data)}] {cause}")
        
        # First, generate a base image from cause using regular SD
        base_seed = args.seed + row_id * 1000
        base_gen = torch.Generator(device=device).manual_seed(base_seed)
        
        base_image = sd_pipe(
            prompt=f"{cause}, high quality, detailed",
            height=args.img_size,
            width=args.img_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=base_gen,
        ).images[0]
        
        frames: List[Image.Image] = []
        frame_paths: List[str] = []
        effect_list: List[str] = []

        # Generate each effect variation with ControlNet
        for effect_idx, effect in enumerate(effects):
            # Create varied control image for diversity
            control_image = generate_varied_control_image(
                base_image, 
                effect_idx, 
                args.control_type
            )
            
            # Construct prompt
            prompt = f"Consistent scene: {cause}. Therefore, {effect}"
            
            seed = base_seed + effect_idx + 1
            gen = torch.Generator(device=device).manual_seed(seed)

            # Generate with ControlNet
            img = controlnet_pipe(
                prompt=prompt,
                image=control_image,
                height=args.img_size,
                width=args.img_size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.controlnet_scale,
                generator=gen,
            ).images[0]

            # Save
            img_path = sample_dir / f"{effect_idx:02d}_{effect.replace(' ', '_')[:30]}.png"
            img.save(img_path)
            frames.append(img)
            frame_paths.append(str(img_path))
            effect_list.append(effect)
            
            print(f"  [{effect_idx+1}/{args.num_effects}] {effect}")

        # Save grid
        grid = make_grid_row(frames, pad=8)
        grid_path = sample_dir / "GRID.png"
        grid.save(grid_path)

        # Save metadata
        meta = {
            "row_id": row_id,
            "cause": cause,
            "effects": effect_list,
            "frame_paths": frame_paths,
            "grid_path": str(grid_path),
            "method": args.out_method,
            "num_effects": args.num_effects,
            "sd_model": args.base_model,
            "control_type": args.control_type,
            "controlnet_id": controlnet_id(args.control_type),
            "steps": args.steps,
            "guidance": args.guidance,
            "controlnet_scale": args.controlnet_scale,
            "seed_base": base_seed,
            "img_size": args.img_size,
            "note": "ControlNet baseline for counterfactual generation with varied control signals"
        }
        
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved grid: {grid_path.name}")

    print("\n" + "="*60)
    print(f"✅ COMPLETED: Generated {len(data)} cause-effect sets")
    print(f"   Output: {out_root}")
    print("="*60)


if __name__ == "__main__":
    main()