#!/usr/bin/env python3
"""
InstructPix2Pix Baseline for Counterfactual Generation
Generates 8 variations using sequential editing with effect-based instructions
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
)

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
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cause = row['cause']
            effects = [row[f'effect{i}'] for i in range(1, 9)]
            data.append({'cause': cause, 'effects': effects})
    return data

def make_grid(images: List[Image.Image], pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("No images")
    widths, heights = zip(*[img.size for img in images])
    total_w = sum(widths) + pad * (len(images) - 1)
    max_h = max(heights)
    grid = Image.new("RGB", (total_w, max_h), bg)
    x = 0
    for img in images:
        y = (max_h - img.size[1]) // 2
        grid.paste(img, (x, y))
        x += img.size[0] + pad
    return grid

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--csv_file", type=str, default="data/test_prompts_with_effects.csv")
    parser.add_argument("--out_method", type=str, default="instructpix2pix_counterfactual")
    parser.add_argument("--num_effects", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # Setup
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root(Path(__file__))
    
    csv_file = Path(args.csv_file)
    if not csv_file.is_absolute():
        csv_file = (project_root / csv_file).resolve()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = ensure_dir(project_root / "outputs" / args.out_method / str(args.num_effects))
    
    print("="*60)
    print("INSTRUCTPIX2PIX COUNTERFACTUAL BASELINE")
    print("="*60)
    print(f"CSV: {csv_file}")
    print(f"Output: {out_root}")
    print(f"Device: {device}")
    print()
    
    # Load data
    data = load_cause_effects(csv_file)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Loaded {len(data)} cause-effect sets\n")
    
    # Load models
    print("Loading SD txt2img...")
    txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    txt2img.enable_attention_slicing()
    
    print("Loading InstructPix2Pix...")
    ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    ip2p.enable_attention_slicing()
    print()
    
    # Generate
    for row_id, item in enumerate(tqdm(data, desc="Generating")):
        cause = item['cause']
        effects = item['effects']
        
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")
        print(f"\n[{row_id+1}/{len(data)}] {cause}")
        
        # Generate base image
        base_seed = args.seed + row_id * 1000
        gen = torch.Generator(device=device).manual_seed(base_seed)
        
        base_image = txt2img(
            f"{cause}, high quality, detailed",
            height=args.img_size,
            width=args.img_size,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]
        
        frames = []
        frame_paths = []
        effect_list = []
        
        # Generate each effect variation
        current = base_image
        for effect_idx, effect in enumerate(effects):
            # Create instruction
            instruction = f"Transform the scene to show: {effect}"
            
            seed = base_seed + effect_idx + 1
            gen_t = torch.Generator(device=device).manual_seed(seed)
            
            edited = ip2p(
                instruction,
                image=current,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7.5,
                generator=gen_t,
            ).images[0]
            
            # Save
            path = sample_dir / f"{effect_idx:02d}_{effect.replace(' ', '_')[:30]}.png"
            edited.save(path)
            frames.append(edited)
            frame_paths.append(str(path))
            effect_list.append(effect)
            
            # Chain edits OR start fresh from base (change this for different behavior)
            # current = edited  # Chain: accumulates changes
            current = base_image  # Independent: each from base
            
            print(f"  [{effect_idx+1}/{args.num_effects}] {effect}")
        
        # Grid
        grid = make_grid(frames, pad=8)
        grid_path = sample_dir / "GRID.png"
        grid.save(grid_path)
        
        # Metadata
        meta = {
            "row_id": row_id,
            "cause": cause,
            "effects": effect_list,
            "frame_paths": frame_paths,
            "grid_path": str(grid_path),
            "method": args.out_method,
            "num_effects": args.num_effects,
            "seed_base": base_seed,
            "img_size": args.img_size,
            "note": "InstructPix2Pix baseline - each edit from base image"
        }
        
        with open(sample_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  ✓ Saved: {grid_path.name}")
    
    print("\n" + "="*60)
    print(f"✅ COMPLETED: {len(data)} samples")
    print(f"   Output: {out_root}")
    print("="*60)

if __name__ == "__main__":
    main()