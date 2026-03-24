#!/usr/bin/env python3
"""
Generate temporal sequences using InstructPix2Pix baseline.
Uses same test_prompts.txt as ControlNet baseline.

Output structure matches evaluation script expectations:
    outputs/instructpix2pix/<NUM_FRAMES>/<row_id>/
        00.png, 01.png, ..., meta.json

Usage:
    python run_instructpix2pix_baseline.py \
        --prompt_file data/test_prompts.txt \
        --num_frames 5 \
        --max_samples 190
"""

import os
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

# ---------- Grid generation ----------
def make_grid(images: List[Image.Image], pad: int = 8, bg_color=(255, 255, 255)) -> Image.Image:
    """Create horizontal grid from list of PIL images"""
    if not images:
        raise ValueError("No images to create grid")
    
    widths, heights = zip(*[img.size for img in images])
    total_width = sum(widths) + pad * (len(images) - 1)
    max_height = max(heights)
    
    grid = Image.new("RGB", (total_width, max_height), bg_color)
    
    x_offset = 0
    for img in images:
        # Center vertically
        y_offset = (max_height - img.size[1]) // 2
        grid.paste(img, (x_offset, y_offset))
        x_offset += img.size[0] + pad
    
    return grid

# ---------- Project root ----------
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

def load_prompts(prompt_file: Path) -> List[str]:
    """Load prompts from test_prompts.txt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file empty: {prompt_file}")
    return prompts

# ---------- Sequential editing ----------
@torch.no_grad()
def generate_sequence(
    txt2img_pipe: StableDiffusionPipeline,
    ip2p_pipe: StableDiffusionInstructPix2PixPipeline,
    base_prompt: str,
    num_frames: int,
    seed: int,
    device: str,
    img_size: int = 512,
    strength: float = 0.55,
) -> List[Image.Image]:
    """
    Generate sequence using InstructPix2Pix sequential editing.
    
    Frame 0: Generate from base_prompt using txt2img
    Frame 1-N: Apply generic progression instructions
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Frame 0: Initial image
    initial = txt2img_pipe(
        base_prompt,
        height=img_size,
        width=img_size,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    
    images = [initial]
    
    # Generic progression instructions
    if num_frames == 5:
        instructions = [
            "continue the action showing early progression",
            "show the action further progressed",
            "show the action nearly complete",
            "show the final result and aftermath",
        ]
    elif num_frames == 8:
        instructions = [
            "start showing the action beginning",
            "continue the action showing early progression",
            "show the action progressing further",
            "show the action halfway through",
            "show the action nearly complete",
            "show the action completing",
            "show the final result and aftermath",
        ]
    else:
        # Generic fallback
        instructions = [
            f"continue progressing the action (step {i}/{num_frames-1})"
            for i in range(1, num_frames)
        ]
    
    # Sequential editing
    current = initial
    for t, instruction in enumerate(instructions, start=1):
        gen_t = torch.Generator(device=device).manual_seed(seed + t)
        
        edited = ip2p_pipe(
            instruction,
            image=current,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=gen_t,
        ).images[0]
        
        images.append(edited)
        current = edited  # Chain for next edit
    
    return images

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True, 
                       help="Path to test_prompts.txt")
    parser.add_argument("--out_method", type=str, default="instructpix2pix")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.55,
                       help="img2img strength for InstructPix2Pix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # Setup
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root(Path(__file__))
    
    prompt_file = Path(args.prompt_file)
    if not prompt_file.is_absolute():
        prompt_file = (project_root / prompt_file).resolve()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = ensure_dir(project_root / "outputs" / args.out_method / str(args.num_frames))
    
    print("=" * 60)
    print("INSTRUCTPIX2PIX BASELINE GENERATION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Prompt file: {prompt_file}")
    print(f"Output dir: {out_root}")
    print(f"Num frames: {args.num_frames}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Load models
    print("\n[1/2] Loading Stable Diffusion (txt2img)...")
    txt2img_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    txt2img_pipe.enable_attention_slicing()
    print("✓ SD loaded")
    
    print("\n[2/2] Loading InstructPix2Pix...")
    ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    ip2p_pipe.enable_attention_slicing()
    print("✓ InstructPix2Pix loaded")
    
    # Generate sequences
    print("\n" + "=" * 60)
    print("GENERATING SEQUENCES")
    print("=" * 60)
    
    for row_id, base_prompt in enumerate(tqdm(prompts, desc="Generating")):
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")
        
        try:
            # Generate sequence
            images = generate_sequence(
                txt2img_pipe=txt2img_pipe,
                ip2p_pipe=ip2p_pipe,
                base_prompt=base_prompt,
                num_frames=args.num_frames,
                seed=args.seed + row_id * 1000,
                device=device,
                img_size=args.img_size,
                strength=args.strength,
            )
            
            # Save frames
            frame_paths = []
            pil_images = []
            for t, img in enumerate(images):
                path = sample_dir / f"{t:02d}.png"
                img.save(path)
                frame_paths.append(str(path))
                pil_images.append(img)
            
            # Create and save grid
            grid = make_grid(pil_images, pad=8)
            grid_path = sample_dir / "grid.png"
            grid.save(grid_path)
            
            # Build prompts list (matches ControlNet format)
            frame_prompts = [base_prompt] + [
                f"{base_prompt} Frame {t}/{args.num_frames}."
                for t in range(2, args.num_frames + 1)
            ]
            
            # Save metadata
            meta = {
                "row_id": row_id,
                "base_prompt": base_prompt,
                "prompts": frame_prompts,
                "frame_paths": frame_paths,
                "grid_path": str(grid_path),
                "method": args.out_method,
                "num_frames": args.num_frames,
                "seed": args.seed + row_id * 1000,
                "img_size": args.img_size,
                "strength": args.strength,
            }
            
            with open(sample_dir / "meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
        
        except Exception as e:
            print(f"\nError on sample {row_id} ({base_prompt[:50]}...): {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output: {out_root}")
    print(f"Generated: {len(list(out_root.iterdir()))} samples")
    print("\nRun evaluation:")
    print(f"python eval_causal_metrics.py --methods {args.out_method} --num_frames {args.num_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()