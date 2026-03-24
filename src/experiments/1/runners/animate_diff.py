#!/usr/bin/env python3
"""
Generate temporal sequences using AnimateDiff baseline.
Uses test_prompts.txt to generate video sequences, then extracts frames.

AnimateDiff is a state-of-the-art video generation model that adds temporal
layers to Stable Diffusion for smooth motion generation.

Output structure:
    outputs/animatediff/<NUM_FRAMES>/<row_id>/
        00.png, 01.png, ..., grid.png, meta.json

Usage:
    python run_animatediff_baseline.py \
        --prompt_file data/test_prompts.txt \
        --num_frames 5 \
        --max_samples 190

Note: AnimateDiff generates smooth video motion but may not handle
      discrete state changes (like breaking, exploding) as well as
      causal models.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm

# Import after checking availability
try:
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_gif
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False
    print("Warning: AnimateDiff not available. Install with:")
    print("pip install diffusers[torch] accelerate")

# ---------- Utils ----------
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
        raise ValueError(f"Empty prompt file: {prompt_file}")
    return prompts

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

# ---------- AnimateDiff Generation ----------
@torch.no_grad()
def generate_animatediff_sequence(
    pipe: AnimateDiffPipeline,
    prompt: str,
    num_frames: int,
    seed: int,
    device: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> List[Image.Image]:
    """
    Generate video sequence using AnimateDiff.
    
    AnimateDiff generates 16 frames by default, we'll extract evenly-spaced
    frames to get the desired number.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # AnimateDiff generates 16 frames by default
    # We'll generate more and subsample to get desired count
    total_frames = max(16, num_frames * 2)  # Generate extra for better sampling
    
    # Generate video
    output = pipe(
        prompt=prompt,
        num_frames=total_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    # Extract frames from video
    frames = output.frames[0]  # List of PIL Images
    
    # Subsample to get desired number of frames
    if len(frames) > num_frames:
        # Evenly spaced indices
        indices = torch.linspace(0, len(frames) - 1, num_frames).long().tolist()
        frames = [frames[i] for i in indices]
    elif len(frames) < num_frames:
        # Not enough frames - repeat last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])
    
    return frames

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True,
                       help="Path to test_prompts.txt")
    parser.add_argument("--out_method", type=str, default="animatediff")
    parser.add_argument("--num_frames", type=int, default=5,
                       help="Number of frames to extract from video")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Generation parameters
    parser.add_argument("--base_model", type=str, 
                       default="runwayml/stable-diffusion-v1-5",
                       help="Base SD model for AnimateDiff")
    parser.add_argument("--motion_adapter", type=str,
                       default="guoyww/animatediff-motion-adapter-v1-5-2",
                       help="Motion adapter model")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if not ANIMATEDIFF_AVAILABLE:
        raise ImportError(
            "AnimateDiff not available. Install with:\n"
            "pip install diffusers[torch] accelerate transformers"
        )
    
    # Setup paths
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
    print("ANIMATEDIFF BASELINE GENERATION")
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
    
    # Load AnimateDiff pipeline
    print("\n[1/2] Loading Motion Adapter...")
    adapter = MotionAdapter.from_pretrained(
        args.motion_adapter,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    print("✓ Motion adapter loaded")
    
    print("\n[2/2] Loading AnimateDiff Pipeline...")
    pipe = AnimateDiffPipeline.from_pretrained(
        args.base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    
    # Use DDIM scheduler for better quality
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
    print("✓ AnimateDiff pipeline loaded")
    
    # Generate sequences
    print("\n" + "=" * 60)
    print("GENERATING SEQUENCES")
    print("=" * 60)
    
    for row_id, base_prompt in enumerate(tqdm(prompts, desc="Generating")):
        sample_dir = ensure_dir(out_root / f"{row_id:06d}")
        
        try:
            # Generate video frames
            frames = generate_animatediff_sequence(
                pipe=pipe,
                prompt=base_prompt,
                num_frames=args.num_frames,
                seed=args.seed + row_id * 1000,
                device=device,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )
            
            # Save individual frames
            frame_paths = []
            for t, img in enumerate(frames):
                path = sample_dir / f"{t:02d}.png"
                img.save(path)
                frame_paths.append(str(path))
            
            # Create and save grid
            grid = make_grid(frames, pad=8)
            grid_path = sample_dir / "grid.png"
            grid.save(grid_path)
            
            # Build prompts list (matches other baselines)
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
                "base_model": args.base_model,
                "motion_adapter": args.motion_adapter,
                "steps": args.steps,
                "guidance": args.guidance,
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