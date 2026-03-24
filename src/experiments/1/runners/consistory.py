#!/usr/bin/env python3
"""
Generate temporal sequences using ConsiStory baseline.
Uses test_prompts.txt to generate consistent image sequences.

ConsiStory is designed for consistent character generation across scenes,
not causal progression. This script adapts it for temporal sequence generation.

Output structure:
    outputs/consistory/<NUM_FRAMES>/<row_id>/
        00.png, 01.png, ..., meta.json

Usage:
    python run_consistory_baseline.py \
        --consistory_repo /path/to/NVlabs/consistory \
        --prompt_file data/test_prompts.txt \
        --num_frames 5 \
        --max_samples 190

Note: ConsiStory generates "consistent" scenes (same subject, different contexts)
      rather than temporal causality. This makes it a useful baseline showing
      consistency without causal progression.
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from PIL import Image

# Set GPU (can override with --gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------
# Utils
# -----------------------
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
    lines = [l.strip() for l in prompt_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"Empty prompt file: {prompt_file}")
    return lines

# Stopwords for concept token extraction
STOP_WORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "to", "from", 
    "of", "into", "over", "frame", "therefore", "consistent", "scene"
}

def extract_concept_token(text: str) -> str:
    """Extract main concept from prompt (last content word)"""
    words = re.findall(r"[A-Za-z]+", text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return words[-1] if words else "subject"

def create_settings_for_progression(num_frames: int) -> List[str]:
    """
    Create scene settings that could show temporal progression.
    
    ConsiStory uses these as different "contexts" for the subject.
    We adapt them to suggest temporal flow.
    """
    if num_frames == 5:
        return [
            "in the initial moment",
            "as the action begins",
            "during the main action",
            "as the action completes",
            "showing the final result",
        ]
    elif num_frames == 8:
        return [
            "at the starting point",
            "as motion begins",
            "showing early progression",
            "during active movement",
            "in mid-progression",
            "as completion approaches",
            "nearly finished",
            "showing the final state",
        ]
    else:
        # Generic fallback
        return [f"at stage {i+1} of {num_frames}" for i in range(num_frames)]

# -----------------------
# ConsiStory CLI wrapper
# -----------------------
def run_consistory(
    consistory_repo: Path,
    out_dir: Path,
    subject: str,
    concept_token: str,
    settings: List[str],
    gpu: int = 0,
    seed: int = 42,
) -> None:
    """
    Call ConsiStory CLI from NVlabs/consistory repo.
    
    Official usage:
        python consistory_CLI.py --subject "a cute dog" --concept_token "dog" 
               --settings "in park" "at beach" ... --out_dir output/
    """
    cli_script = consistory_repo / "consistory_CLI.py"
    if not cli_script.exists():
        raise FileNotFoundError(
            f"ConsiStory CLI not found: {cli_script}\n"
            f"Please clone: git clone https://github.com/NVlabs/consistory"
        )
    
    # Build command
    cmd = [
        "python", str(cli_script),
        "--run_type", "batch",
        "--gpu", str(gpu),
        "--seed", str(seed),
        "--subject", subject,
        "--concept_token", concept_token,
        "--out_dir", str(out_dir),
        "--settings"
    ] + settings
    
    # Run ConsiStory
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ConsiStory error: {e.stderr}")
        raise

def collect_generated_images(directory: Path) -> List[Path]:
    """Collect generated images from ConsiStory output directory"""
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return sorted(images, key=lambda x: x.name)

def make_grid(images: List[Image.Image], pad: int = 8) -> Image.Image:
    """Create horizontal grid of images"""
    if not images:
        raise ValueError("No images to grid")
    
    widths, heights = zip(*[img.size for img in images])
    total_width = sum(widths) + pad * (len(images) - 1)
    max_height = max(heights)
    
    grid = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.size[1]) // 2
        grid.paste(img, (x_offset, y_offset))
        x_offset += img.size[0] + pad
    
    return grid

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistory_repo", type=str, required=True,
                       help="Path to NVlabs/consistory repository")
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, required=True,
                       help="Path to test_prompts.txt")
    
    parser.add_argument("--out_method", type=str, default="consistory")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup paths
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root(Path(__file__))
    
    prompt_file = Path(args.prompt_file)
    if not prompt_file.is_absolute():
        prompt_file = (project_root / prompt_file).resolve()
    
    consistory_repo = Path(args.consistory_repo).resolve()
    
    out_root = ensure_dir(project_root / "outputs" / args.out_method / str(args.num_frames))
    tmp_root = ensure_dir(project_root / "outputs" / "_tmp_consistory")
    
    print("=" * 60)
    print("CONSISTORY BASELINE GENERATION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Prompt file: {prompt_file}")
    print(f"ConsiStory repo: {consistory_repo}")
    print(f"Output dir: {out_root}")
    print(f"Num frames: {args.num_frames}")
    print(f"GPU: {args.gpu}")
    print("=" * 60)
    
    # Load prompts
    prompts = load_prompts(prompt_file)
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Generate settings for temporal progression
    settings = create_settings_for_progression(args.num_frames)
    print(f"✓ Using {len(settings)} temporal settings")
    
    # Generate sequences
    print("\n" + "=" * 60)
    print("GENERATING SEQUENCES")
    print("=" * 60)
    
    from tqdm import tqdm
    
    for row_id, subject in enumerate(tqdm(prompts, desc="Generating")):
        # Create temporary directory for this sample
        tmp_out = ensure_dir(tmp_root / f"sample_{row_id:06d}")
        
        # Clean tmp directory
        for f in tmp_out.glob("*"):
            if f.is_file():
                f.unlink()
        
        # Extract concept token
        concept_token = extract_concept_token(subject)
        
        try:
            # Run ConsiStory
            run_consistory(
                consistory_repo=consistory_repo,
                out_dir=tmp_out,
                subject=subject,
                concept_token=concept_token,
                settings=settings,
                gpu=args.gpu,
                seed=args.seed + row_id,
            )
            
            # Collect generated images
            generated_images = collect_generated_images(tmp_out)
            
            if len(generated_images) != args.num_frames:
                print(f"\nWarning: Expected {args.num_frames} images, got {len(generated_images)}")
                if len(generated_images) < args.num_frames:
                    continue  # Skip this sample
                generated_images = generated_images[:args.num_frames]
            
            # Create output directory
            sample_dir = ensure_dir(out_root / f"{row_id:06d}")
            
            # Save frames with standard naming
            frame_paths = []
            pil_images = []
            for i, img_path in enumerate(generated_images):
                img = Image.open(img_path).convert("RGB")
                out_path = sample_dir / f"{i:02d}.png"
                img.save(out_path)
                frame_paths.append(str(out_path))
                pil_images.append(img)
            
            # Create grid
            grid = make_grid(pil_images)
            grid_path = sample_dir / "grid.png"
            grid.save(grid_path)
            
            # Create prompts list (matches other baselines)
            frame_prompts = [f"{subject} {setting}" for setting in settings]
            
            # Save metadata
            meta = {
                "row_id": row_id,
                "base_prompt": subject,
                "prompts": frame_prompts,
                "frame_paths": frame_paths,
                "grid_path": str(grid_path),
                "method": args.out_method,
                "num_frames": args.num_frames,
                "concept_token": concept_token,
                "settings": settings,
                "seed": args.seed + row_id,
            }
            
            with open(sample_dir / "meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
        
        except Exception as e:
            print(f"\nError on sample {row_id} ({subject[:50]}...): {e}")
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