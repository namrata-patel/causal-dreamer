#!/usr/bin/env python3
"""
AnimateDiff baseline for *counterfactual* generation (CSV-driven).

This matches your SD/SDXL counterfactual baselines:
- For each CSV row: one fixed CAUSE + NUM_EFFECTS effects (effect1..effect8)
- For each effect: generate a short AnimateDiff video (total_frames), then
  subsample to NUM_FRAMES frames
- Save outputs to:
    outputs/animatediff_counterfactual/<NUM_FRAMES>/<row_id>/<effect_idx>/
        00.png, 01.png, ..., grid.png, meta.json

Usage:
  python run_animatediff_counterfactual_csv.py \
    --csv_file data/test_counterfactual.csv \
    --num_frames 5 \
    --num_effects 8 \
    --max_samples 190

Notes:
- AnimateDiff is SD1.5-based; keep base_model to v1-5 for fairness.
- Each (cause,effect) branch is independent (not temporal reasoning),
  but AnimateDiff provides smooth temporal motion within each branch.
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm

try:
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False


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

def load_cause_effects(csv_file: Path, num_effects: int = 8) -> List[Dict]:
    """Load rows: cause + effect1..effectN"""
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cause = (row.get("cause") or "").strip()
            effects = []
            for i in range(1, num_effects + 1):
                effects.append((row.get(f"effect{i}") or "").strip())
            rows.append({"cause": cause, "effects": effects})
    if not rows:
        raise ValueError(f"CSV empty or unreadable: {csv_file}")
    return rows

def make_grid(images: List[Image.Image], pad: int = 8, bg_color=(255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("No images to create grid")
    widths, heights = zip(*[img.size for img in images])
    total_width = sum(widths) + pad * (len(images) - 1)
    max_height = max(heights)
    grid = Image.new("RGB", (total_width, max_height), bg_color)
    x = 0
    for img in images:
        y = (max_height - img.size[1]) // 2
        grid.paste(img, (x, y))
        x += img.size[0] + pad
    return grid

def sanitize_filename(s: str, max_len: int = 64) -> str:
    s = (s or "").strip().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:max_len] if len(out) > max_len else out


# ---------- AnimateDiff generation ----------
@torch.no_grad()
def generate_animatediff_sequence(
    pipe: AnimateDiffPipeline,
    prompt: str,
    num_frames: int,
    seed: int,
    device: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    total_frames: int = 16,
    negative_prompt: Optional[str] = None,
) -> List[Image.Image]:
    """
    Generate video frames with AnimateDiff and return exactly num_frames
    via even subsampling.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=total_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g,
    )

    frames = out.frames[0]  # list[PIL]
    if len(frames) == 0:
        raise RuntimeError("AnimateDiff returned 0 frames")

    # subsample to desired num_frames
    if len(frames) > num_frames:
        idx = torch.linspace(0, len(frames) - 1, num_frames).long().tolist()
        frames = [frames[i] for i in idx]
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--csv_file", type=str, required=True)
    ap.add_argument("--out_method", type=str, default="animatediff_counterfactual")
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--num_effects", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None)

    # AnimateDiff params
    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--motion_adapter", type=str, default="guoyww/animatediff-motion-adapter-v1-5-2")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)

    # video sampling
    ap.add_argument("--total_frames", type=int, default=16, help="Frames generated by AnimateDiff before subsampling")
    ap.add_argument("--negative_prompt", type=str, default="low quality, blurry, jpeg artifacts, deformed, distorted, watermark, text")

    args = ap.parse_args()

    if not ANIMATEDIFF_AVAILABLE:
        raise ImportError(
            "AnimateDiff not available. Install with:\n"
            "pip install diffusers[torch] accelerate transformers"
        )

    # roots
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root(Path(__file__))

    csv_file = Path(args.csv_file)
    if not csv_file.is_absolute():
        csv_file = (project_root / csv_file).resolve()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_root = ensure_dir(project_root / "outputs" / args.out_method / str(args.num_frames))

    print("=" * 60)
    print("ANIMATEDIFF COUNTERFACTUAL (CSV) BASELINE")
    print("=" * 60)
    print("Project root:", project_root)
    print("CSV file:", csv_file)
    print("Output dir:", out_root)
    print("Device:", device)
    print("Num frames:", args.num_frames)
    print("Num effects:", args.num_effects)
    print("Total frames (generated):", args.total_frames)
    print("=" * 60)

    # load CSV
    data = load_cause_effects(csv_file, num_effects=args.num_effects)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    print(f"✓ Loaded {len(data)} cause-effect sets")

    # load pipeline
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print("\n[1/2] Loading Motion Adapter...")
    adapter = MotionAdapter.from_pretrained(args.motion_adapter, torch_dtype=dtype)
    print("✓ Motion adapter loaded")

    print("\n[2/2] Loading AnimateDiff Pipeline...")
    pipe = AnimateDiffPipeline.from_pretrained(
        args.base_model,
        motion_adapter=adapter,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # memory / speed
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    # Avoid forcing cpu offload if you're already on GPU and want speed.
    # If you hit OOM, enable it.
    # pipe.enable_model_cpu_offload()

    print("✓ AnimateDiff pipeline loaded\n")

    # loop
    for row_id, item in enumerate(tqdm(data, desc="Generating CSV counterfactuals")):
        cause = item["cause"]
        effects = item["effects"]

        row_dir = ensure_dir(out_root / f"{row_id:06d}")

        for effect_idx, effect in enumerate(effects):
            if not effect:
                continue

            branch_dir = ensure_dir(row_dir / f"{effect_idx:02d}_{sanitize_filename(effect, 40)}")

            # Match your counterfactual prompt template
            prompt = f"Consistent scene: {cause}. Therefore, {effect}"

            seed = args.seed + row_id * 1000 + effect_idx

            try:
                frames = generate_animatediff_sequence(
                    pipe=pipe,
                    prompt=prompt,
                    num_frames=args.num_frames,
                    seed=seed,
                    device=device,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    total_frames=max(2, args.total_frames),
                    negative_prompt=args.negative_prompt,
                )
            except Exception as e:
                print(f"\n[ERROR] row={row_id} effect={effect_idx} :: {e}")
                continue

            # save frames
            frame_paths = []
            for t, img in enumerate(frames):
                fp = branch_dir / f"{t:02d}.png"
                img.save(fp)
                frame_paths.append(str(fp))

            # grid
            grid = make_grid(frames, pad=8)
            grid_path = branch_dir / "grid.png"
            grid.save(grid_path)

            meta = {
                "row_id": row_id,
                "effect_idx": effect_idx,
                "cause": cause,
                "effect": effect,
                "prompt": prompt,
                "frame_paths": frame_paths,
                "grid_path": str(grid_path),
                "method": args.out_method,
                "num_frames": args.num_frames,
                "seed": seed,
                "base_model": args.base_model,
                "motion_adapter": args.motion_adapter,
                "steps": args.steps,
                "guidance": args.guidance,
                "total_frames_generated": args.total_frames,
                "negative_prompt": args.negative_prompt,
                "note": "AnimateDiff baseline: smooth temporal motion within each counterfactual branch; no causal intervention.",
            }
            with open(branch_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ GENERATION COMPLETE")
    print("=" * 60)
    print("Output:", out_root)
    print("=" * 60)
    print("Example evaluation call:")
    print(f"python eval_causal_metrics.py --methods {args.out_method} --num_frames {args.num_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()
