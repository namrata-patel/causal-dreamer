"""
Vanilla SDXL Baseline for Counterfactual Generation
Generates images using manually specified effects from CSV
(Structure + metadata matches your vanilla_sd_counterfactual baseline)

Notes for SDXL:
- SDXL is best at 1024x1024. You *can* run 512, but quality drops.
- This is pure text2img (each effect independently) to match your SD baseline.
"""

import os
import csv
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ---------------- PATHS ----------------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root")

PROJECT_ROOT = find_project_root(Path(__file__))
DATA_ROOT = PROJECT_ROOT / "data"
# CSV_FILE = DATA_ROOT / "test_counterfactual.csv"
CSV_FILE = DATA_ROOT / "test_cf2.csv"

# Output structure matches CausalDreamer
NUM_EFFECTS = 8
OUT_ROOT = PROJECT_ROOT / "outputs" / "paper_sdxl" / f"{NUM_EFFECTS}"

# ---------------- SDXL CONFIG ----------------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# SDXL recommended: 1024. (512 works but worse.)
IMG_SIZE = 1024

STEPS = 30
GUIDANCE = 7.5
BASE_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional (usually helps reduce artifacts)
NEG_PROMPT = "low quality, blurry, jpeg artifacts, deformed, distorted, watermark, text"

# Debug: set to small number for testing
MAX_SAMPLES = None
# -------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_cause_effects(csv_file: Path):
    """Load cause-effect sets from CSV: cause + effect1..effect8"""
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cause = row["cause"]
            effects = [row[f"effect{i}"] for i in range(1, 9)]
            data.append({"cause": cause, "effects": effects})
    return data

def make_grid_row(pil_images, pad=8, bg=(255, 255, 255)):
    if len(pil_images) == 0:
        raise ValueError("No images provided for grid.")
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

def sanitize_filename(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().replace(" ", "_")
    # keep it filesystem-friendly
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:max_len] if len(out) > max_len else out

def main():
    print("=" * 60)
    print("VANILLA SDXL COUNTERFACTUAL BASELINE")
    print("=" * 60)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CSV_FILE:", CSV_FILE)
    print("OUT_ROOT:", OUT_ROOT)
    print("DEVICE:", DEVICE)
    print()

    ensure_dir(OUT_ROOT)

    print("Loading cause-effect data from CSV...")
    data = load_cause_effects(CSV_FILE)

    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]

    print(f"Loaded {len(data)} cause-effect sets")
    print()

    print("Loading SDXL...")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(DEVICE)

    # memory/perf tweaks
    pipe.enable_attention_slicing()
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # helps reduce random nondeterminism
    pipe.set_progress_bar_config(disable=True)
    print()

    for row_id, item in enumerate(data):
        cause = item["cause"]
        effects = item["effects"]

        out_dir = ensure_dir(OUT_ROOT / f"{row_id:06d}")

        print(f"[{row_id+1}/{len(data)}] Generating for: {cause}")

        frame_paths = []
        frames_pil = []
        effect_list = []

        for effect_idx, effect in enumerate(effects):
            prompt = f"Consistent scene: {cause}. Therefore, {effect}"

            seed = BASE_SEED + row_id * 1000 + effect_idx
            gen = torch.Generator(device=DEVICE).manual_seed(seed)

            # SDXL call
            img = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                height=IMG_SIZE,
                width=IMG_SIZE,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).images[0]

            img_name = f"{effect_idx:02d}_{sanitize_filename(effect, max_len=40)}.png"
            img_path = out_dir / img_name
            img.save(img_path)

            frame_paths.append(str(img_path))
            frames_pil.append(img)
            effect_list.append(effect)

            print(f"  [{effect_idx+1}/{NUM_EFFECTS}] {effect}")

        # Grid (single row)
        grid = make_grid_row(frames_pil, pad=8)
        grid_path = out_dir / "GRID.png"
        grid.save(grid_path)

        meta = {
            "row_id": row_id,
            "cause": cause,
            "effects": effect_list,
            "frame_paths": frame_paths,
            "grid_path": str(grid_path),
            "method": "vanilla_sdxl_manual_effects",
            "sd_model": MODEL_ID,
            "num_effects": NUM_EFFECTS,
            "seed_base": BASE_SEED,
            "img_size": IMG_SIZE,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "negative_prompt": NEG_PROMPT,
            "note": "Uses manually specified effects from CSV (ground truth comparison)",
        }

        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved grid: {grid_path.name}")
        print()

    print("=" * 60)
    print(f"✅ COMPLETED: Generated {len(data)} cause-effect sets")
    print(f"   Output: {OUT_ROOT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
