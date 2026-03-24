import os
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image

# ---------------- PATHS (ROBUST) ----------------
def find_project_root(start: Path) -> Path:
    """
    Find project root by walking up until we find data/test_prompts.txt
    """
    p = start.resolve()
    for _ in range(15):
        if (p / "data" / "test_prompts.txt").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root containing data/test_prompts.txt")

PROJECT_ROOT = find_project_root(Path(__file__))
print(PROJECT_ROOT)

DATA_ROOT = PROJECT_ROOT / "data"
PROMPT_FILE = DATA_ROOT / "test_prompts.txt"   # one prompt per line

# ✅ Match CausalDreamer-style structure: outputs/<method>/<num_frames>/<row_id>/
NUM_FRAMES = 5
OUT_ROOT = PROJECT_ROOT / "outputs" / "vanilla_sd2" / f"{NUM_FRAMES}"

# ---------------- SD CONFIG ----------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512
STEPS = 30
GUIDANCE = 7.5
BASE_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug: set int (e.g., 5) for quick test; None for full run
MAX_SAMPLES = None
# -------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_prompt_list(prompt_file: Path):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompts

def build_prompts(base_prompt: str):
    # Keep your same behavior (frame annotations)
    prompts = [base_prompt]
    for t in range(2, NUM_FRAMES + 1):
        prompts.append(f"{base_prompt} Frame {t}/{NUM_FRAMES}.")
    return prompts

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

def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("PROMPT_FILE:", PROMPT_FILE)
    print("OUT_ROOT:", OUT_ROOT)
    print("DEVICE:", DEVICE)

    ensure_dir(OUT_ROOT)

    print("Loading prompt file...")
    prompts_list = load_prompt_list(PROMPT_FILE)

    if MAX_SAMPLES is not None:
        prompts_list = prompts_list[:MAX_SAMPLES]

    print(f"Loaded {len(prompts_list)} prompts.")

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    pipe.enable_attention_slicing()

    for row_id, base_prompt in enumerate(prompts_list):
        # Match CausalDreamer folder naming: 000000, 000001, ...
        out_dir = ensure_dir(OUT_ROOT / f"{row_id:06d}")
        print("saving output in ", out_dir)

        frame_prompts = build_prompts(base_prompt)

        frame_paths = []
        frames_pil = []

        for t, prompt in enumerate(frame_prompts):
            seed = BASE_SEED + row_id * 1000 + t
            gen = torch.Generator(device=DEVICE).manual_seed(seed)

            img = pipe(
                prompt=prompt,
                height=IMG_SIZE,
                width=IMG_SIZE,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).images[0]

            img_path = out_dir / f"{t:02d}.png"
            img.save(img_path)
            frame_paths.append(str(img_path))
            frames_pil.append(img)

        # grid.png (same as you asked)
        grid = make_grid_row(frames_pil, pad=8)
        grid_path = out_dir / "grid.png"
        grid.save(grid_path)

        meta = {
            "row_id": row_id,
            "base_prompt": base_prompt,
            "prompts": frame_prompts,
            "frame_paths": frame_paths,
            "grid_path": str(grid_path),
            "method": "vanilla_sd",
            "sd_model": MODEL_ID,
            "num_frames": NUM_FRAMES,
            "seed_rule": f"seed = {BASE_SEED} + row_id*1000 + t",
            "img_size": IMG_SIZE,
            "steps": STEPS,
            "guidance": GUIDANCE,
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if (row_id + 1) % 10 == 0:
            print(f"[Vanilla SD] Generated {row_id+1}/{len(prompts_list)}")

    print("✅ Vanilla SD generation completed.")

if __name__ == "__main__":
    main()
