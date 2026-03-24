import os
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------- Robust project root ----------
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "outputs").exists() and (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root (need outputs/ and data/)")

# ---------- OpenCLIP (avoids transformers torch>=2.6 restriction) ----------
def load_openclip(device: str = "cuda"):
    import open_clip
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_images(model, preprocess, image_paths, device):
    feats = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # [N, D]

@torch.no_grad()
def encode_images_pil(model, preprocess, pil_images, device):
    """Encode from PIL images directly"""
    feats = []
    for img in pil_images:
        x = preprocess(img).unsqueeze(0).to(device)
        f = model.encode_image(x)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # [N, D]

@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32):
    feats = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        t = tokenizer(chunk).to(device)
        f = model.encode_text(t)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)  # [N, D]

# ---------- DINOv2 ----------
def load_dinov2(device="cuda", model_name="dinov2_vits14"):
    """
    model_name options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    vits14 is fastest.
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model

def dinov2_preprocess(pil_img, size=518):
    """DINOv2 expects ImageNet-style normalization."""
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pil_img = pil_img.resize((nw, nh), Image.BICUBIC)

    left = (nw - size) // 2
    top = (nh - size) // 2
    pil_img = pil_img.crop((left, top, left + size, top + size))

    x = torch.from_numpy(np.array(pil_img)).float() / 255.0  # [H,W,3]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

@torch.no_grad()
def dinov2_embed(model, pil_img, device="cuda"):
    x = dinov2_preprocess(pil_img).to(device)
    feat = model(x)  # [1, D]
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0)  # [D]

@torch.no_grad()
def compute_dino_scores_for_paths(dino_model, image_paths, device="cuda"):
    """
    image_paths: list[str|Path] in temporal order
    Returns:
      DINO_TCS: mean cosine sim consecutive frames
      DINO_IDCons: mean cosine sim between first and others
    """
    if len(image_paths) < 2:
        return float("nan"), float("nan")

    feats = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        feats.append(dinov2_embed(dino_model, im, device=device))

    if len(feats) < 2:
        return float("nan"), float("nan")

    feats = torch.stack(feats, dim=0)  # [T, D] normalized

    dino_tcs = (feats[:-1] * feats[1:]).sum(dim=-1).mean().item()
    dino_id  = (feats[1:] * feats[0:1]).sum(dim=-1).mean().item()
    return float(dino_tcs), float(dino_id)

# ---------- NEW CAUSAL METRICS ----------

@torch.no_grad()
def compute_causal_progression_score(
    model,
    tokenizer,
    device: str,
    image_feats: torch.Tensor,  # [N, D] normalized
    cause_text: str,
    effect_text: str,
) -> Dict[str, float]:
    """
    Causal Progression Score (CPS):
    Measures if sequence progresses monotonically from cause to effect.
    
    Returns:
        CPS: Total progression (effect_similarity_last - cause_similarity_first)
        CPS_monotonic: 1.0 if progression is monotonic, else 0.0
        CPS_avg_progress: Average frame-to-frame progression
    """
    if len(image_feats) < 2 or not cause_text.strip() or not effect_text.strip():
        return {
            "CPS": float("nan"),
            "CPS_monotonic": float("nan"),
            "CPS_avg_progress": float("nan"),
        }
    
    # Encode cause and effect
    cause_emb = encode_texts(model, tokenizer, [cause_text], device)  # [1, D]
    effect_emb = encode_texts(model, tokenizer, [effect_text], device)  # [1, D]
    
    # Compute progression scores
    cause_sims = (image_feats * cause_emb).sum(dim=-1)  # [N]
    effect_sims = (image_feats * effect_emb).sum(dim=-1)  # [N]
    
    # Progression = moving away from cause, toward effect
    progression = effect_sims - cause_sims
    
    # Total progression (first to last)
    cps_total = (progression[-1] - progression[0]).item()
    
    # Check monotonicity
    is_monotonic = all(
        progression[i] <= progression[i+1] 
        for i in range(len(progression)-1)
    )
    cps_monotonic = 1.0 if is_monotonic else 0.0
    
    # Average frame-to-frame progress
    frame_progress = [(progression[i+1] - progression[i]).item() for i in range(len(progression)-1)]
    cps_avg = np.mean(frame_progress) if frame_progress else 0.0
    
    return {
        "CPS": float(cps_total),
        "CPS_monotonic": float(cps_monotonic),
        "CPS_avg_progress": float(cps_avg),
    }


@torch.no_grad()
def compute_event_transition_coherence(
    image_feats: torch.Tensor,  # [N, D] normalized
    optimal_similarity_range: Tuple[float, float] = (0.6, 0.85),
) -> Dict[str, float]:
    """
    Event Transition Coherence (ETC):
    Measures if frame-to-frame transitions are realistic.
    
    Optimal transitions should be:
    - Not too similar (static/boring): > 0.85
    - Not too different (jarring/unrealistic): < 0.6
    - Just right: 0.6-0.85
    
    Returns:
        ETC: Average coherence score
        ETC_min: Minimum coherence (worst transition)
    """
    if len(image_feats) < 2:
        return {
            "ETC": float("nan"),
            "ETC_min": float("nan"),
        }
    
    coherence_scores = []
    min_range, max_range = optimal_similarity_range
    
    for i in range(len(image_feats) - 1):
        similarity = (image_feats[i] * image_feats[i+1]).sum().item()
        
        if min_range <= similarity <= max_range:
            # Optimal range
            coherence = 1.0
        elif similarity > max_range:
            # Too similar (static)
            coherence = 0.5
        else:
            # Too different (scale linearly from 0 to 1)
            coherence = similarity / min_range
        
        coherence_scores.append(coherence)
    
    return {
        "ETC": float(np.mean(coherence_scores)),
        "ETC_min": float(np.min(coherence_scores)),
    }


@torch.no_grad()
def compute_semantic_diversity_score(
    image_feats: torch.Tensor,  # [N, D] normalized
) -> Dict[str, float]:
    """
    Semantic Diversity Score (SDS):
    Measures diversity across frames (useful for counterfactual mode).
    
    Higher diversity = better exploration of possibilities
    
    Returns:
        SDS_mean: Average pairwise diversity
        SDS_min: Minimum pairwise diversity
        SDS_max: Maximum pairwise diversity
    """
    if len(image_feats) < 2:
        return {
            "SDS_mean": float("nan"),
            "SDS_min": float("nan"),
            "SDS_max": float("nan"),
        }
    
    # Compute pairwise diversity (1 - similarity)
    diversities = []
    for i in range(len(image_feats)):
        for j in range(i+1, len(image_feats)):
            similarity = (image_feats[i] * image_feats[j]).sum().item()
            diversity = 1.0 - similarity
            diversities.append(diversity)
    
    if not diversities:
        return {
            "SDS_mean": float("nan"),
            "SDS_min": float("nan"),
            "SDS_max": float("nan"),
        }
    
    return {
        "SDS_mean": float(np.mean(diversities)),
        "SDS_min": float(np.min(diversities)),
        "SDS_max": float(np.max(diversities)),
    }


# ---------- misc ----------
def safe_read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def collect_samples(method_root: Path):
    """
    Expects:
      outputs/causal_dreamer/<NUM_FRAMES>/<sample_id>/generated_effects.json
    """
    samples = []
    for sample_dir in sorted([d for d in method_root.iterdir() if d.is_dir()]):
        jf = sample_dir / "generated_effects.json"
        if jf.exists():
            samples.append((sample_dir.name, jf))
    return samples

def extract_cause_and_effect(data: dict) -> Tuple[str, str]:
    """Extract cause and effect texts from generated_effects.json"""
    base_prompt = data.get("prompt", "")
    
    # Try to extract cause from base_prompt
    cause = base_prompt
    if "Therefore," in base_prompt:
        cause = base_prompt.split("Therefore,")[0].strip()
    
    # Try to extract effect from rows
    rows = data.get("rows", [])
    if rows:
        # Use last row's prompt as effect
        last_prompt = rows[-1].get("image_prompt", "")
        if "Therefore," in last_prompt:
            effect = last_prompt.split("Therefore,")[1].strip()
            # Remove "Frame X/Y" if present
            if "Frame" in effect:
                effect = effect.split("Frame")[0].strip().rstrip(".")
        else:
            effect = last_prompt
    else:
        effect = ""
    
    return cause, effect

def compute_metrics_for_sample(
    model, preprocess, tokenizer,
    dino_model,
    device,
    sample_id,
    jf_path: Path,
    compute_causal_metrics: bool = True,
):
    data = safe_read_json(jf_path)
    if not data or "rows" not in data:
        return None

    base_prompt = data.get("prompt", None)
    rows = data["rows"]

    # image paths & prompts
    image_paths = []
    image_prompts = []
    for r in rows:
        ip = r.get("image_file", None)
        txt = r.get("image_prompt", None)
        if ip and Path(ip).exists():
            image_paths.append(str(ip))
            image_prompts.append(txt if txt else "")

    N = len(image_paths)
    if N < 2:
        result = {
            "method": "causal_dreamer",
            "sample_id": sample_id,
            "N": N,
            "TCS": float("nan"),
            "CCS": float("nan"),
            "ID_Cons": float("nan"),
            "CLIPScore": float("nan"),
            "DINO_TCS": float("nan"),
            "DINO_IDCons": float("nan"),
            "base_prompt": base_prompt,
            "json_path": str(jf_path),
        }
        if compute_causal_metrics:
            result.update({
                "CPS": float("nan"),
                "CPS_monotonic": float("nan"),
                "CPS_avg_progress": float("nan"),
                "ETC": float("nan"),
                "ETC_min": float("nan"),
                "SDS_mean": float("nan"),
                "SDS_min": float("nan"),
                "SDS_max": float("nan"),
            })
        return result

    # ---- OpenCLIP encodings ----
    img_feat = encode_images(model, preprocess, image_paths, device)              # [N, D]
    txt_feat_imgprompt = encode_texts(model, tokenizer, image_prompts, device)   # [N, D]

    # TCS: consecutive similarity
    tcs = (img_feat[:-1] * img_feat[1:]).sum(dim=-1).mean().item()

    # ID Consistency: similarity to first frame
    id_cons = (img_feat[1:] * img_feat[0:1]).sum(dim=-1).mean().item()

    # CCS: image vs its image_prompt (your definition)
    ccs = (img_feat * txt_feat_imgprompt).sum(dim=-1).mean().item()

    # CLIPScore: image vs base prompt (same prompt for all images)
    if base_prompt and isinstance(base_prompt, str) and base_prompt.strip():
        txt_base = encode_texts(model, tokenizer, [base_prompt], device)  # [1, D]
        clipscore = (img_feat * txt_base).sum(dim=-1).mean().item()
    else:
        clipscore = float("nan")

    # ---- DINO scores ----
    dino_tcs, dino_id = compute_dino_scores_for_paths(dino_model, image_paths, device=device)

    result = {
        "method": "causal_dreamer",
        "sample_id": sample_id,
        "N": N,
        "TCS": float(tcs),
        "CCS": float(ccs),
        "ID_Cons": float(id_cons),
        "CLIPScore": float(clipscore),
        "DINO_TCS": float(dino_tcs),
        "DINO_IDCons": float(dino_id),
        "base_prompt": base_prompt,
        "json_path": str(jf_path),
    }

    # ---- NEW CAUSAL METRICS ----
    if compute_causal_metrics:
        # Extract cause and effect
        cause_text, effect_text = extract_cause_and_effect(data)
        
        # CPS: Causal Progression Score
        cps_metrics = compute_causal_progression_score(
            model=model,
            tokenizer=tokenizer,
            device=device,
            image_feats=img_feat,
            cause_text=cause_text,
            effect_text=effect_text,
        )
        result.update(cps_metrics)
        result["cause_text"] = cause_text
        result["effect_text"] = effect_text
        
        # ETC: Event Transition Coherence
        etc_metrics = compute_event_transition_coherence(image_feats=img_feat)
        result.update(etc_metrics)
        
        # SDS: Semantic Diversity Score
        sds_metrics = compute_semantic_diversity_score(image_feats=img_feat)
        result.update(sds_metrics)

    return result

def main():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--method", type=str, default="causal_dreamer")
    ap.add_argument("--num_frames", type=str, default="8", help="Folder name under outputs/<method>/")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="eval_results")
    ap.add_argument("--no_causal_metrics", action="store_true", help="Skip new causal metrics")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root(Path(__file__))
    
    outputs_root = project_root / "outputs"
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    METHOD = args.method
    NUM_FRAMES_FOLDER = args.num_frames
    method_root = outputs_root / METHOD / NUM_FRAMES_FOLDER

    print("=" * 60)
    print("CAUSAL DREAMER EVALUATION")
    print("=" * 60)
    print(f"PROJECT_ROOT: {project_root}")
    print(f"METHOD_ROOT: {method_root}")
    print(f"OUT_DIR: {out_dir}")
    print(f"DEVICE: {device}")
    print(f"COMPUTE_CAUSAL_METRICS: {not args.no_causal_metrics}")
    print("=" * 60)

    assert method_root.exists(), f"Missing: {method_root}"

    print("\n[1/3] Loading OpenCLIP...")
    model, preprocess, tokenizer = load_openclip(device=device)
    print("✓ OpenCLIP loaded")

    print("\n[2/3] Loading DINOv2...")
    dino_model = load_dinov2(device=device, model_name="dinov2_vits14")
    print("✓ DINOv2 loaded")

    samples = collect_samples(method_root)
    print(f"\n[3/3] Found {len(samples)} samples")

    rows_out = []
    for sample_id, jf in tqdm(samples, desc=f"Eval {METHOD}/{NUM_FRAMES_FOLDER}"):
        res = compute_metrics_for_sample(
            model, preprocess, tokenizer, dino_model, device, sample_id, jf,
            compute_causal_metrics=not args.no_causal_metrics
        )
        if res:
            rows_out.append(res)

    df = pd.DataFrame(rows_out)
    per_sample_csv = out_dir / f"{METHOD}_per_sample.csv"
    df.to_csv(per_sample_csv, index=False)
    print(f"\n✓ Saved per-sample results: {per_sample_csv}")

    # Summary statistics
    metrics = ["TCS", "CCS", "ID_Cons", "CLIPScore", "DINO_TCS", "DINO_IDCons"]
    if not args.no_causal_metrics:
        metrics.extend([
            "CPS", "CPS_monotonic", "CPS_avg_progress",
            "ETC", "ETC_min",
            "SDS_mean", "SDS_min", "SDS_max"
        ])
    
    summary_data = {"method": METHOD, "N_samples": len(df)}
    for m in metrics:
        if m in df.columns:
            summary_data[f"{m}_mean"] = float(df[m].mean(skipna=True))
            summary_data[f"{m}_std"] = float(df[m].std(skipna=True))
    
    summary = pd.DataFrame([summary_data])
    summary_csv = out_dir / f"{METHOD}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary: {summary_csv}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # Print key metrics
    key_metrics = ["CPS", "CCS", "ETC", "DINO_IDCons", "CLIPScore"]
    if args.no_causal_metrics:
        key_metrics = ["CCS", "DINO_IDCons", "TCS", "CLIPScore"]
    
    print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12}")
    print("-" * 45)
    for m in key_metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in summary_data:
            mean_val = summary_data[mean_col]
            std_val = summary_data[std_col]
            if not np.isnan(mean_val):
                print(f"{m:<20} {mean_val:>12.4f} {std_val:>12.4f}")
            else:
                print(f"{m:<20} {'N/A':>12} {'N/A':>12}")
    
    print("\n" + "=" * 60)
    print("METRIC DESCRIPTIONS:")
    print("-" * 60)
    print("CPS (Causal Progression): Progression from cause→effect")
    print("  → Higher = better (0.4+ good, 0.02 baseline)")
    print("CCS (Causal Consistency): Semantic causality")
    print("  → Positive = good, negative = violations")
    print("ETC (Event Coherence): Realistic transitions")
    print("  → 1.0 = optimal, 0.5 = static/jarring")
    print("SDS (Semantic Diversity): Outcome variety")
    print("  → Higher = more diverse (counterfactual mode)")
    print("DINO_IDCons: Identity preservation (gold standard)")
    print("  → Higher = better consistency")
    print("=" * 60)

if __name__ == "__main__":
    main()