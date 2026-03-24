# src/inference/causal_intervention_generator.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
import datetime
import Path

# ---------------- PATHS (ROBUST) ----------------
def find_project_root(start: Path) -> Path:
    """
    Find project root by walking up until we find data/causal_full.csv.
    """
    p = start.resolve()
    for _ in range(12):
        if (p / "data" / "causal_full.csv").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root containing data/causal_full.csv")

PROJECT_ROOT = find_project_root(Path(__file__))
DATA_ROOT = PROJECT_ROOT / "data"

from src.models.causal_adapter import CausalAdapter, CausalAdapterConfig
from src.models.causal_intervention_processor import CausalInterventionProcessor
from src.models.causal_intervention_attention import CausalInterventionAttention

# -----------------------
# Text helpers
# -----------------------
@torch.no_grad()
def encode_text(tok, txt, texts, device):
    """
    Returns (pooled, tokens):
      pooled: [B, D_txt]
      tokens: [B, N, D_txt]
    """
    t = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    o = txt(**t)
    pooled = getattr(o, "pooler_output", None)
    last = o.last_hidden_state
    if pooled is None:
        pooled = last.mean(dim=1)
    return pooled, last


def _build_adapter_config(txt_dim: int, attn_dim: int = 1024, rank: int = 16, dropout: float = 0.1):
    """Adapter config compatible with older/newer CausalAdapterConfig signatures."""
    try:
        return CausalAdapterConfig(
            txt_dim=txt_dim,
            attn_dim=attn_dim,
            rank=rank,
            dropout=dropout,
        )
    except TypeError:
        # Backward-compat args
        n_heads = 8
        d_model = max(attn_dim, n_heads * max(8, attn_dim // n_heads))
        return CausalAdapterConfig(
            txt_dim=txt_dim,
            d_model=d_model,
            n_heads=n_heads,
            rank=rank,
            dropout=dropout,
        )


# -----------------------
# Checkpoint loading (robust to version drift)
# -----------------------
def _reconstruct_effect_vocab(ckpt):
    """
    Training saves either:
      - 'effect_to_idx' (dict: text -> index), or
      - 'effect_vocab' (list)
    We return a list in correct index order if possible.
    """
    if isinstance(ckpt, dict):
        if "effect_vocab" in ckpt and isinstance(ckpt["effect_vocab"], list):
            return ckpt["effect_vocab"]
        if "effect_to_idx" in ckpt and isinstance(ckpt["effect_to_idx"], dict):
            pairs = sorted(ckpt["effect_to_idx"].items(), key=lambda kv: kv[1])
            return [p[0] for p in pairs]
    return []


def _filter_state_dict_for_model(state_dict, model):
    """
    Keep only keys that exist in model.state_dict() *and* have the same shape.
    This avoids size-mismatch crashes when architectures drift (e.g., delta_head).
    """
    msd = model.state_dict()
    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped.append((k, tuple(v.shape) if hasattr(v, "shape") else None,
                            tuple(msd[k].shape) if k in msd else None))
    return filtered, dropped


def load_trained_adapter(adapter_path, device="cuda"):
    """
    Loads a trained adapter checkpoint in a way that tolerates architecture drift:
    - Drops any keys whose shapes don't match the current CausalAdapter
    - Strips old classifier weights automatically
    Returns: (adapter, cfg_dict, effect_vocab)
    """
    try:
        ckpt = torch.load(adapter_path, map_location=device, weights_only=True)  # torch >= 2.4
    except TypeError:
        ckpt = torch.load(adapter_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
        cfg_dict = ckpt.get("config", {}) or {}
    else:
        raw_sd = ckpt
        cfg_dict = {}

    # Build adapter skeleton
    txt_dim = int(cfg_dict.get("txt_dim", 384))
    attn_dim = int(cfg_dict.get("attn_dim", cfg_dict.get("d_model", 1024)))
    dropout = float(cfg_dict.get("dropout", 0.1))
    rank = int(cfg_dict.get("rank", 16))

    cfg = _build_adapter_config(txt_dim=txt_dim, attn_dim=attn_dim, rank=rank, dropout=dropout)
    adapter = CausalAdapter(cfg).to(device)

    # Drop legacy classifier keys explicitly (old runs)
    raw_sd = {k: v for k, v in raw_sd.items() if not k.startswith("effect_classifier.")}

    # Filter by exact shape match
    filtered_sd, dropped = _filter_state_dict_for_model(raw_sd, adapter)
    if dropped:
        # show only a few lines to keep logs readable
        print(f"[load] Dropped {len(dropped)} keys due to shape mismatch. Examples:")
        for i, (k, sv, mv) in enumerate(dropped[:6]):
            print(f"       - {k}: ckpt={sv}  model={mv}")
        if len(dropped) > 6:
            print("       ...")

    adapter.load_state_dict(filtered_sd, strict=False)
    adapter.eval().half()

    effect_vocab = _reconstruct_effect_vocab(ckpt)
    return adapter, cfg_dict, effect_vocab


# -----------------------
# “Enhanced” wrapper (keeps your intervention knob)
# -----------------------
class EnhancedCausalAdapter(nn.Module):
    def __init__(self, original_adapter, d_model=1024, n_heads=16, txt_dim=384):
        super().__init__()
        self.original_adapter = original_adapter
        self.intervention_attention = CausalInterventionAttention(d_model=d_model, n_heads=n_heads)

        # Optional learned strength; static for now
        self.strength_predictor = nn.Sequential(
            nn.Linear(txt_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, e_c, e_a, tok_cause=None, tok_effect=None):
        out = self.original_adapter(e_c, e_a, tok_cause=tok_cause, tok_effect=tok_effect)
        intervention_strength = torch.tensor(0.7, device=e_c.device, dtype=e_c.dtype)
        return {**out, "intervention_strength": intervention_strength}


# -----------------------
# Generator
# -----------------------
class CausalInterventionGenerator:
    def __init__(self, adapter_path, base_model="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device

        print("Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        sd_txt_dim = int(getattr(self.pipe.text_encoder.config, "hidden_size", 768))

        print("Loading trained adapter...")
        self.original_adapter, cfg_dict, effect_vocab = load_trained_adapter(adapter_path, device=self.device)
        self.adapter_txt_dim = int(cfg_dict.get("txt_dim", 384))
        self.effect_vocab = effect_vocab or []

        # Text projector: SD text-encoder dim -> adapter txt_dim
        print("Setting up text projector...")
        self.original_adapter.txt_proj = nn.Linear(sd_txt_dim, self.adapter_txt_dim, bias=False).to(
            self.device, dtype=self.pipe.text_encoder.dtype
        )
        nn.init.normal_(self.original_adapter.txt_proj.weight, std=0.02)

        print("Setting up enhanced adapter...")
        attn_dim = int(cfg_dict.get("attn_dim", cfg_dict.get("d_model", 1024)))
        n_heads = int(cfg_dict.get("n_heads", 16))
        self.enhanced_adapter = EnhancedCausalAdapter(
            self.original_adapter, d_model=attn_dim, n_heads=n_heads, txt_dim=self.adapter_txt_dim
        ).to(self.device).eval()

        print("Installing intervention processors...")
        self.install_intervention_processors()

        print("Building effect index (embedding cache)...")
        self._build_effect_index()

        print("CausalInterventionGenerator ready!")
        self.adapter_ckpt = adapter_path

    # ---- UNet attention processors with intervention ----
    def install_intervention_processors(self):
        self.processors = {}
        for name in self.pipe.unet.attn_processors.keys():
            intervention_attn = CausalInterventionAttention(
                d_model=1024,  # SD 1.5 UNet attn dim
                n_heads=16
            )
            processor = CausalInterventionProcessor(intervention_attn)
            self.processors[name] = processor
        self.pipe.unet.set_attn_processor(self.processors)

    # ---- Effect vocabulary embedding cache ----
    @torch.no_grad()
    def _build_effect_index(self):
        """
        Encode all effect_vocab texts with SD text encoder, project with adapter.txt_proj,
        L2-normalize and cache for cosine retrieval.
        """
        if not self.effect_vocab:
            self.effect_embs = None
            return

        batch = 128
        embs = []
        for i in range(0, len(self.effect_vocab), batch):
            chunk = self.effect_vocab[i:i + batch]
            pooled, _ = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, chunk, self.device)
            proj = self.original_adapter.txt_proj(pooled)
            embs.append(F.normalize(proj, dim=-1))
        self.effect_embs = torch.cat(embs, dim=0).to(self.device, dtype=self.pipe.text_encoder.dtype)  # [V, D]

    # ---- Utility: token mask from token embeddings ----
    @torch.no_grad()
    def create_token_mask(self, token_embeddings):
        return (token_embeddings.abs().sum(dim=-1) > 0.1).float()

    # ---- Cosine retrieval of top-k effects from cause/action ----
    @torch.no_grad()
    def predict_top_effects_cosine(self, cause_text, action_text="", top_k=4):
        if self.effect_embs is None or len(self.effect_vocab) == 0:
            return [f"effect {i+1}" for i in range(top_k)], None

        e_c_pool, _ = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [cause_text], self.device)
        e_c_proj = self.original_adapter.txt_proj(e_c_pool)

        if action_text:
            e_a_pool, _ = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [action_text], self.device)
            e_a_proj = self.original_adapter.txt_proj(e_a_pool)
        else:
            e_a_proj = torch.zeros_like(e_c_proj)

        out = self.original_adapter(e_c_proj, e_a_proj, tok_cause=None, tok_effect=None)
        delta_e = out.get("delta_e", None)
        if delta_e is None:
            raise RuntimeError("Adapter did not return 'delta_e' in forward output.")

        ctx = F.normalize(delta_e + e_c_proj, dim=-1)   # [1, D]
        sims = (ctx @ self.effect_embs.T).squeeze(0)    # [V]
        top_scores, top_idx = torch.topk(sims, k=min(top_k, sims.numel()))
        effects = [self.effect_vocab[i] for i in top_idx.tolist()]
        return effects, top_scores

    # ---- Public API: generate from prompt using predicted effects ----
    @torch.no_grad()
    def generate_from_prompt(self, prompt, num_effects=4, **gen_kwargs):
        import json
        print(f"Generating {num_effects} adapter-predicted effects for prompt: {prompt}")

        cause = self.extract_cause_from_prompt(prompt)
        print(f"Adapter extracted cause: {cause}")

        # get predicted effects + (optional) scores
        predicted_effects, scores = self.predict_top_effects_cosine(cause, "", top_k=num_effects)
        print(f"Adapter predicted effects: {predicted_effects}")

        # Encode cause once (for masks)
        e_c_pool, T_c = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [cause], self.device)
        e_c_proj = self.original_adapter.txt_proj(e_c_pool)
        T_c_proj = self.original_adapter.txt_proj(T_c)
        cause_mask = self.create_token_mask(T_c)

        images = []
        default_kwargs = dict(num_inference_steps=30, guidance_scale=7.5)
        default_kwargs.update(gen_kwargs)

        # We’ll collect per-image metadata to write a txt+json later
        meta_rows = []

        for i, effect in enumerate(predicted_effects):
            print(f"Generating image {i+1}/{len(predicted_effects)}: {cause} -> {effect}")

            # Encode this effect for masking
            _, T_e = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [effect], self.device)
            T_e_proj = self.original_adapter.txt_proj(T_e)
            effect_mask = self.create_token_mask(T_e)

            # Gates / intervention strength
            out = self.enhanced_adapter(e_c_proj, torch.zeros_like(e_c_proj), T_c_proj, T_e_proj)
            intervention_strength = float(out["intervention_strength"])

            # Set intervention context
            for processor in self.processors.values():
                processor.set_causal_context(cause_mask, effect_mask, intervention_strength)

            # Generate
            image_prompt = f"Consistent scene: {cause}. Therefore, {effect}"
            generator = torch.Generator(device=self.device).manual_seed(42)
            image = self.pipe(image_prompt, generator=generator, **default_kwargs).images[0]
            images.append((effect, image))

            # Record metadata row
            score_val = float(scores[i]) if scores is not None else None
            meta_rows.append({
                "index": i,
                "prompt": prompt,
                "extracted_cause": cause,
                "effect": effect,
                "image_prompt": image_prompt,
                "similarity_score": score_val,
                "intervention_strength": intervention_strength,
            })

            # Clear context
            for processor in self.processors.values():
                processor.clear_context()

        ts = datetime.datetime.now().strftime("%m%d%y_%H%M")
        outdir = f"outputs_diff_{ts}"
        os.makedirs(outdir, exist_ok=True)
        image_list = []

        # Save images + augment metadata with filenames
        for i, (effect, img) in enumerate(images):
            safe_effect = effect.replace(" ", "_")
            path = os.path.join(outdir, f"{i}_{safe_effect}.png")
            img.save(path)
            image_list.append(img)
            print("Saved:", path)
            meta_rows[i]["image_file"] = path

        # Save a grid for convenience (unchanged)
        if image_list:
            grid = create_image_grid(image_list, rows=2, cols=5)
            grid_filename = os.path.join(outdir, f"GRID_{prompt.replace(' ', '_')}.png")
            grid.save(grid_filename)
            print(f"Saved grid: {grid_filename}")

        # --- NEW: save text + JSON metadata sidecar files ---
        # Plain-text summary (easy to skim)
        txt_path = os.path.join(outdir, "generated_effects.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Extracted cause: {cause}\n")
            f.write(f"Adapter checkpoint: {getattr(self, 'adapter_ckpt', 'unknown')}\n")
            f.write("\nGenerated effects:\n")
            for r in meta_rows:
                line = (
                    f"[{r['index']:02d}] effect='{r['effect']}' "
                    f"score={r['similarity_score'] if r['similarity_score'] is not None else 'NA'} "
                    f"strength={r['intervention_strength']:.3f} "
                    f"file={os.path.basename(r['image_file'])}"
                )
                f.write(line + "\n")
        print(f"Saved metadata text: {txt_path}")

        # JSON with full details (good for scripting)
        json_path = os.path.join(outdir, "generated_effects.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "prompt": prompt,
                "extracted_cause": cause,
                "adapter_checkpoint": getattr(self, "adapter_ckpt", "unknown"),
                "num_effects": len(meta_rows),
                "rows": meta_rows,
                "grid_file": grid_filename if image_list else None,
                "output_dir": outdir,
                "timestamp": ts,
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata JSON: {json_path}")

        return images


    # ---- Optional: multi-effect branching from a cause ----
    @torch.no_grad()
    def generate_with_intervention(self, cause, action="", num_effects=4, **gen_kwargs):
        print(f"Generating {num_effects} possible effects for: {cause}")
        predicted_effects, _ = self.predict_top_effects_cosine(cause, action, top_k=num_effects)

        # Encode cause once
        e_c_pool, T_c = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [cause], self.device)
        e_c_proj = self.original_adapter.txt_proj(e_c_pool)
        T_c_proj = self.original_adapter.txt_proj(T_c)
        cause_mask = self.create_token_mask(T_c)

        images = []
        default_kwargs = dict(num_inference_steps=30, guidance_scale=7.5)
        default_kwargs.update(gen_kwargs)

        for i, effect in enumerate(predicted_effects):
            print(f"Generating image {i+1}/{num_effects}: {cause} -> {effect}")
            _, T_e = encode_text(self.pipe.tokenizer, self.pipe.text_encoder, [effect], self.device)
            T_e_proj = self.original_adapter.txt_proj(T_e)
            effect_mask = self.create_token_mask(T_e)

            out = self.enhanced_adapter(e_c_proj, torch.zeros_like(e_c_proj), T_c_proj, T_e_proj)
            strength = float(out["intervention_strength"])

            for processor in self.processors.values():
                processor.set_causal_context(cause_mask, effect_mask, strength)

            prompt = f"Consistent scene: {cause}. {action}. Therefore, {effect}"
            generator = torch.Generator(device=self.device).manual_seed(42)
            image = self.pipe(prompt, generator=generator, **default_kwargs).images[0]
            images.append((effect, image))

            for processor in self.processors.values():
                processor.clear_context()

        return images

    # ---- Simple cause extraction ----
    def extract_cause_from_prompt(self, prompt):
        indicators = [" causes ", " leads to ", " results in ", " creates ", " produces "]
        for ind in indicators:
            if ind in prompt:
                return prompt.split(ind)[0].strip()
        return prompt.split(".")[0].strip()


# -----------------------
# Grid utility
# -----------------------
def create_image_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# -----------------------
# Quick tests
# -----------------------
adapter_path = "/checkpoint/causal_adapter_epoch_200.pt"
def test_generator():
    timestamp = datetime.datetime.now().strftime("%m%d%y_%H%M")
    output_dir = f"outputs_diff_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    generator = CausalInterventionGenerator(adapter_path=adapter_path)
    prompt = "A bucket overflows, spilling water onto the ground"

    images = generator.generate_from_prompt(prompt=prompt, num_effects=10)

    image_list = []
    for i, (effect, img) in enumerate(images):
        filename = os.path.join(output_dir, f"{i}_{effect.replace(' ', '_')}.png")
        img.save(filename)
        image_list.append(img)
        print(f"Saved: {filename}")

    if image_list:
        grid = create_image_grid(image_list, rows=2, cols=5)
        grid_filename = os.path.join(output_dir, f"GRID_{prompt.replace(' ', '_')}.png")
        grid.save(grid_filename)
        print(f"Saved grid: {grid_filename}")

    print(f"All images saved to: {output_dir}")
    return images


def test_with_debug():
    generator = CausalInterventionGenerator(adapter_path=adapter_path)
    test_causes = ["a dog jumping on a trampoline","A bucket overflows, spilling water onto the ground"]
 #   test_causes = ["A person is riding a bicycle", "A cat is playing with a ball of yarn"]
    #test_causes = ["A child is drawing with crayons"]
    #test_causes = ["A cat is playing with a ball of yarn"]
    test_causes = ["A child blowing candle of birthday cake"]
    test_causes = ["A white dog is riding a bicycle"]

    for cause in test_causes:
        effects, scores = generator.predict_top_effects_cosine(cause, "", top_k=5)
        print("\n=== DEBUG EFFECT PREDICTION ===")
        print(f"Cause: {cause}")
        for i, e in enumerate(effects):
            s = float(scores[i]) if scores is not None else float('nan')
            print(f"  {i+1:2d}. {e} (sim: {s:.3f})")
        print("=== END DEBUG ===\n")

        _ = generator.generate_from_prompt(cause, num_effects=5)


if __name__ == "__main__":
    # test_generator()
    test_with_debug()
