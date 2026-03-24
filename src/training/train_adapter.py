# -*- coding: utf-8 -*-
"""
Causal Adapter training (rebalanced/index-aware)
Objectives:
  (1) Delta alignment (emb_e - emb_c)
  (2) Triplet ranking (pos closer than neg)
  (3) Retrieval (InfoNCE, CLIP-style) with optional semi-hard top-K negatives
Optional:
  (4) Effect classification head (linear) on the causal context, with separate logging CSV

Features (unchanged core):
- Row-index split (--list_is_indices / --list_index_base) and legacy ID split
- Frozen HF text encoder (--text_max_len)
- Learnable logit scale (temperature) with clamping
- Cosine schedule + warmup, EMA, gradient clipping
- Safe numerics, robust logging (main CSV + effect CSV)
- NEW: checkpoints also store `effect_vocab` and a clean `adapter_state_dict` for inference
"""

import os, re, csv, math, random
os.environ["CUDA_VISIBLE_DEVICES"]="6"

from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------- repo-local --------
from src.models.causal_adapter import CausalAdapter, CausalAdapterConfig

# -------- HF text encoder --------
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise SystemExit("pip install transformers")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ================= CSV & splits =================

def read_full_csv(full_csv: str, id_col: Optional[str] = None):
    import pandas as pd
    df = pd.read_csv(full_csv, dtype=str).fillna("")
    rows = df.to_dict(orient="records")
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"--id_col '{id_col}' not in CSV header {list(df.columns)}")
        return rows, id_col
    for c in ["id", "frame_id", "sample_id", "guid", "idx"]:
        if c in df.columns:
            return rows, c
    return rows, None


def _read_index_list(path: str, index_base: int, n_rows: int) -> List[int]:
    vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            for tok in re.split(r"[,\s]+", s):
                if not tok:
                    continue
                try:
                    idx = int(tok) - index_base
                    if 0 <= idx < n_rows:
                        vals.append(idx)
                except Exception:
                    pass
    return vals


def _normalize_id_token(tok: str, strip_ext: bool, take_basename: bool) -> str:
    s = tok.strip()
    if take_basename:
        s = os.path.basename(s)
    if strip_ext:
        s = os.path.splitext(s)[0]
    return s


def _read_id_list(path: str, strip_ext: bool, take_basename: bool) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            for tok in re.split(r"[,\s]+", s):
                if not tok:
                    continue
                ids.append(_normalize_id_token(tok, strip_ext, take_basename))
    return ids


def build_split(
    rows: List[Dict],
    train_list_path: str,
    val_list_path: str,
    list_is_indices: bool,
    list_index_base: int,
    id_col: Optional[str],
    strip_ext: bool,
    take_basename: bool,
):
    n_rows = len(rows)
    if n_rows == 0:
        raise SystemExit("FATAL: CSV has 0 rows.")

    if list_is_indices:
        train_idx = _read_index_list(train_list_path, list_index_base, n_rows)
        val_idx   = _read_index_list(val_list_path,   list_index_base, n_rows)
        want_train = sum(1 for _ in open(train_list_path, "r", encoding="utf-8"))
        want_val   = sum(1 for _ in open(val_list_path,   "r", encoding="utf-8"))
        print(f"[split:index] base={list_index_base} | n_rows={n_rows} | "
              f"train wanted≈{want_train}, matched={len(train_idx)} | "
              f"val wanted≈{want_val}, matched={len(val_idx)}")
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise SystemExit("FATAL: Split is empty in index mode. Check --list_index_base (0/1) and bounds.")
        train_rows = [rows[i] for i in train_idx]
        val_rows   = [rows[i] for i in val_idx]
        return train_rows, val_rows

    if not id_col:
        raise SystemExit(
            "FATAL: Not in index mode and no id column could be inferred. "
            "Either pass --id_col or use --list_is_indices."
        )

    print(f"[id] Using CSV id column: '{id_col}'")
    id_to_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        rid = (r.get(id_col) or "").strip()
        id_to_indices.setdefault(rid, []).append(i)

    train_ids = _read_id_list(train_list_path, strip_ext, take_basename)
    val_ids   = _read_id_list(val_list_path,   strip_ext, take_basename)

    def match_ids(wanted: List[str]) -> Tuple[List[int], int, List[str]]:
        matched: List[int] = []
        miss: List[str] = []
        for w in wanted:
            if w in id_to_indices:
                matched.extend(id_to_indices[w])
            else:
                miss.append(w)
        return matched, len(matched), miss

    tr_idx, tr_n, tr_miss = match_ids(train_ids)
    va_idx, va_n, va_miss = match_ids(val_ids)

    def _sample(xs, k=5):
        return xs[:k] if len(xs) > 0 else []

    print(f"[split]  train_ids: {len(train_ids)} wanted, {tr_n} matched, sample_missing={_sample(tr_miss)} | "
          f"val_ids: {len(val_ids)} wanted, {va_n} matched, sample_missing={_sample(va_miss)}")

    if tr_n == 0 or va_n == 0:
        examples = [str(rows[i].get(id_col, "")) for i in range(min(5, len(rows)))]
        raise SystemExit(
            "FATAL: Split is empty after ID matching.\n"
            "Common causes:\n"
            "- List files contain paths like '.../dir/123.jpg' but CSV has bare ids '123'\n"
            "- Case / extensions differ (use --list_strip_ext/--list_take_basename)\n"
            "- You passed indices but forgot --list_is_indices/--list_index_base\n"
            f"- CSV id column used: '{id_col}'\n\n"
            f"Examples from CSV ids: {examples}"
        )

    train_rows = [rows[i] for i in tr_idx]
    val_rows   = [rows[i] for i in va_idx]
    return train_rows, val_rows


# ================= Dataset =================

class CausalTextRowsDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cause  = (r.get("cause") or "").strip()
        action = (r.get("action") or "").strip()
        effect = (r.get("effect") or "").strip()
        return cause, action, effect


# ================= Text encoder =================

class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", text_max_len=77):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name)
        self.out_dim = self.enc.config.hidden_size
        self.text_max_len = text_max_len
        self.eval()

    @torch.no_grad()
    def encode_text(self, texts: List[str], device):
        batch = self.tok(
            texts, padding=True, truncation=True, max_length=self.text_max_len,
            return_tensors="pt"
        ).to(device)
        out = self.enc(**batch)
        last = out.last_hidden_state
        attn = batch["attention_mask"].unsqueeze(-1).to(last.dtype)
        denom = attn.sum(1).clamp_min(1e-6)
        pooled = (last * attn).sum(1) / denom
        pooled = F.normalize(pooled, dim=-1)
        return pooled, last  # [B,D], [B,T,D]


# ================= Effect vocab & freq =================

def build_effect_vocab_and_freq(rows: List[Dict], add_oov: bool = True) -> Tuple[Dict[str, int], Dict[str, int], Optional[int]]:
    vocab: Dict[str, int] = {}
    freq: Dict[str, int] = {}
    for r in rows:
        e = (r.get("effect") or "").strip()
        if e not in vocab:
            vocab[e] = len(vocab)
            freq[e] = 0
        freq[e] += 1
    oov_idx = None
    if add_oov:
        oov_idx = len(vocab)
        vocab["[OOV]"] = oov_idx
        freq["[OOV]"] = 0
    return vocab, freq, oov_idx


# ================= Adapter wrapper =================

class AdapterWrapper(nn.Module):
    """
    Wraps CausalAdapter and (optionally) an effect classifier on the causal context.
    If CausalAdapter forward doesn't return 'delta_e', falls back to a tiny MLP delta head.
    """
    def __init__(self, txt_dim, attn_dim=1024, rank=16, dropout=0.1,
                 learnable_logit_scale=False, logit_scale_init=0.07,
                 logit_scale_min=0.01, logit_scale_max=2.0,
                 effect_cls=False, num_effects: int = 0):
        super().__init__()
        try:
            cfg = CausalAdapterConfig(txt_dim=txt_dim, attn_dim=attn_dim, rank=rank, dropout=dropout)
        except TypeError:
            n_heads = 8
            d_model = max(attn_dim, n_heads * max(8, attn_dim // n_heads))
            cfg = CausalAdapterConfig(txt_dim=txt_dim, d_model=d_model, n_heads=n_heads, rank=rank, dropout=dropout)
        self.adapter = CausalAdapter(cfg)

        self.delta_head = nn.Sequential(
            nn.Linear(txt_dim * 2, txt_dim),
            nn.GELU(),
            nn.Linear(txt_dim, txt_dim)
        )

        # effect classifier (optional)
        self.effect_cls = effect_cls and (num_effects > 0)
        self.effect_classifier = nn.Linear(txt_dim, num_effects) if self.effect_cls else None

        self.learnable_logit_scale = learnable_logit_scale
        if learnable_logit_scale:
            self._logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / logit_scale_init), dtype=torch.float32))
            self.register_buffer("_logit_min", torch.tensor(logit_scale_min, dtype=torch.float32))
            self.register_buffer("_logit_max", torch.tensor(logit_scale_max, dtype=torch.float32))
        else:
            self._logit_scale = None
            self.register_buffer("_logit_const", torch.tensor(1.0 / logit_scale_init, dtype=torch.float32))

    def current_logit_scale(self):
        if self.learnable_logit_scale:
            s = torch.exp(self._logit_scale)
            return torch.clamp(s, float(self._logit_min.item()), float(self._logit_max.item()))
        else:
            return self._logit_const

    def forward(self, emb_cause, emb_action, tok_cause=None, tok_effect=None):
        out = None
        try:
            out = self.adapter(emb_cause=emb_cause, emb_action=emb_action,
                               tok_cause=tok_cause, tok_effect=tok_effect)
        except TypeError:
            try:
                out = self.adapter(emb_cause, emb_action)
            except Exception:
                out = None

        if isinstance(out, dict) and ("delta_e" in out):
            delta_e = out["delta_e"]
        else:
            if emb_action is None or emb_action.numel() == 0:
                emb_action = torch.zeros_like(emb_cause)
            delta_e = self.delta_head(torch.cat([emb_cause, emb_action], dim=-1))

        ctx = F.normalize(delta_e + emb_cause, dim=-1)

        logits_eff = None
        if self.effect_cls:
            logits_eff = self.effect_classifier(ctx)  # [B, num_effects]

        return {"delta_e": delta_e, "ctx": ctx, "effect_logits": logits_eff}


# ================= Losses =================

def triplet_rank_loss(ctx, pos, neg, margin=0.2):
    pos_sim = (ctx * pos).sum(-1)
    neg_sim = (ctx * neg).sum(-1)
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    acc = (pos_sim > neg_sim).float().mean().item()
    return loss, acc, pos_sim.mean().item()


def retrieval_infonce_loss(
    q, k, tau=0.07, symmetric=False,
    retr_topk_neg: int = 0, semi_hard_skip: int = 2, semi_hard_keep: Optional[int] = 8
):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    sims = (q @ k.t()).clamp(-50, 50)
    logits_full = sims * (1.0 / max(tau, 1e-6))
    targets = torch.arange(q.size(0), device=q.device)

    if retr_topk_neg and retr_topk_neg > 0:
        B = q.size(0)
        with torch.no_grad():
            vals, idxs = torch.sort(logits_full, dim=1, descending=True)
            keep = semi_hard_keep or min(8, retr_topk_neg)
            mask = torch.zeros_like(logits_full, dtype=torch.bool)
            for i in range(B):
                order = idxs[i].tolist()
                pos_rank = order.index(i)
                start = min(pos_rank + semi_hard_skip, B)
                end   = min(start + keep, B)
                mask[i, i] = True
                if start < end:
                    mask[i, idxs[i, start:end]] = True
        logits = logits_full.masked_fill(~mask, float("-inf"))
    else:
        logits = logits_full

    loss_q = F.cross_entropy(logits, targets)
    acc_q = (logits.argmax(dim=-1) == targets).float().mean().item()

    if symmetric:
        loss_k = F.cross_entropy(logits.t(), targets)
        acc_k = (logits.t().argmax(dim=-1) == targets).float().mean().item()
        loss = 0.5 * (loss_q + loss_k)
        acc = 0.5 * (acc_q + acc_k)
    else:
        loss = loss_q
        acc = acc_q
    return loss, acc


# ============ Val recall by effect freq (approx, chunked) ============

@torch.no_grad()
def compute_val_recall_by_freq(
    model, enc, val_rows: List[Dict], effect_to_idx: Dict[str, int], freq: Dict[str, int],
    tau=0.07, k_list=(1,5)
):
    model.eval()
    B = 128
    totals = {f"recall@{k}": 0 for k in k_list}
    counts = 0
    cutoffs = [1,2,5,10,20,50,100,1_000_000]
    bucket_hits = {f"r@1<=${c}": 0 for c in cutoffs}
    bucket_tot  = {f"r@1<=${c}": 0 for c in cutoffs}

    for i in range(0, len(val_rows), B):
        chunk = val_rows[i:i+B]
        cause  = [(r.get("cause")  or "").strip() for r in chunk]
        action = [(r.get("action") or "").strip() for r in chunk]
        effect = [(r.get("effect") or "").strip() for r in chunk]

        emb_c, T_c = enc.encode_text(cause,  DEVICE)
        emb_e, T_e = enc.encode_text(effect, DEVICE)
        if action and action[0]:
            emb_a, _ = enc.encode_text(action, DEVICE)
        else:
            emb_a = torch.zeros_like(emb_c)

        out = model(emb_cause=emb_c, emb_action=emb_a, tok_cause=T_c, tok_effect=T_e)
        ctx = F.normalize(out["delta_e"] + emb_c, dim=-1)
        emb_e = F.normalize(emb_e, dim=-1)

        sims = (ctx @ emb_e.t()).clamp(-50, 50) * (1.0 / max(tau, 1e-6))
        for n in range(len(chunk)):
            gold = n
            scores = sims[n]
            sorted_idx = torch.argsort(scores, descending=True)
            for k in k_list:
                if gold in sorted_idx[:k]:
                    totals[f"recall@{k}"] += 1
            counts += 1

            f = freq.get((chunk[n].get("effect") or "").strip(), 0)
            for cutoff in cutoffs:
                if f <= cutoff:
                    bucket_tot[f"r@1<=${cutoff}"] += 1
                    if gold == sorted_idx[0].item():
                        bucket_hits[f"r@1<=${cutoff}"] += 1
                    break

    out = {k: (v / max(counts, 1)) for k, v in totals.items()}
    for c in cutoffs:
        tot = bucket_tot[f"r@1<=${c}"]
        hit = bucket_hits[f"r@1<=${c}"]
        out[f"recall@1_freq<=$${c}"] = (hit / max(tot, 1)) if tot > 0 else 0.0
    return out


# ================= Utilities (logging/sched/EMA) =================

def _append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.data, 1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])


def build_scheduler(optimizer, epochs, steps_per_epoch, sched="cosine",
                    warmup_epochs=0, base_lr=2e-4, min_lr=1e-6):
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = max(0, warmup_epochs * max(1, steps_per_epoch))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = step - warmup_steps
        T = max(1, total_steps - warmup_steps)
        if sched == "cosine":
            min_ratio = min_lr / base_lr
            return min_ratio + 0.5 * (1 + math.cos(math.pi * t / T)) * (1 - min_ratio)
        else:
            frac = t / T
            if frac < 0.3: return 1.0
            elif frac < 0.6: return 0.5
            elif frac < 0.85: return 0.25
            else: return min_lr / base_lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ================= Train / Eval (with optional effect head) =================

def batch_effect_indices(effects: List[str], effect_to_idx: Dict[str, int], oov_idx: Optional[int]) -> torch.Tensor:
    idxs = []
    for e in effects:
        e = (e or "").strip()
        if e in effect_to_idx:
            idxs.append(effect_to_idx[e])
        else:
            idxs.append(oov_idx if oov_idx is not None else -1)
    return torch.tensor(idxs, dtype=torch.long, device=DEVICE)


def train_epoch(
    model, enc, dl, opt,
    lambda_triplet=0.5, lambda_delta=1.0, lambda_retr=10.0, lambda_effect=1.0,
    tau=0.07, use_symmetric_infonce=False,
    retr_topk_neg: int = 0,
    grad_clip_norm: float = 0.0,
    effect_cls: bool = False,
    effect_to_idx: Optional[Dict[str,int]] = None,
    oov_idx: Optional[int] = None,
):
    model.train()
    tot_loss = tot_trip = tot_delta = tot_retr = 0.0
    tot_eff = 0.0
    acc_trip = pos_sim_mean = acc_retr = 0.0
    eff_acc1 = 0.0
    n_batches = 0

    ce = nn.CrossEntropyLoss(ignore_index=-1) if effect_cls else None

    for (cause, action, effect) in dl:
        causes = list(cause); actions = list(action); effects = list(effect)

        with torch.no_grad():
            emb_c, T_c = enc.encode_text(causes, DEVICE)
            emb_e, T_e = enc.encode_text(effects, DEVICE)
            shuffled = torch.randperm(emb_e.size(0))
            emb_n = emb_e[shuffled]
            if actions and actions[0]:
                emb_a, _ = enc.encode_text(actions, DEVICE)
            else:
                emb_a = torch.zeros_like(emb_c)

        out = model(emb_cause=emb_c, emb_action=emb_a, tok_cause=T_c, tok_effect=T_e)
        delta_e = out["delta_e"]
        ctx     = out["ctx"]

        # Δ loss
        target_delta = emb_e - emb_c
        loss_delta = ((delta_e - target_delta) ** 2).mean()

        # Triplet
        l_trip, t_acc, pos_sim = triplet_rank_loss(ctx, emb_e, emb_n, margin=0.2)

        # Retrieval
        l_retr, r_acc = retrieval_infonce_loss(
            ctx, emb_e, tau=tau, symmetric=use_symmetric_infonce,
            retr_topk_neg=retr_topk_neg
        )

        # Effect classification (optional)
        l_eff, a_eff = 0.0, 0.0
        if effect_cls and out["effect_logits"] is not None:
            targets = batch_effect_indices(effects, effect_to_idx, oov_idx)
            logits = out["effect_logits"]
            l_eff = ce(logits, targets)
            with torch.no_grad():
                mask = targets != -1
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    a_eff = (preds[mask] == targets[mask]).float().mean().item()

        loss = lambda_delta * loss_delta + lambda_triplet * l_trip + lambda_retr * l_retr
        if effect_cls:
            loss = loss + lambda_effect * l_eff

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()

        tot_loss  += float(loss.detach())
        tot_trip  += float((lambda_triplet * l_trip).detach())
        tot_delta += float((lambda_delta * loss_delta).detach())
        tot_retr  += float((lambda_retr * l_retr).detach())
        if effect_cls:
            tot_eff += float((lambda_effect * l_eff).detach())
            eff_acc1 += a_eff

        acc_trip += t_acc
        pos_sim_mean += pos_sim
        acc_retr += r_acc
        n_batches += 1

    n_batches = max(n_batches, 1)
    out = {
        "train_loss": tot_loss / n_batches,
        "train_loss_triplet": tot_trip / n_batches,
        "train_loss_delta": tot_delta / n_batches,
        "train_loss_retr": tot_retr / n_batches,
        "train_acc_retr1": acc_retr / n_batches,
        "train_triplet_acc": acc_trip / n_batches,
        "train_pos_sim_mean": pos_sim_mean / n_batches,
    }
    if effect_cls:
        out["train_loss_effect"] = tot_eff / n_batches
        out["train_acc_effect1"] = eff_acc1 / n_batches
    return out


@torch.no_grad()
def eval_epoch(
    model, enc, dl, tau=0.07, use_symmetric_infonce=False, retr_topk_neg: int = 0,
    effect_cls: bool = False,
    effect_to_idx: Optional[Dict[str,int]] = None,
    oov_idx: Optional[int] = None,
):
    model.eval()
    tot_loss = tot_trip = tot_delta = tot_retr = 0.0
    tot_eff = 0.0
    acc_trip = pos_sim_mean = acc_retr = 0.0
    eff_acc1 = 0.0
    n_batches = 0

    ce = nn.CrossEntropyLoss(ignore_index=-1) if effect_cls else None

    for (cause, action, effect) in dl:
        causes = list(cause); actions = list(action); effects = list(effect)
        emb_c, T_c = enc.encode_text(causes, DEVICE)
        emb_e, T_e = enc.encode_text(effects, DEVICE)
        shuffled = torch.randperm(emb_e.size(0))
        emb_n = emb_e[shuffled]
        if actions and actions[0]:
            emb_a, _ = enc.encode_text(actions, DEVICE)
        else:
            emb_a = torch.zeros_like(emb_c)

        out = model(emb_cause=emb_c, emb_action=emb_a, tok_cause=T_c, tok_effect=T_e)
        delta_e = out["delta_e"]
        ctx     = out["ctx"]

        target_delta = emb_e - emb_c
        loss_delta = ((delta_e - target_delta) ** 2).mean()

        l_trip, t_acc, pos_sim = triplet_rank_loss(ctx, emb_e, emb_n, margin=0.2)
        l_retr, r_acc = retrieval_infonce_loss(
            ctx, emb_e, tau=tau, symmetric=use_symmetric_infonce,
            retr_topk_neg=retr_topk_neg
        )

        l_eff, a_eff = 0.0, 0.0
        if effect_cls and out["effect_logits"] is not None:
            targets = batch_effect_indices(effects, effect_to_idx, oov_idx)
            logits = out["effect_logits"]
            l_eff = ce(logits, targets)
            mask = targets != -1
            if mask.any():
                preds = logits.argmax(dim=-1)
                a_eff = (preds[mask] == targets[mask]).float().mean().item()

        loss = loss_delta + l_trip + l_retr
        if effect_cls:
            loss = loss + l_eff

        tot_loss  += float(loss.detach())
        tot_trip  += float(l_trip.detach())
        tot_delta += float(loss_delta.detach())
        tot_retr  += float(l_retr.detach())
        if effect_cls:
            tot_eff += float(l_eff.detach())
            eff_acc1 += a_eff

        acc_trip += t_acc
        pos_sim_mean += pos_sim
        acc_retr += r_acc
        n_batches += 1

    n_batches = max(n_batches, 1)
    out = {
        "val_loss": tot_loss / n_batches,
        "val_loss_triplet": tot_trip / n_batches,
        "val_loss_delta": tot_delta / n_batches,
        "val_loss_retr": tot_retr / n_batches,
        "val_acc_retr1": acc_retr / n_batches,
        "val_triplet_acc": acc_trip / n_batches,
        "val_pos_sim_mean": pos_sim_mean / n_batches,
    }
    if effect_cls:
        out["val_loss_effect"] = tot_eff / n_batches
        out["val_acc_effect1"] = eff_acc1 / n_batches
    return out


# ================= Main =================

def main():
    import argparse
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        _tz = ZoneInfo("America/New_York")
    except Exception:
        _tz = None

    ap = argparse.ArgumentParser()
    # splits / csv
    ap.add_argument("--full_csv", required=True)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list",   required=True)
    ap.add_argument("--id_col", default="")
    ap.add_argument("--list_is_indices", action="store_true")
    ap.add_argument("--list_index_base", type=int, default=0)
    ap.add_argument("--list_strip_ext", action="store_true")
    ap.add_argument("--list_take_basename", action="store_true")

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--txt_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--text_max_len", type=int, default=77)
    ap.add_argument("--attn_dim", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)

    # losses
    ap.add_argument("--lambda_triplet", type=float, default=0.5)
    ap.add_argument("--lambda_delta", type=float, default=1.0)
    ap.add_argument("--lambda_retr", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--symmetric_infonce", action="store_true")

    # effect head
    ap.add_argument("--effect_cls", action="store_true", help="Enable linear effect classifier on context")
    ap.add_argument("--lambda_effect", type=float, default=1.0)
    ap.add_argument("--effect_log_csv", default="effect_loss.csv")  # separate CSV for effect losses/acc

    # logit scale / negatives
    ap.add_argument("--learnable_logit_scale", action="store_true")
    ap.add_argument("--logit_scale_init", type=float, default=0.07)
    ap.add_argument("--logit_scale_min", type=float, default=0.01)
    ap.add_argument("--logit_scale_max", type=float, default=2.0)
    ap.add_argument("--retr_topk_neg", type=int, default=0)

    # schedule / ema / clipping
    ap.add_argument("--sched", choices=["cosine", "step"], default="cosine")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--grad_clip_norm", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=0.0)  # legacy

    ap.add_argument("--num_workers", type=int, default=4)

    # io
    ap.add_argument("--save_dir", default="outputs")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log_csv", default="")

    args = ap.parse_args()

    # out dir
    now = datetime.now(tz=_tz) if _tz is not None else datetime.now()
    stamp = now.strftime("%m%d%Y_%H%M")
    out_dir = os.path.join(args.save_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    if not args.log_csv:
        args.log_csv = os.path.join(out_dir, "train_log.csv")
    if not args.effect_log_csv:
        args.effect_log_csv = os.path.join(out_dir, "effect_log.csv")
    save_latest = os.path.join(out_dir, "causal_adapter.pt")
    save_best   = os.path.join(out_dir, "causal_adapter_best.pt")

    print("out_dir:", out_dir)

    # data
    rows, used_id_col = read_full_csv(args.full_csv, id_col=args.id_col if args.id_col else None)
    train_rows, val_rows = build_split(
        rows=rows,
        train_list_path=args.train_list,
        val_list_path=args.val_list,
        list_is_indices=args.list_is_indices,
        list_index_base=args.list_index_base,
        id_col=used_id_col,
        strip_ext=args.list_strip_ext,
        take_basename=args.list_take_basename,
    )
    print(f"Train examples: {len(train_rows)} | Val examples: {len(val_rows)}")

    # vocab/freq
    effect_to_idx, effect_freq, oov_idx = build_effect_vocab_and_freq(train_rows, add_oov=True)
    effect_vocab = list(effect_to_idx.keys())  # <-- used for saving to ckpt
    print(f"Unique effects (train): {len(effect_to_idx)} (includes [OOV]={oov_idx})")

    # loaders
    train_dl = DataLoader(CausalTextRowsDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(CausalTextRowsDataset(val_rows),   batch_size=args.batch_size, shuffle=False,
                          drop_last=False, num_workers=args.num_workers, pin_memory=True)

    # encoder / model
    enc = TextEncoder(args.txt_model, text_max_len=args.text_max_len).to(DEVICE).eval()
    txt_dim = enc.out_dim

    model = AdapterWrapper(
        txt_dim=txt_dim,
        attn_dim=args.attn_dim,
        rank=args.rank,
        dropout=args.dropout,
        learnable_logit_scale=args.learnable_logit_scale,
        logit_scale_init=args.logit_scale_init if args.learnable_logit_scale else args.tau,
        logit_scale_min=args.logit_scale_min,
        logit_scale_max=args.logit_scale_max,
        effect_cls=args.effect_cls,
        num_effects=len(effect_to_idx)
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = max(1, len(train_dl))
    sched = build_scheduler(
        opt, args.epochs, steps_per_epoch,
        sched=args.sched, warmup_epochs=args.warmup_epochs,
        base_lr=args.lr, min_lr=args.min_lr
    )
    ema_obj = EMA(model, decay=args.ema_decay) if args.ema else None
    global_step = 0

    start_epoch = 1
    best_val = math.inf

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected:
            print(f"[resume] Ignored unexpected keys: {unexpected}")
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[resume] Skipping optimizer load: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", math.inf)
        print(f"Resumed from epoch {start_epoch-1}, best_val={best_val:.4f}")

    # write headers (if empty files)
    if not os.path.exists(args.log_csv):
        _append_csv(args.log_csv, {
            "epoch": 0,
            "train_loss": "", "train_loss_triplet": "", "train_loss_delta": "", "train_loss_retr": "",
            "train_acc_retr1": "", "train_triplet_acc": "", "train_pos_sim_mean": "",
            "val_loss": "", "val_loss_triplet": "", "val_loss_delta": "", "val_loss_retr": "",
            "val_acc_retr1": "", "val_triplet_acc": "", "val_pos_sim_mean": "",
            "val_recall@1": "", "val_recall@5": "",
            "val_recall@1_freq<=$1": "", "val_recall@1_freq<=$2": "", "val_recall@1_freq<=$5": "",
            "val_recall@1_freq<=$10": "", "val_recall@1_freq<=$20": "", "val_recall@1_freq<=$50": "",
            "val_recall@1_freq<=$100": "", "val_recall@1_freq<=$1000000": "",
            "lr": ""
        })
    if args.effect_cls and not os.path.exists(args.effect_log_csv):
        _append_csv(args.effect_log_csv, {
            "epoch": 0,
            "train_effect_loss": "", "train_effect_acc1": "",
            "val_effect_loss": "", "val_effect_acc1": ""
        })

    # ---------- train loop ----------
    for e in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model, enc, train_dl, opt,
            lambda_triplet=args.lambda_triplet,
            lambda_delta=args.lambda_delta,
            lambda_retr=args.lambda_retr,
            lambda_effect=args.lambda_effect,
            tau=args.tau,
            use_symmetric_infonce=args.symmetric_infonce,
            retr_topk_neg=args.retr_topk_neg,
            grad_clip_norm=(args.grad_clip_norm if args.grad_clip_norm > 0 else args.clip_grad),
            effect_cls=args.effect_cls,
            effect_to_idx=effect_to_idx,
            oov_idx=oov_idx
        )
        val_metrics = eval_epoch(
            model, enc, val_dl,
            tau=args.tau,
            use_symmetric_infonce=args.symmetric_infonce,
            retr_topk_neg=args.retr_topk_neg,
            effect_cls=args.effect_cls,
            effect_to_idx=effect_to_idx,
            oov_idx=oov_idx
        )

        # FIXED: pass true effect_freq (not effect_to_idx)
        recall_metrics = compute_val_recall_by_freq(
            model, enc, val_rows, effect_to_idx, effect_freq,
            tau=args.tau, k_list=(1,5)
        )

        for _ in range(steps_per_epoch):
            sched.step()
            global_step += 1

        cur_lr = opt.param_groups[0]["lr"]
        if ema_obj is not None:
            ema_obj.update(model)

        # main CSV
        row = {
            "epoch": e,
            **{k: v for k, v in train_metrics.items() if k.startswith("train_")},
            **{k: v for k, v in val_metrics.items() if k.startswith("val_")},
            "val_recall@1": recall_metrics.get("recall@1", 0.0),
            "val_recall@5": recall_metrics.get("recall@5", 0.0),
            "val_recall@1_freq<=$1": recall_metrics.get("recall@1_freq<=$$1", 0.0),
            "val_recall@1_freq<=$2": recall_metrics.get("recall@1_freq<=$$2", 0.0),
            "val_recall@1_freq<=$5": recall_metrics.get("recall@1_freq<=$$5", 0.0),
            "val_recall@1_freq<=$10": recall_metrics.get("recall@1_freq<=$$10", 0.0),
            "val_recall@1_freq<=$20": recall_metrics.get("recall@1_freq<=$$20", 0.0),
            "val_recall@1_freq<=$50": recall_metrics.get("recall@1_freq<=$$50", 0.0),
            "val_recall@1_freq<=$100": recall_metrics.get("recall@1_freq<=$$100", 0.0),
            "val_recall@1_freq<=$1000000": recall_metrics.get("recall@1_freq<=$$1000000", 0.0),
            "lr": cur_lr,
        }
        _append_csv(args.log_csv, row)

        # effect CSV (separate)
        if args.effect_cls:
            eff_row = {
                "epoch": e,
                "train_effect_loss": train_metrics.get("train_loss_effect", 0.0),
                "train_effect_acc1": train_metrics.get("train_acc_effect1", 0.0),
                "val_effect_loss": val_metrics.get("val_loss_effect", 0.0),
                "val_effect_acc1": val_metrics.get("val_acc_effect1", 0.0),
            }
            _append_csv(args.effect_log_csv, eff_row)

        # console print
        print(
            f"[epoch {e:03d}] "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"Δ={val_metrics['val_loss_delta']:.4f} | "
            f"trip={val_metrics['val_loss_triplet']:.4f} | "
            f"retr={val_metrics['val_loss_retr']:.4f} | "
            f"train_acc(retr@1)={train_metrics['train_acc_retr1']:.3f} | "
            f"val_acc(retr@1)={val_metrics['val_acc_retr1']:.3f} | "
            f"train_trip_acc={train_metrics['train_triplet_acc']:.3f} | "
            f"val_trip_acc={val_metrics['val_triplet_acc']:.3f} | "
            f"pos_sim={val_metrics['val_pos_sim_mean']:.3f} | "
            f"r@1={row['val_recall@1']:.3f} | r@5={row['val_recall@5']:.3f} | "
            f"lr={cur_lr:.6f}"
            + ("" if not args.effect_cls else
               f" | eff(train)={train_metrics.get('train_loss_effect',0.0):.3f}/{train_metrics.get('train_acc_effect1',0.0):.3f}"
               f" eff(val)={val_metrics.get('val_loss_effect',0.0):.3f}/{val_metrics.get('val_acc_effect1',0.0):.3f}")
        )

        # ====== Save checkpoints ======
        # Common payload
        ckpt_common = {
            "state_dict": model.state_dict(),                 # full wrapper (for continued training)
            "adapter_state_dict": model.adapter.state_dict(), # CLEAN CausalAdapter only (for inference)
            "optimizer": opt.state_dict(),
            "config": {"txt_dim": txt_dim, "attn_dim": args.attn_dim, "rank": args.rank, "dropout": args.dropout},
            "txt_model": args.txt_model,
            "epoch": e,
            "val_loss": val_metrics["val_loss"],
            "effect_vocab": effect_vocab,                     # <-- for inference retrieval
        }

        # save latest
        torch.save(ckpt_common, save_latest)

        # periodic every 20
        if e % 20 == 0:
            ck = os.path.join(out_dir, f"causal_adapter_epoch_{e}.pt")
            torch.save(ckpt_common, ck)
            print(f"Saved checkpoint at epoch {e}: {ck}")

        # best
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            best_payload = dict(ckpt_common)
            best_payload["val_loss"] = best_val
            torch.save(best_payload, save_best)

    print("Done.")


if __name__ == "__main__":
    main()
