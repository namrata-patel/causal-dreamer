import torch
import torch.nn as nn

class CausalAdapterConfig:
    def __init__(self, txt_dim=768, d_model=1024, n_heads=8, rank=16, dropout=0.1, use_logit_bias=True, num_effect_candidates=10):
        self.txt_dim = txt_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.dropout = dropout
        self.use_logit_bias = use_logit_bias
        self.num_effect_candidates = num_effect_candidates
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    @property
    def head_dim(self):
        return self.d_model // self.n_heads

class CausalAdapter(nn.Module):
    def __init__(self, cfg: CausalAdapterConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.txt_dim
        H, Dh = cfg.n_heads, cfg.head_dim

        # Core components only
        self.fuse = nn.Sequential(
            nn.LayerNorm(d*2), 
            nn.Linear(d*2, d), 
            nn.GELU(), 
            nn.Dropout(cfg.dropout), 
            nn.Linear(d, d)
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d), 
            nn.Linear(d, d)
        )
        self.q_gate = nn.Sequential(
            nn.Linear(d, H*Dh), 
            nn.SiLU()
        )
        self.k_gate = nn.Sequential(
            nn.Linear(d, H*Dh), 
            nn.SiLU()
        )

        # === ADD EFFECT PREDICTION HEAD HERE ===
        self.effect_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),  # Predict effect embedding
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d)
        )
        
        # Effect classification head (if you have discrete effects)
        print(f"DEBUG - cfg.num_effects: {getattr(cfg, 'num_effects', 'NOT SET')}")  # ADD THIS
        if hasattr(cfg, 'num_effects') and cfg.num_effects > 0:
            print(f"DEBUG - Creating effect_classifier with {cfg.num_effects} classes")  # ADD THIS
            self.effect_classifier = nn.Linear(d, cfg.num_effects)
        else:
            print("DEBUG - NOT creating effect_classifier")  # ADD THIS
        
        if cfg.use_logit_bias:
            self.u_proj = nn.Linear(d, H*Dh, bias=False)  # effect side
            self.v_proj = nn.Linear(d, H*Dh, bias=False)  # cause side
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, emb_cause, emb_action, tok_cause=None, tok_effect=None, target_effect=None):
        B, d = emb_cause.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        
        # Fusion of cause and action
        ctx = self.fuse(torch.cat([emb_cause, emb_action], dim=-1))
        delta_e = self.delta_head(ctx)
        
        # Effect embedding
        if tok_effect is not None:
            pooled_effect = tok_effect.mean(dim=1)
        else:
            pooled_effect = emb_cause
            
        e_shifted = pooled_effect + delta_e

        # === ADD EFFECT PREDICTION HERE ===
        effect_pred = self.effect_head(ctx)
        
        # Effect classification logits
        effect_logits = None
        if hasattr(self, 'effect_classifier'):
            effect_logits = self.effect_classifier(effect_pred)

        # Attention gates
        qg = self.q_gate(ctx).view(B, H, Dh)
        kg = self.k_gate(ctx).view(B, H, Dh)
        q_gate = 1.0 + torch.tanh(qg)
        k_gate = 1.0 + torch.tanh(kg)

        # Logit bias for causal attention
        logit_bias = None
        if self.cfg.use_logit_bias and (tok_cause is not None) and (tok_effect is not None):
            u = self.u_proj(e_shifted).view(B, H, Dh)
            v = self.v_proj(tok_cause).view(B, tok_cause.size(1), H, Dh).transpose(1, 2)
            Nq = tok_effect.size(1)
            u_q = u.unsqueeze(2).expand(B, H, Nq, Dh)
            v_k = v.unsqueeze(2).expand(B, H, Nq, tok_cause.size(1), Dh)
            bias = (u_q.unsqueeze(3) * v_k).sum(dim=-1) / (Dh ** 0.5)
            logit_bias = torch.exp(self.logit_scale) * bias

        return {
            "q_gate": q_gate, 
            "k_gate": k_gate, 
            "e_shifted": e_shifted, 
            "delta_e": delta_e, 
            "logit_bias": logit_bias,
            "effect_pred": effect_pred,
            "effect_logits": effect_logits
        }
    
# class CausalAdapter(nn.Module):
#     def __init__(self, cfg: CausalAdapterConfig):
#         super().__init__()
#         self.cfg = cfg
#         d = cfg.txt_dim
#         H, Dh = cfg.n_heads, cfg.head_dim

#         # Core components only
#         self.fuse = nn.Sequential(
#             nn.LayerNorm(d*2), 
#             nn.Linear(d*2, d), 
#             nn.GELU(), 
#             nn.Dropout(cfg.dropout), 
#             nn.Linear(d, d)
#         )
#         self.delta_head = nn.Sequential(
#             nn.LayerNorm(d), 
#             nn.Linear(d, d)
#         )
#         self.q_gate = nn.Sequential(
#             nn.Linear(d, H*Dh), 
#             nn.SiLU()
#         )
#         self.k_gate = nn.Sequential(
#             nn.Linear(d, H*Dh), 
#             nn.SiLU()
#         )

#         if cfg.use_logit_bias:
#             self.u_proj = nn.Linear(d, H*Dh, bias=False)  # effect side
#             self.v_proj = nn.Linear(d, H*Dh, bias=False)  # cause side
#         self.logit_scale = nn.Parameter(torch.tensor(0.0))

#     def forward(self, emb_cause, emb_action, tok_cause=None, tok_effect=None):
#         B, d = emb_cause.shape
#         H, Dh = self.cfg.n_heads, self.cfg.head_dim
        
#         # Fusion of cause and action
#         ctx = self.fuse(torch.cat([emb_cause, emb_action], dim=-1))
#         delta_e = self.delta_head(ctx)
        
#         # Effect embedding
#         if tok_effect is not None:
#             pooled_effect = tok_effect.mean(dim=1)
#         else:
#             pooled_effect = emb_cause
            
#         e_shifted = pooled_effect + delta_e

#         # Attention gates
#         qg = self.q_gate(ctx).view(B, H, Dh)
#         kg = self.k_gate(ctx).view(B, H, Dh)
#         q_gate = 1.0 + torch.tanh(qg)
#         k_gate = 1.0 + torch.tanh(kg)

#         # Logit bias for causal attention
#         logit_bias = None
#         if self.cfg.use_logit_bias and (tok_cause is not None) and (tok_effect is not None):
#             u = self.u_proj(e_shifted).view(B, H, Dh)
#             v = self.v_proj(tok_cause).view(B, tok_cause.size(1), H, Dh).transpose(1, 2)
#             Nq = tok_effect.size(1)
#             u_q = u.unsqueeze(2).expand(B, H, Nq, Dh)
#             v_k = v.unsqueeze(2).expand(B, H, Nq, tok_cause.size(1), Dh)
#             bias = (u_q.unsqueeze(3) * v_k).sum(dim=-1) / (Dh ** 0.5)
#             logit_bias = torch.exp(self.logit_scale) * bias

#         return {
#             "q_gate": q_gate, 
#             "k_gate": k_gate, 
#             "e_shifted": e_shifted, 
#             "delta_e": delta_e, 
#             "logit_bias": logit_bias
#         }

# class CausalAdapterConfig:
#     def __init__(self, txt_dim=768, d_model=1024, n_heads=8, rank=16, dropout=0.1, use_logit_bias=True, num_effect_candidates=10):
#         self.txt_dim = txt_dim
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.rank = rank
#         self.dropout = dropout
#         self.use_logit_bias = use_logit_bias
#         self.num_effect_candidates = num_effect_candidates
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
#     @property
#     def head_dim(self):
#         return self.d_model // self.n_heads

# class CausalAdapter(nn.Module):
#     def __init__(self, cfg: CausalAdapterConfig):
#         super().__init__()
#         self.cfg = cfg
#         d = cfg.txt_dim
#         H, Dh = cfg.n_heads, cfg.head_dim

#         # Existing components
#         self.fuse = nn.Sequential(
#             nn.LayerNorm(d*2), nn.Linear(d*2, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d)
#         )
#         self.delta_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
#         self.q_gate = nn.Sequential(nn.Linear(d, H*Dh), nn.SiLU())
#         self.k_gate = nn.Sequential(nn.Linear(d, H*Dh), nn.SiLU())

#         if cfg.use_logit_bias:
#             self.u_proj = nn.Linear(d, H*Dh, bias=False)  # effect side
#             self.v_proj = nn.Linear(d, H*Dh, bias=False)  # cause side
#         self.logit_scale = nn.Parameter(torch.tensor(0.0))

#         self.effect_predictor = nn.Sequential(
#             nn.Linear(d, d), 
#             nn.ReLU(),
#         )
        
#         # Effect candidate embeddings
#         self.effect_embeddings = nn.Parameter(torch.randn(cfg.num_effect_candidates, d))
#         nn.init.normal_(self.effect_embeddings, mean=0.0, std=0.02)
        
#         # FIXED: effect_scorer should be Linear(d, 1) not Linear(num_effects, 1)
#         self.effect_scorer = nn.Linear(d, 1)

#     def forward(self, emb_cause, emb_action, tok_cause=None, tok_effect=None):
#         B, d = emb_cause.shape
#         H, Dh = self.cfg.n_heads, self.cfg.head_dim
        
#         # Existing fusion logic
#         ctx = self.fuse(torch.cat([emb_cause, emb_action], dim=-1))
#         delta_e = self.delta_head(ctx)
        
#         if tok_effect is not None:
#             pooled_effect = tok_effect.mean(dim=1)
#         else:
#             pooled_effect = emb_cause
            
#         e_shifted = pooled_effect + delta_e

#         # Existing gate logic
#         qg = self.q_gate(ctx).view(B, H, Dh)
#         kg = self.k_gate(ctx).view(B, H, Dh)
#         q_gate = 1.0 + torch.tanh(qg)
#         k_gate = 1.0 + torch.tanh(kg)

#         # Existing logit bias logic
#         logit_bias = None
#         if self.cfg.use_logit_bias and (tok_cause is not None) and (tok_effect is not None):
#             u = self.u_proj(e_shifted).view(B, H, Dh)
#             v = self.v_proj(tok_cause).view(B, tok_cause.size(1), H, Dh).transpose(1, 2)
#             Nq = tok_effect.size(1)
#             u_q = u.unsqueeze(2).expand(B, H, Nq, Dh)
#             v_k = v.unsqueeze(2).expand(B, H, Nq, tok_cause.size(1), Dh)
#             bias = (u_q.unsqueeze(3) * v_k).sum(dim=-1) / (Dh ** 0.5)
#             logit_bias = torch.exp(self.logit_scale) * bias

#         # FIX: ALWAYS compute effect scores when we have context
#         effect_scores = self.predict_effect_scores(ctx)

#         return {
#             "q_gate": q_gate, 
#             "k_gate": k_gate, 
#             "e_shifted": e_shifted, 
#             "delta_e": delta_e, 
#             "logit_bias": logit_bias,
#             "effect_scores": effect_scores  # ALWAYS return effect scores
#         }

#     def predict_effect_scores(self, context_embedding):
#         """Simplified effect prediction"""
#         B, d = context_embedding.shape
        
#         # Simple transformation
#         effect_context = self.effect_predictor(context_embedding)  # [B, d]
        
#         # Dot product with effect embeddings
#         scores = torch.matmul(effect_context, self.effect_embeddings.T)  # [B, num_effects]
        
#         return scores

#     def predict_top_effects(self, emb_cause, emb_action, top_k=4):
#         """Predict top-k most likely effects"""
#         with torch.no_grad():
#             # Get context from cause and action
#             ctx = self.fuse(torch.cat([emb_cause, emb_action], dim=-1))
            
#             # Get effect scores
#             scores = self.predict_effect_scores(ctx)  # [B, num_effects]
            
#             # Get top-k effects
#             top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            
#             return top_indices, top_scores
        
    
#     @property
#     def effect_vocab(self):
#         return self._effect_vocab

#     @effect_vocab.setter
#     def effect_vocab(self, vocab):
#         self._effect_vocab = vocab