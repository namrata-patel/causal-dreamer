# src/models/causal_intervention_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalInterventionAttention(nn.Module):
    """
    Implements do-calculus in attention mechanism
    Key idea: When we intervene on cause tokens, we cut their incoming causal dependencies
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard attention projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Intervention-specific projections
        self.intervention_proj = nn.Linear(d_model, d_model)
        self.causal_gate = nn.Linear(d_model * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cause_mask=None, effect_mask=None, intervention_strength=0.0):
        """
        Args:
            x: input tokens [batch_size, seq_len, d_model]
            cause_mask: binary mask for cause tokens [batch_size, seq_len]
            effect_mask: binary mask for effect tokens [batch_size, seq_len] 
            intervention_strength: 0.0 (observational) to 1.0 (full intervention)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.w_q(x)  # [B, N, D]
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal intervention if specified
        if cause_mask is not None and intervention_strength > 0:
            scores = self.apply_causal_intervention(
                scores, Q, K, cause_mask, effect_mask, intervention_strength
            )
        
        # Softmax and attention output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def apply_causal_intervention(self, scores, Q, K, cause_mask, effect_mask, strength):
        """
        Apply do-operator: Cut dependencies TO cause tokens when we intervene
        """
        batch_size, n_heads, seq_len, _ = scores.shape
        
        # Expand masks for multi-head
        cause_mask_expanded = cause_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_heads, -1, seq_len)
        effect_mask_expanded = effect_mask.unsqueeze(1).unsqueeze(2).expand(-1, n_heads, seq_len, -1)
        
        # Create intervention mask: 
        # - effect → cause attention should be reduced (we're intervening on cause)
        # - cause → effect attention should be maintained or enhanced
        intervention_mask = torch.ones_like(scores)
        
        # Reduce attention FROM effects TO causes (cutting causal links to intervened variables)
        effect_to_cause_mask = effect_mask_expanded & cause_mask_expanded
        intervention_mask = intervention_mask - strength * effect_to_cause_mask.float()
        
        # Optional: Enhance attention FROM causes TO effects
        cause_to_effect_mask = cause_mask_expanded & effect_mask_expanded
        intervention_mask = intervention_mask + strength * 0.5 * cause_to_effect_mask.float()
        
        # Apply intervention
        scores = scores * intervention_mask
        
        return scores