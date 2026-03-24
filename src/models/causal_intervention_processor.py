# src/models/causal_intervention_processor.py

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from src.models.causal_intervention_attention import CausalInterventionAttention

class CausalInterventionProcessor:
    """
    Wrapper that integrates CausalInterventionAttention with Stable Diffusion UNet
    """
    
    def __init__(self, causal_attention: CausalInterventionAttention):
        self.causal_attention = causal_attention
        self.cause_mask = None
        self.effect_mask = None
        self.intervention_strength = 0.0
        
    def set_causal_context(self, cause_mask, effect_mask, strength=0.7):
        self.cause_mask = cause_mask
        self.effect_mask = effect_mask
        self.intervention_strength = strength
        
    def clear_context(self):
        self.cause_mask = None
        self.effect_mask = None
        self.intervention_strength = 0.0
        
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, **kwargs):
        # For cross-attention (text conditioning)
        if encoder_hidden_states is not None:
            return self.cross_attention_forward(attn, hidden_states, encoder_hidden_states, **kwargs)
        # For self-attention
        else:
            return self.self_attention_forward(attn, hidden_states, **kwargs)
    
    def cross_attention_forward(self, attn, hidden_states, encoder_hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Prepare attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        # Get queries from hidden states, keys/values from encoder
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Apply causal intervention if context is set
        if self.cause_mask is not None and self.intervention_strength > 0:
            # For cross-attention, apply intervention to text tokens
            x_combined = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            output = self.causal_attention(
                x_combined, 
                cause_mask=torch.cat([torch.zeros_like(self.cause_mask), self.cause_mask], dim=1),
                effect_mask=torch.cat([torch.ones_like(self.effect_mask), self.effect_mask], dim=1),
                intervention_strength=self.intervention_strength
            )
            # Split back
            hidden_states_out = output[:, :hidden_states.shape[1]]
        else:
            # Standard cross-attention
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states_out = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        
        return hidden_states_out
    
    def self_attention_forward(self, attn, hidden_states, **kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Prepare attention mask
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            
            # Self-attention: Q, K, V all come from hidden_states
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
            # Standard self-attention (no intervention for now)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states_out = attn.batch_to_head_dim(hidden_states)
            
            # Linear projection
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            
            return hidden_states_out
    
    def update_strength(self, s: float):
        self._strength = float(s)
