"""
MEANTIME Model Adapter for TAGDiff Framework

This module adapts the MEANTIME model to work with the TAGDiff framework
by implementing the AbstractModel interface and preserving MEANTIME's
multi-time embedding mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import AbstractModel
from modules import TransformerEncoder
import math
from typing import Dict, Optional, List, Tuple


# ============= Embedding Components from MEANTIME =============

class TokenEmbedding(nn.Module):
    """Token (item) embedding from MEANTIME"""
    def __init__(self, config):
        super().__init__()
        vocab_size = config.get('item_num', 0) + 1
        hidden = config['hidden_size']
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)

    def forward(self, batch):
        # batch['item_seqs'] shape: [batch_size, seq_len]
        # Return shape: [batch_size, seq_len, hidden_size]
        return self.emb(batch['item_seqs'])


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding from MEANTIME"""
    def __init__(self, config):
        super().__init__()
        max_len = config['max_seq_length']
        hidden = config['hidden_size']
        self.emb = nn.Embedding(max_len, hidden)

    def forward(self, batch):
        batch_size, seq_len = batch['item_seqs'].shape
        positions = torch.arange(
            seq_len, 
            dtype=torch.long, 
            device=batch['item_seqs'].device
        )
        # Get position embeddings [seq_len, hidden_size]
        pos_emb = self.emb(positions)
        # Expand to [batch_size, seq_len, hidden_size]
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_emb


class DayEmbedding(nn.Module):
    """Day-based temporal embedding from MEANTIME"""
    def __init__(self, config):
        super().__init__()
        num_days = config.get('num_days', 732)  # default 2 years
        hidden = config['hidden_size']
        self.emb = nn.Embedding(num_days, hidden, padding_idx=0)

    def forward(self, batch):
        days = batch.get('days_seqs', batch.get('timestamps_seqs', None))
        if days is None:
            # Fallback: convert timestamps to days if not provided
            timestamps = batch['timestamps_seqs']
            days = timestamps // 86400  # Convert seconds to days
        # days shape: [batch_size, seq_len]
        # Return shape: [batch_size, seq_len, hidden_size]
        return self.emb(days)


class ConstantEmbedding(nn.Module):
    """Constant embedding (always returns same value)"""
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden_size']
        self.emb = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.emb, mean=0.0, std=config.get('model_init_range', 0.02))

    def forward(self, batch):
        batch_size, seq_len = batch['item_seqs'].shape
        # Expand to [batch_size, seq_len, hidden_size]
        return self.emb.expand(batch_size, seq_len, -1)


class SinusoidTimeDiffEmbedding(nn.Module):
    """Sinusoidal relative time difference embedding"""
    def __init__(self, config):
        super().__init__()
        self.time_unit_divide = config.get('time_unit_divide', 1)
        self.hidden = config['hidden_size']
        self.freq = config.get('freq', 10000)

    def forward(self, batch):
        t = batch['timestamps_seqs']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float) / self.time_unit_divide

        freq_seq = torch.arange(0, self.hidden, 2.0, dtype=torch.float, device=time_diff.device)
        inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))

        sinusoid_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return pos_emb


class ExponentialTimeDiffEmbedding(nn.Module):
    """Exponential relative time difference embedding"""
    def __init__(self, config):
        super().__init__()
        self.time_unit_divide = config.get('time_unit_divide', 1)
        hidden = config['hidden_size']
        self.linear = nn.Linear(1, hidden)

    def forward(self, batch):
        t = batch['timestamps_seqs']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float) / self.time_unit_divide
        time_diff_abs_log = torch.log1p(time_diff.abs())
        time_diff_sign = time_diff.sign()
        time_diff_log = time_diff_abs_log * time_diff_sign
        outputs = self.linear(time_diff_log.unsqueeze(-1))
        return outputs


class Log1pTimeDiffEmbedding(nn.Module):
    """Log1p relative time difference embedding"""
    def __init__(self, config):
        super().__init__()
        self.time_unit_divide = config.get('time_unit_divide', 1)
        hidden = config['hidden_size']
        self.linear = nn.Linear(1, hidden)

    def forward(self, batch):
        t = batch['timestamps_seqs']
        time_diff = t.unsqueeze(2) - t.unsqueeze(1)
        time_diff = time_diff.to(torch.float) / self.time_unit_divide
        time_diff_log1p = torch.log1p(time_diff.abs()) * time_diff.sign()
        outputs = self.linear(time_diff_log1p.unsqueeze(-1))
        return outputs


# ============= MEANTIME Transformer Components =============

class MeantimeMultiHeadAttention(nn.Module):
    """Multi-head attention with absolute and relative temporal kernels"""
    def __init__(self, config, La, Lr):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.La = La  # number of absolute kernels
        self.Lr = Lr  # number of relative kernels
        self.L = La + Lr
        
        assert self.hidden_size % self.L == 0
        self.head_size = self.hidden_size // self.L
        
        # Linear projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
        self.attention_dropout = nn.Dropout(config['dropout'])

    def forward(self, hidden_states, attention_mask, absolute_embeddings, relative_embeddings):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, values
        Q = self.query(hidden_states)  # [batch_size, seq_len, hidden_size]
        K = self.key(hidden_states)    # [batch_size, seq_len, hidden_size]
        V = self.value(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, L, head_size] -> [batch_size, L, seq_len, head_size]
        Q = Q.view(batch_size, seq_len, self.L, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.L, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.L, self.head_size).transpose(1, 2)
        
        # Now Q, K, V have shape [batch_size, L, seq_len, head_size]
        
        # Initialize attention scores [batch_size, L, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        
        # Add absolute kernel contributions
        if self.La > 0 and absolute_embeddings is not None:
            for i, abs_emb in enumerate(absolute_embeddings):
                # abs_emb shape: [batch_size, seq_len, hidden_size]
                abs_K = self.key(abs_emb)  # [batch_size, seq_len, hidden_size]
                abs_K = abs_K.view(batch_size, seq_len, self.L, self.head_size)  # [batch_size, seq_len, L, head_size]
                
                # Extract the i-th head: [batch_size, seq_len, head_size]
                abs_K_i = abs_K[:, :, i, :]  # [batch_size, seq_len, head_size]
                
                # Q[:, i] has shape [batch_size, seq_len, head_size]
                # Compute attention scores for this absolute kernel
                abs_scores = torch.matmul(Q[:, i], abs_K_i.transpose(-1, -2)) / math.sqrt(self.head_size)
                attention_scores[:, i] += abs_scores
        
        # Add relative kernel contributions
        if self.Lr > 0 and relative_embeddings is not None:
            for i, rel_emb in enumerate(relative_embeddings):
                # rel_emb shape: [batch_size, seq_len, seq_len, hidden_size]
                rel_K = self.key(rel_emb.view(-1, self.hidden_size))  # [(batch_size * seq_len * seq_len), hidden_size]
                rel_K = rel_K.view(batch_size, seq_len, seq_len, self.L, self.head_size)  # [batch_size, seq_len, seq_len, L, head_size]
                
                # Extract the (La+i)-th head: [batch_size, seq_len, seq_len, head_size]
                rel_K_i = rel_K[:, :, :, self.La + i, :]  # [batch_size, seq_len, seq_len, head_size]
                
                # Q[:, self.La + i] has shape [batch_size, seq_len, head_size]
                # We need to compute attention between each query position and all key positions considering relative embeddings
                # rel_scores shape should be [batch_size, seq_len, seq_len]
                Q_i = Q[:, self.La + i]  # [batch_size, seq_len, head_size]
                
                # Compute attention scores with relative keys
                # Q_i: [batch_size, seq_len, head_size] -> [batch_size, seq_len, 1, head_size]
                # rel_K_i: [batch_size, seq_len, seq_len, head_size]
                Q_i_expanded = Q_i.unsqueeze(2)  # [batch_size, seq_len, 1, head_size]
                rel_scores = torch.matmul(Q_i_expanded, rel_K_i.transpose(-1, -2)).squeeze(2)  # [batch_size, seq_len, seq_len]
                
                attention_scores[:, self.La + i] += rel_scores / math.sqrt(self.head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask shape: [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask  # Broadcasting will handle the dimension
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, L, seq_len, seq_len]
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, V)  # [batch_size, L, seq_len, head_size]
        
        # Add absolute value contributions
        if self.La > 0 and absolute_embeddings is not None:
            for i, abs_emb in enumerate(absolute_embeddings):
                abs_V = self.value(abs_emb)  # [batch_size, seq_len, hidden_size]
                abs_V = abs_V.view(batch_size, seq_len, self.L, self.head_size)  # [batch_size, seq_len, L, head_size]
                abs_V_i = abs_V[:, :, i, :]  # [batch_size, seq_len, head_size]
                
                # attention_probs[:, i] has shape [batch_size, seq_len, seq_len]
                # abs_V_i has shape [batch_size, seq_len, head_size]
                abs_context = torch.matmul(attention_probs[:, i], abs_V_i)  # [batch_size, seq_len, head_size]
                context_layer[:, i] += abs_context
        
        # Add relative value contributions
        if self.Lr > 0 and relative_embeddings is not None:
            for i, rel_emb in enumerate(relative_embeddings):
                # rel_emb shape: [batch_size, seq_len, seq_len, hidden_size]
                rel_V = self.value(rel_emb.view(-1, self.hidden_size))  # [(batch_size * seq_len * seq_len), hidden_size]
                rel_V = rel_V.view(batch_size, seq_len, seq_len, self.L, self.head_size)  # [batch_size, seq_len, seq_len, L, head_size]
                
                # Extract the (La+i)-th head
                rel_V_i = rel_V[:, :, :, self.La + i, :]  # [batch_size, seq_len, seq_len, head_size]
                
                # attention_probs[:, self.La + i] has shape [batch_size, seq_len, seq_len]
                # Compute context with relative values
                # For each query position q, sum over all key positions k: sum_k(attention[q,k] * rel_V[q,k])
                att_probs_i = attention_probs[:, self.La + i]  # [batch_size, seq_len, seq_len]
                
                # Use einsum for the computation
                rel_context = torch.einsum('bqk,bqkh->bqh', att_probs_i, rel_V_i)  # [batch_size, seq_len, head_size]
                context_layer[:, self.La + i] += rel_context
        
        # Reshape and project output
        # context_layer: [batch_size, L, seq_len, head_size] -> [batch_size, seq_len, L, head_size] -> [batch_size, seq_len, hidden_size]
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.output(context_layer)
        output = self.dropout(output)
        
        return output


class MeantimeTransformerBlock(nn.Module):
    """Transformer block with MEANTIME attention"""
    def __init__(self, config, La, Lr):
        super().__init__()
        self.attention = MeantimeMultiHeadAttention(config, La, Lr)
        self.layernorm1 = nn.LayerNorm(config['hidden_size'])
        self.layernorm2 = nn.LayerNorm(config['hidden_size'])
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] * 4),
            nn.GELU(),
            nn.Linear(config['hidden_size'] * 4, config['hidden_size']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, hidden_states, attention_mask, absolute_embeddings, relative_embeddings):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, absolute_embeddings, relative_embeddings)
        hidden_states = residual + hidden_states
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============= Main MEANTIME Model =============

class MEANTIME(AbstractModel):
    """
    MEANTIME model adapted for TAGDiff framework.
    
    Implements Multi-time Embeddings for Sequential Recommendation with:
    - Absolute temporal kernels (position, day, constant)
    - Relative temporal kernels (sinusoidal, exponential, log1p time differences)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # Parse kernel types
        self.absolute_kernel_types = config.get('absolute_kernel_types', 'p-d').split('-') if config.get('absolute_kernel_types') else []
        self.relative_kernel_types = config.get('relative_kernel_types', 's-l').split('-') if config.get('relative_kernel_types') else []
        
        # Initialize embeddings
        self.token_embedding = TokenEmbedding(config)
        
        # Absolute embeddings
        self.absolute_kernel_embeddings_list = nn.ModuleList()
        for kernel_type in self.absolute_kernel_types:
            if kernel_type == 'p':  # position
                emb = PositionalEmbedding(config)
            elif kernel_type == 'd':  # day
                emb = DayEmbedding(config)
            elif kernel_type == 'c':  # constant
                emb = ConstantEmbedding(config)
            else:
                raise ValueError(f"Unknown absolute kernel type: {kernel_type}")
            self.absolute_kernel_embeddings_list.append(emb)
        
        # Relative embeddings
        self.relative_kernel_embeddings_list = nn.ModuleList()
        for kernel_type in self.relative_kernel_types:
            if kernel_type == 's':  # sinusoidal time difference
                emb = SinusoidTimeDiffEmbedding(config)
            elif kernel_type == 'e':  # exponential
                emb = ExponentialTimeDiffEmbedding(config)
            elif kernel_type == 'l':  # log1p
                emb = Log1pTimeDiffEmbedding(config)
            else:
                raise ValueError(f"Unknown relative kernel type: {kernel_type}")
            self.relative_kernel_embeddings_list.append(emb)
        
        # Kernel counts
        self.La = len(self.absolute_kernel_embeddings_list)
        self.Lr = len(self.relative_kernel_embeddings_list)
        self.L = self.La + self.Lr
        
        # Sanity check
        assert config['hidden_size'] % self.L == 0, 'hidden_size must be divisible by total number of kernels'
        assert self.L > 0, 'At least one kernel type must be specified'
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            MeantimeTransformerBlock(config, self.La, self.Lr)
            for _ in range(config.get('num_blocks', 2))
        ])
        
        # Output head
        self.output_head = nn.Linear(config['hidden_size'], config['item_num'] + 1)
        if config.get('head_use_ln', True):
            self.head_ln = nn.LayerNorm(config['hidden_size'])
        else:
            self.head_ln = None
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        init_range = self.config.get('model_init_range', 0.02)
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=init_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def get_embeddings(self, items):
        """Get item embeddings (required by AbstractModel)"""
        return self.token_embedding.emb(items)
    
    def get_all_embeddings(self, device=None):
        """Get all item embeddings (required by AbstractModel)"""
        if device:
            return self.token_embedding.emb.weight.data.to(device)
        return self.token_embedding.emb.weight.data
    
    def get_representation(self, batch):
        """
        Get sequence representation using MEANTIME's multi-time attention.
        
        Args:
            batch: Dictionary containing:
                - item_seqs: [batch_size, seq_len]
                - timestamps_seqs: [batch_size, seq_len]
                - seq_lengths: [batch_size]
                
        Returns:
            hidden_states: [batch_size, hidden_size]
        """
        # Get token embeddings
        hidden_states = self.token_embedding(batch)
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask
        seq_mask = (batch['item_seqs'] > 0).float()
        extended_attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get absolute embeddings
        absolute_embeddings = []
        for abs_emb_module in self.absolute_kernel_embeddings_list:
            abs_emb = abs_emb_module(batch)
            abs_emb = self.dropout(abs_emb)
            absolute_embeddings.append(abs_emb)
        
        # Get relative embeddings
        relative_embeddings = []
        for rel_emb_module in self.relative_kernel_embeddings_list:
            rel_emb = rel_emb_module(batch)
            relative_embeddings.append(rel_emb)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                extended_attention_mask,
                absolute_embeddings if len(absolute_embeddings) > 0 else None,
                relative_embeddings if len(relative_embeddings) > 0 else None
            )
        
        # Extract representation from last position
        batch_size = hidden_states.size(0)
        seq_lengths = batch['seq_lengths']
        last_indices = (seq_lengths - 1).clamp(min=0)
        row_indices = torch.arange(batch_size, device=hidden_states.device)
        
        final_hidden = hidden_states[row_indices, last_indices]
        
        return final_hidden
    
    def forward(self, batch):
        """Forward pass for training"""
        # Get sequence representation
        hidden_states = self.get_representation(batch)
        
        # Apply layer norm if configured
        if self.head_ln is not None:
            hidden_states = self.head_ln(hidden_states)
        
        # Get predictions
        logits = self.output_head(hidden_states)
        
        # Calculate loss
        labels = batch['labels'].view(-1)
        loss = self.loss_fn(logits, labels)
        
        return {'loss': loss}
    
    def predict(self, batch, n_return_sequences=1):
        """
        Predict next items for evaluation.
        
        Args:
            batch: Input batch
            n_return_sequences: Number of items to return
            
        Returns:
            predictions: [batch_size, n_return_sequences]
        """
        # Get sequence representation
        hidden_states = self.get_representation(batch)
        
        # Apply layer norm if configured
        if self.head_ln is not None:
            hidden_states = self.head_ln(hidden_states)
        
        # Get all item embeddings
        all_item_embeddings = self.get_all_embeddings(hidden_states.device)
        
        # Compute scores
        scores = torch.matmul(hidden_states, all_item_embeddings.transpose(0, 1))
        
        # Apply selection pool if configured
        if 'select_pool' in self.config:
            start_idx = self.config['select_pool'][0]
            end_idx = self.config['select_pool'][1]
            scores = scores[:, start_idx:end_idx]
            
            # Get top-k predictions
            predictions = scores.topk(n_return_sequences, dim=-1).indices + start_idx
        else:
            # Get top-k predictions
            predictions = scores.topk(n_return_sequences, dim=-1).indices
        
        return predictions


# Optional: Helper function to create MEANTIME model with specific configurations
def create_meantime_model(config):
    """
    Create MEANTIME model with validated configuration.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        MEANTIME model instance
    """
    # Set default values if not provided
    defaults = {
        'hidden_size': 64,
        'num_heads': 2,
        'num_blocks': 2,
        'dropout': 0.2,
        'max_seq_length': 200,
        'absolute_kernel_types': 'p-d',
        'relative_kernel_types': 's-l',
        'time_unit_divide': 1,
        'freq': 10000,
        'model_init_range': 0.02,
        'head_use_ln': True
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return MEANTIME(config)