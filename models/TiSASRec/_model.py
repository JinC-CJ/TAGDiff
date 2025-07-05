import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from base import AbstractModel


def computeRePos(time_seq, time_span):
    """
    Compute relative time interval matrix for TiSASRec.
    
    Args:
        time_seq: [batch_size, seq_len] - timestamps in days
        time_span: max time interval to consider
    
    Returns:
        time_matrix: [batch_size, seq_len, seq_len] - time interval matrix
    """
    batch_size = time_seq.shape[0]
    seq_len = time_seq.shape[1]
    
    # Create time matrices by expanding dimensions
    time_matrix_i = time_seq.unsqueeze(2).expand(-1, -1, seq_len)  # [B, L, L]
    time_matrix_j = time_seq.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, L]
    
    # Compute time intervals
    time_intervals = time_matrix_i - time_matrix_j  # [B, L, L]
    time_intervals = time_intervals.abs()  # absolute time intervals
    
    # Clamp to time_span
    time_intervals = torch.clamp(time_intervals, 0, time_span).long()
    
    return time_intervals


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, device):
        super(TimeAwareMultiHeadAttention, self).__init__()
        
        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = device

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        """
        Forward pass of time-aware multi-head attention.
        
        Args:
            queries: [batch_size, seq_len, hidden_size]
            keys: [batch_size, seq_len, hidden_size]
            time_mask: [batch_size, seq_len, seq_len] - mask for padding positions
            attn_mask: [seq_len, seq_len] - causal attention mask
            time_matrix_K: [batch_size, seq_len, seq_len, hidden_size]
            time_matrix_V: [batch_size, seq_len, seq_len, hidden_size]
            abs_pos_K: [batch_size, seq_len, hidden_size]
            abs_pos_V: [batch_size, seq_len, hidden_size]
        """
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)
        
        # Get actual sequence length from queries
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        
        # Split heads and concatenate for batch processing
        # [batch_size, seq_len, hidden_size] -> [batch_size * head_num, seq_len, head_size]
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)
        
        # Check and adjust time matrices dimensions if needed
        # time_matrix_K: [B, T, T, D] where T might be different from seq_len
        if time_matrix_K.shape[1] != seq_len or time_matrix_K.shape[2] != seq_len:
            # Truncate time matrices to match actual sequence length
            time_matrix_K = time_matrix_K[:, :seq_len, :seq_len, :]
            time_matrix_V = time_matrix_V[:, :seq_len, :seq_len, :]
        
        # Process time matrices - shape: [batch_size, seq_len, seq_len, hidden_size]
        # Split by heads: [batch_size * head_num, seq_len, seq_len, head_size]
        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        
        # Check and adjust absolute position embeddings if needed
        if abs_pos_K.shape[1] != seq_len:
            abs_pos_K = abs_pos_K[:, :seq_len, :]
            abs_pos_V = abs_pos_V[:, :seq_len, :]
        
        # Process absolute position embeddings
        # [batch_size, seq_len, hidden_size] -> [batch_size * head_num, seq_len, head_size]
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)
        
        # Calculate attention scores
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))  # [B*h, seq_len, seq_len]
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        
        # Add time-aware attention
        # time_matrix_K_: [B*h, seq_len, seq_len, head_size]
        # Q_: [B*h, seq_len, head_size]
        # Using einsum for efficient computation
        time_attn = torch.einsum('blij,blj->bli', time_matrix_K_, Q_)  # [B*h, seq_len, seq_len]
        attn_weights += time_attn
        
        # Scale attention weights
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)
        
        # Handle masks
        # Adjust mask dimensions if needed
        if time_mask.shape[1] != seq_len or time_mask.shape[2] != seq_len:
            time_mask = time_mask[:, :seq_len, :seq_len]
        if attn_mask.shape[0] != seq_len or attn_mask.shape[1] != seq_len:
            attn_mask = attn_mask[:seq_len, :seq_len]
        
        # Expand masks for all heads
        # time_mask: [B, seq_len, seq_len] -> [B*h, seq_len, seq_len]
        # Using repeat_interleave to properly expand along batch dimension
        time_mask_expanded = time_mask.repeat_interleave(self.head_num, dim=0)
        
        # attn_mask: [seq_len, seq_len] -> [B*h, seq_len, seq_len]
        # First add batch dimension, then expand
        attn_mask_expanded = attn_mask.unsqueeze(0).expand(batch_size * self.head_num, -1, -1)
        
        # Apply masks with large negative values
        paddings = torch.ones_like(attn_weights) * (-2**32 + 1)
        attn_weights = torch.where(time_mask_expanded, paddings, attn_weights)
        attn_weights = torch.where(attn_mask_expanded, paddings, attn_weights)
        
        # Apply softmax and dropout
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        outputs = attn_weights.matmul(V_)  # [B*h, seq_len, head_size]
        outputs += attn_weights.matmul(abs_pos_V_)
        
        # Apply time-aware value transformation
        # attn_weights: [B*h, seq_len, seq_len]
        # time_matrix_V_: [B*h, seq_len, seq_len, head_size]
        # Using einsum for efficient computation
        time_output = torch.einsum('bij,bijh->bih', attn_weights, time_matrix_V_)  # [B*h, seq_len, head_size]
        outputs += time_output
        
        # Concatenate heads back
        # [batch_size * head_num, seq_len, head_size] -> [batch_size, seq_len, hidden_size]
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=2)
        
        return outputs


class TiSASRec(AbstractModel):
    def __init__(self, config):
        super(TiSASRec, self).__init__(config)
        
        self.config = config
        self.item_num = config['item_num']
        self.hidden_size = config['hidden_size']
        self.max_len = config.get('max_seq_length', 50)
        self.device = config.get('device', 'cuda')
        
        # TiSASRec specific parameters
        self.num_blocks = config.get('num_blocks', 2)
        self.num_heads = config.get('num_heads', 1)
        self.dropout_rate = config.get('dropout', 0.2)
        self.time_span = config.get('time_span', 256)
        
        # Embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.abs_pos_K_emb = nn.Embedding(self.max_len, self.hidden_size)
        self.abs_pos_V_emb = nn.Embedding(self.max_len, self.hidden_size)
        self.time_matrix_K_emb = nn.Embedding(self.time_span + 1, self.hidden_size)
        self.time_matrix_V_emb = nn.Embedding(self.time_span + 1, self.hidden_size)
        
        # Dropout layers
        self.item_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.abs_pos_K_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.abs_pos_V_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=self.dropout_rate)
        
        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        self.last_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
        
        for _ in range(self.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = TimeAwareMultiHeadAttention(
                self.hidden_size,
                self.num_heads,
                self.dropout_rate,
                self.device
            )
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(self.hidden_size, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass  # Skip parameters that can't be initialized this way
    
    def get_attention_mask(self, item_seq):
        """Generate causal attention mask for the given sequence length."""
        seq_len = item_seq.shape[1]
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        return attention_mask.to(self.device)
    
    def forward(self, batch):
        """
        Forward pass of TiSASRec model.
        
        Args:
            batch: Dictionary containing:
                - item_seqs: [batch_size, seq_len]
                - timestamps_seqs: [batch_size, seq_len] (in days)
                - targets: [batch_size] or [batch_size, seq_len] (optional, for training)
        
        Returns:
            dict: Dictionary containing:
                - loss: Scalar loss value (if targets are provided)
                - logits: [batch_size, seq_len, item_num+1] (always returned)
                - representations: [batch_size, seq_len, hidden_size] (sequence representations)
        """
        # Get sequence representations
        log_feats = self._get_sequence_representations(batch)  # [B, seq_len, D]
        
        # Compute logits
        item_embs = self.item_emb.weight  # [item_num+1, D]
        logits = log_feats.matmul(item_embs.transpose(0, 1))  # [B, seq_len, item_num+1]
        
        # Prepare output dictionary
        outputs = {
            'logits': logits,
            'representations': log_feats
        }
        
        # Compute loss if targets are provided (training mode)
        if 'targets' in batch:
            loss = self._compute_loss(batch, log_feats)
            outputs['loss'] = loss
        elif self.training:
            # In training mode, compute loss using next-item prediction
            loss = self._compute_next_item_loss(batch, log_feats)
            outputs['loss'] = loss
        
        return outputs
    
    def _get_sequence_representations(self, batch):
        """
        Get sequence representations through the transformer layers.
        This is the core forward pass logic.
        """
        item_seq = batch['item_seqs'].to(self.device)
        time_seq = batch['timestamps_seqs'].to(self.device)
        
        batch_size, seq_len = item_seq.shape
        
        # Ensure sequence length doesn't exceed max_len
        if seq_len > self.max_len:
            # Truncate sequences to max_len
            item_seq = item_seq[:, -self.max_len:]
            time_seq = time_seq[:, -self.max_len:]
            seq_len = self.max_len
        
        # Get item embeddings
        seqs = self.item_emb(item_seq)
        seqs = self.item_emb_dropout(seqs) * (item_seq != 0).float().unsqueeze(-1)
        
        # Get position embeddings - only for actual sequence length
        positions = torch.arange(seq_len, device=self.device)
        abs_pos_K = self.abs_pos_K_emb(positions).unsqueeze(0).expand(batch_size, -1, -1)
        abs_pos_V = self.abs_pos_V_emb(positions).unsqueeze(0).expand(batch_size, -1, -1)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)
        
        # Get time matrices - computed for actual sequence length
        time_matrices = computeRePos(time_seq, self.time_span)  # [B, seq_len, seq_len]
        time_matrix_K = self.time_matrix_K_emb(time_matrices)  # [B, seq_len, seq_len, D]
        time_matrix_V = self.time_matrix_V_emb(time_matrices)  # [B, seq_len, seq_len, D]
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)
        
        # Create masks for actual sequence length
        time_mask = (time_seq == 0).unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, seq_len]
        attn_mask = self.get_attention_mask(item_seq)  # [seq_len, seq_len]
        
        # Forward through transformer blocks
        for i in range(self.num_blocks):
            # Self-attention
            seqs = torch.transpose(seqs, 0, 1)  # [seq_len, B, D]
            Q = self.attention_layernorms[i](seqs)
            Q = torch.transpose(Q, 0, 1)  # [B, seq_len, D]
            
            seqs = torch.transpose(seqs, 0, 1)  # [B, seq_len, D]
            mha_outputs = self.attention_layers[i](
                Q, seqs, time_mask, attn_mask,
                time_matrix_K, time_matrix_V,
                abs_pos_K, abs_pos_V
            )
            seqs = mha_outputs + seqs
            
            # Point-wise feed-forward
            seqs = torch.transpose(seqs, 0, 1)  # [seq_len, B, D]
            Q = self.forward_layernorms[i](seqs)
            Q = torch.transpose(Q, 0, 1)  # [B, seq_len, D]
            seqs = self.forward_layers[i](Q)
        
        # Final layer norm
        seqs = torch.transpose(seqs, 0, 1)  # [seq_len, B, D]
        seqs = self.last_layernorm(seqs)
        seqs = torch.transpose(seqs, 0, 1)  # [B, seq_len, D]
        
        return seqs
    
    def get_representation(self, batch):
        """Get sequence representations from the model."""
        outputs = self.forward(batch)
        return outputs.get('representations', outputs.get('logits', None))
    
    def get_embeddings(self, items):
        """Get embeddings for specific items."""
        if not isinstance(items, torch.Tensor):
            items = torch.tensor(items, dtype=torch.long, device=self.device)
        return self.item_emb(items)
    
    def get_all_embeddings(self, device=None):
        """Get all item embeddings."""
        if device is not None:
            return self.item_emb.weight.data.to(device)
        return self.item_emb.weight.data
    
    def predict(self, batch, n_return_sequences=1):
        """
        Predict next items for sequences in batch.
        
        Args:
            batch: Dictionary containing input sequences
            n_return_sequences: Number of sequences to return per input
        
        Returns:
            predictions: [batch_size, n_return_sequences, seq_len]
        """
        # Get model outputs
        outputs = self.forward(batch)
        logits = outputs['logits']  # [B, seq_len, item_num+1]
        
        # For prediction, we typically want the last position's predictions
        last_logits = logits[:, -1, :]  # [B, item_num+1]
        
        # Get top-k predictions
        if n_return_sequences == 1:
            _, predictions = torch.max(last_logits, dim=-1)
            return predictions.unsqueeze(1)
        else:
            _, predictions = torch.topk(last_logits, k=n_return_sequences, dim=-1)
            return predictions
    
    def _compute_loss(self, batch, log_feats):
        """
        Compute training loss with provided targets.
        
        Args:
            batch: Dictionary containing targets
            log_feats: [batch_size, seq_len, hidden_size]
        
        Returns:
            loss: Scalar loss value
        """
        targets = batch['targets'].to(self.device)
        item_embs = self.item_emb.weight  # [item_num+1, D]
        
        # Calculate logits
        logits = log_feats.matmul(item_embs.transpose(0, 1))  # [B, seq_len, item_num+1]
        
        # Reshape for loss calculation
        batch_size, seq_len = logits.shape[:2]
        logits_flat = logits.view(-1, logits.size(-1))
        
        # Handle different target shapes
        if targets.dim() == 1:
            # Targets are [batch_size] - repeat for each position
            targets_flat = targets.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)
        else:
            # Targets are [batch_size, seq_len]
            targets_flat = targets.view(-1)
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)
        
        return loss
    
    def _compute_next_item_loss(self, batch, log_feats):
        """
        Compute next-item prediction loss for training.
        
        Args:
            batch: Dictionary containing item_seqs
            log_feats: [batch_size, seq_len, hidden_size]
        
        Returns:
            loss: Scalar loss value
        """
        item_seq = batch['item_seqs'].to(self.device)
        batch_size, seq_len = item_seq.shape
        
        # Create targets by shifting item_seq left by 1
        # The target for position i is the item at position i+1
        targets = torch.zeros_like(item_seq)
        targets[:, :-1] = item_seq[:, 1:]
        
        # Compute logits
        item_embs = self.item_emb.weight  # [item_num+1, D]
        logits = log_feats.matmul(item_embs.transpose(0, 1))  # [B, seq_len, item_num+1]
        
        # Only compute loss for positions that have valid targets
        # i.e., exclude the last position and padding positions
        valid_mask = (targets != 0).float()
        
        # Flatten for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute cross entropy loss
        loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=0)
        loss_flat = loss_flat.view(batch_size, seq_len)
        
        # Apply mask and compute mean
        masked_loss = loss_flat * valid_mask
        loss = masked_loss.sum() / valid_mask.sum().clamp(min=1.0)
        
        return loss
    
    def compute_loss(self, batch):
        """
        Public interface for computing loss. Called by TAGDiff framework.
        
        Args:
            batch: Dictionary containing input data
        
        Returns:
            loss: Scalar loss value
        """
        outputs = self.forward(batch)
        return outputs.get('loss', torch.tensor(0.0, device=self.device))