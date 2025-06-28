import torch
import torch.nn as nn
import numpy as np
from base import AbstractModel


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class BertSASRec(AbstractModel):
    """SASRec implementation from BERT4Rec project"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # Model parameters
        self.item_num = config['item_num']
        self.maxlen = config.get('maxlen', config['max_seq_length'])
        self.hidden_units = config['hidden_size']
        self.num_blocks = config.get('num_blocks', config.get('layer_num', 2))
        self.num_heads = config.get('num_heads', 2)
        self.dropout_rate = config.get('dropout', 0.1)
        self.initializer_range = config.get('initializer_range', 0.02)
        
        # Embeddings
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        
        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
        
        for _ in range(self.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = nn.MultiheadAttention(
                self.hidden_units,
                self.num_heads,
                self.dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Loss function
        self.loss_type = config.get('loss_type', 'ce')
        if self.loss_type == 'ce':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.BCEWithLogitsLoss()
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_embeddings(self, items):
        """Get item embeddings"""
        return self.item_emb(items)
    
    def get_all_embeddings(self, device=None):
        """Get all item embeddings"""
        return self.item_emb.weight.data
    
    def get_current_embeddings(self, device=None):
        """Get embeddings for current item pool"""
        return self.item_emb.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]
    
    def get_representation(self, batch):
        """Get sequence representation"""
        item_seq = batch['item_seqs']
        
        # Embedding
        seqs = self.item_emb(item_seq)
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        # Position embedding
        positions = torch.arange(item_seq.shape[1], device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        # Mask padding tokens
        timeline_mask = (item_seq == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)
        
        # Create attention mask for causality
        tl = seqs.shape[1]
        attention_mask = torch.triu(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device),
            diagonal=1
        )
        
        # Transformer blocks
        for i in range(len(self.attention_layers)):
            # Multi-head attention
            Q = self.attention_layernorms[i](seqs)
            attn_output, _ = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask
            )
            seqs = Q + attn_output
            
            # Feed forward
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        
        outputs = self.last_layernorm(seqs)
        
        # Extract representation at the last valid position
        seq_lengths = batch['seq_lengths'] - 1
        batch_size = outputs.shape[0]
        relevant_outputs = outputs[torch.arange(batch_size), seq_lengths]
        
        return relevant_outputs
    
    def forward(self, batch):
        """Forward pass with loss calculation"""
        state_hidden = self.get_representation(batch)
        
        if self.loss_type == 'ce':
            # Cross entropy loss over all items
            test_item_emb = self.get_all_embeddings(state_hidden.device)
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            loss = self.loss_func(logits, batch['labels'].view(-1))
        else:
            # Binary cross entropy with negative sampling
            labels = batch['labels'].view(-1)
            pos_embs = self.item_emb(labels)
            pos_scores = (state_hidden * pos_embs).sum(dim=-1)
            
            # Negative sampling
            neg_labels = self._generate_negative_samples(batch)
            neg_embs = self.item_emb(neg_labels)
            neg_scores = (state_hidden.unsqueeze(1) * neg_embs).sum(dim=-1)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1
            
            loss = self.loss_func(scores, labels)
        
        return {'loss': loss}
    
    def predict(self, batch, n_return_sequences=1):
        """Predict next items"""
        state_hidden = self.get_representation(batch)
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        
        scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
        scores = scores[:, self.config['select_pool'][0]:self.config['select_pool'][1]]
        
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds
    
    def _generate_negative_samples(self, batch):
        """Generate negative samples"""
        batch_size = batch['labels'].shape[0]
        num_neg = self.config.get('num_negatives', 1)
        
        neg_samples = []
        for i in range(batch_size):
            negatives = torch.randint(
                self.config['select_pool'][0],
                self.config['select_pool'][1],
                (num_neg,),
                device=batch['labels'].device
            )
            # Ensure negatives are different from positive
            while (negatives == batch['labels'][i]).any():
                negatives = torch.randint(
                    self.config['select_pool'][0],
                    self.config['select_pool'][1],
                    (num_neg,),
                    device=batch['labels'].device
                )
            neg_samples.append(negatives)
        
        return torch.stack(neg_samples)