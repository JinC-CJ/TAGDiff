import torch
import torch.nn as nn
import torch.nn.functional as F
from base import AbstractModel


class GRU4Rec(AbstractModel):
    """GRU4Rec model adapter for PreferDiff framework.
    
    This class wraps a GRU-based sequential recommendation model
    to be compatible with the PreferDiff framework interface.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # Initialize item embeddings
        self.item_embeddings = nn.Embedding(
            num_embeddings=config['item_num'] + 1,
            embedding_dim=config['hidden_size'],
            padding_idx=0
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        
        # Initialize GRU layer
        self.gru = nn.GRU(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            num_layers=config.get('num_layers', 1),
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config.get('num_layers', 1) > 1 else 0
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(config['hidden_size'], config['item_num'] + 1)
        
        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # Loss function
        self.loss_func = nn.CrossEntropyLoss()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def get_embeddings(self, items):
        """Get item embeddings.
        
        Args:
            items: Tensor of item IDs
            
        Returns:
            Item embeddings tensor
        """
        if isinstance(items, torch.Tensor):
            device = items.device
            return self.item_embeddings(items).to(device)
        else:
            # If items is not a tensor, convert it first
            items_tensor = torch.tensor(items, dtype=torch.long)
            return self.item_embeddings(items_tensor)
    
    def get_all_embeddings(self, device=None):
        """Get all item embeddings."""
        if device is not None:
            return self.item_embeddings.weight.data.to(device)
        return self.item_embeddings.weight.data
    
    def get_current_embeddings(self, device=None):
        """Get embeddings for items in the current selection pool."""
        start_idx = self.config['select_pool'][0]
        end_idx = self.config['select_pool'][1]
        embeddings = self.item_embeddings.weight.data[start_idx:end_idx]
        if device is not None:
            return embeddings.to(device)
        return embeddings
    
    def get_representation(self, batch):
        """Get sequence representation using GRU.
        
        Args:
            batch: Dictionary containing 'item_seqs' and other fields
            
        Returns:
            Hidden representation of the sequence
        """
        # Get item sequences
        item_seqs = batch['item_seqs']
        seq_lengths = batch['seq_lengths']
        
        # Get embeddings
        inputs_emb = self.get_embeddings(item_seqs)
        
        # Create mask for padding
        mask = (item_seqs != 0).float().unsqueeze(-1).to(inputs_emb.device)
        
        # Apply mask to embeddings
        inputs_emb = inputs_emb * mask
        
        # Apply dropout to embeddings
        inputs_emb = self.dropout(inputs_emb)
        
        # Pass through GRU
        gru_output, _ = self.gru(inputs_emb)
        
        # Get the last relevant output for each sequence
        # We need to get the output at position (seq_length - 1) for each sequence
        batch_size = item_seqs.shape[0]
        device = item_seqs.device
        
        # Create indices for gathering
        batch_indices = torch.arange(batch_size, device=device)
        time_indices = (seq_lengths - 1).to(device)
        
        # Gather the last relevant hidden states
        state_hidden = gru_output[batch_indices, time_indices]
        
        return state_hidden
    
    def forward(self, batch):
        """Forward pass for training.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary with loss value
        """
        # Get sequence representation
        state_hidden = self.get_representation(batch)
        
        # Project to vocabulary size
        logits = self.output_layer(state_hidden)
        
        # Get labels
        labels = batch['labels'].view(-1)
        
        # Calculate loss
        loss = self.loss_func(logits, labels)
        
        return {'loss': loss}
    
    def predict(self, batch, n_return_sequences=1):
        """Predict next items.
        
        Args:
            batch: Dictionary containing input data
            n_return_sequences: Number of top items to return
            
        Returns:
            Predicted item IDs
        """
        # Get sequence representation
        state_hidden = self.get_representation(batch)
        
        # Ensure state_hidden is 2D
        if state_hidden.dim() == 1:
            state_hidden = state_hidden.unsqueeze(0)
        
        # Get device
        device = state_hidden.device
        
        # Get all item embeddings on the same device
        test_item_emb = self.get_all_embeddings(device)
        
        # Calculate scores using the output layer
        # This maintains consistency with the training forward pass
        logits = self.output_layer(state_hidden)
        
        # Get scores for items in the selection pool
        start_idx = self.config['select_pool'][0]
        end_idx = self.config['select_pool'][1]
        scores = logits[:, start_idx:end_idx]
        
        # Get top-k predictions
        _, indices = torch.topk(scores, k=n_return_sequences, dim=-1)
        preds = indices + start_idx
        
        return preds
    
    def _generate_negative_samples(self, batch):
        """Generate negative samples for training.
        
        This method is included for compatibility but not used in standard GRU4Rec.
        """
        import numpy as np
        
        labels_neg = []
        for index in range(len(batch['labels'])):
            start_idx = self.config['select_pool'][0]
            end_idx = self.config['select_pool'][1]
            neg_samples = np.random.choice(
                range(start_idx, end_idx), 
                size=1,
                replace=False
            )
            # Ensure negative sample is different from positive label
            while neg_samples[0] == batch['labels'][index].item():
                neg_samples = np.random.choice(
                    range(start_idx, end_idx), 
                    size=1,
                    replace=False
                )
            labels_neg.append(neg_samples.tolist())
        
        return torch.LongTensor(labels_neg).to(batch['labels'].device).reshape(-1, 1)