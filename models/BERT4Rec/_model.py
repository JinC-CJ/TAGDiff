import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from base import AbstractModel
from transformers import BertConfig, BertModel

# python main.py --model=BERT4Rec -s B --ab=iids --partition=split
class BERT4RecModel(nn.Module):
    """Original BERT4Rec implementation adapted from the second project"""
    
    def __init__(self, vocab_size, bert_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std
        
        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=bert_config['hidden_size'],
            padding_idx=padding_idx
        )
        self.transformer_model = BertModel(BertConfig(**bert_config))
        
        if self.add_head:
            self.head = nn.Linear(bert_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight
        
        self.init_weights()
    
    def init_weights(self):
        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()
    
    def forward(self, input_ids, attention_mask):
        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state
        
        if self.add_head:
            outputs = self.head(outputs)
        
        return outputs


class BERT4Rec(AbstractModel):
    """BERT4Rec adapter for PreferDiff framework"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # BERT4Rec specific parameters
        self.mlm_probability = config.get('mlm_probability', 0.2)
        self.mask_token_id = config['item_num'] + 1  # Last token ID for mask
        
        # Initialize BERT4Rec model
        bert_config = {
            'vocab_size': config['item_num'] + 2,  # +1 for padding, +1 for mask
            'hidden_size': config['hidden_size'],
            'num_hidden_layers': config.get('num_hidden_layers', 2),
            'num_attention_heads': config.get('num_heads', 2),
            'intermediate_size': config.get('intermediate_size', config['hidden_size'] * 4),
            'max_position_embeddings': config.get('max_seq_length', 50),
            'hidden_dropout_prob': config.get('dropout', 0.1),
            'attention_probs_dropout_prob': config.get('dropout', 0.1),
        }
        
        self.bert_model = BERT4RecModel(
            vocab_size=config['item_num'] + 2,
            bert_config=bert_config,
            add_head=True,
            padding_idx=0
        )
        
        # Use BERT's embedding layer as item embeddings
        self.item_embeddings = self.bert_model.embed_layer
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    def get_embeddings(self, items):
        """Get item embeddings"""
        return self.item_embeddings(items)
    
    def get_all_embeddings(self, device=None):
        """Get all item embeddings"""
        return self.item_embeddings.weight.data
    
    def get_current_embeddings(self, device=None):
        """Get current item embeddings within select pool"""
        return self.item_embeddings.weight.data[
            self.config['select_pool'][0]:self.config['select_pool'][1]
        ]
    
    def mask_tokens(self, tokens):
        """Apply masked language modeling to input tokens"""
        labels = tokens.clone()
        # Ensure probability_matrix is on the same device as tokens
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=tokens.device)
        
        # We don't mask padding tokens
        padding_mask = tokens.eq(0)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Create the mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # We only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        # Replace masked input tokens with mask_token_id
        tokens[masked_indices] = self.mask_token_id
        
        return tokens, labels
    
    def get_representation(self, batch):
        """Get sequence representation from BERT4Rec"""
        item_seqs = batch['item_seqs']
        device = item_seqs.device
        
        # Apply masking for training
        if self.training:
            masked_item_seqs, labels = self.mask_tokens(item_seqs.clone())
        else:
            # For inference, mask only the last position
            masked_item_seqs = item_seqs.clone()
            seq_lengths = batch['seq_lengths']
            for i in range(len(masked_item_seqs)):
                if seq_lengths[i] > 0:
                    masked_item_seqs[i, seq_lengths[i] - 1] = self.mask_token_id
        
        # Create attention mask
        attention_mask = (masked_item_seqs != 0).float()
        
        # Get BERT outputs
        outputs = self.bert_model(masked_item_seqs, attention_mask)
        
        # Extract the representation of the last item position
        seq_lengths = batch['seq_lengths']
        batch_size = outputs.shape[0]
        sequence_output = outputs[
            torch.arange(batch_size, device=device), 
            torch.clamp(seq_lengths - 1, min=0)
        ]
        
        return sequence_output
    
    def forward(self, batch):
        """Forward pass for training"""
        item_seqs = batch['item_seqs']
        labels = batch['labels'].view(-1)
        
        # Apply masking
        masked_item_seqs, masked_labels = self.mask_tokens(item_seqs.clone())
        
        # For BERT4Rec, we predict the masked positions
        # So we need to set the label at the masked position
        seq_lengths = batch['seq_lengths']
        for i in range(len(masked_labels)):
            if seq_lengths[i] > 0:
                # Find the last masked position
                masked_positions = (masked_item_seqs[i] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(masked_positions) > 0:
                    last_masked_pos = masked_positions[-1]
                    masked_labels[i, last_masked_pos] = labels[i]
        
        # Create attention mask
        attention_mask = (masked_item_seqs != 0).float()
        
        # Get BERT outputs
        outputs = self.bert_model(masked_item_seqs, attention_mask)
        
        # Compute loss only on masked positions
        loss = self.loss_fct(outputs.view(-1, outputs.size(-1)), masked_labels.view(-1))
        
        return {'loss': loss}
    
    def predict(self, batch, n_return_sequences=1):
        """Predict next items"""
        item_seqs = batch['item_seqs']
        seq_lengths = batch['seq_lengths']
        device = item_seqs.device
        
        # Mask the last position for prediction
        masked_item_seqs = item_seqs.clone()
        for i in range(len(masked_item_seqs)):
            if seq_lengths[i] > 0:
                masked_item_seqs[i, seq_lengths[i] - 1] = self.mask_token_id
        
        # Create attention mask
        attention_mask = (masked_item_seqs != 0).float()
        
        # Get predictions
        outputs = self.bert_model(masked_item_seqs, attention_mask)
        
        # Get predictions for the masked positions
        batch_size = outputs.shape[0]
        predictions = outputs[torch.arange(batch_size, device=device), torch.clamp(seq_lengths - 1, min=0)]
        
        # Get scores for items in the select pool
        test_item_emb = self.get_all_embeddings(predictions.device)
        scores = predictions[:, self.config['select_pool'][0]:self.config['select_pool'][1]]
        
        # Get top-k predictions
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        
        return preds