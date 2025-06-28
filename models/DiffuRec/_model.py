import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from base import AbstractModel
from .diffurec import DiffuRec as DiffuRecModel
import torch as th
import numpy as np
from .step_sample import *

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class DiffuRec(AbstractModel):
    def __init__(self, config):
        super(DiffuRec, self).__init__(config)
        self.hidden_size = config['hidden_size']
        self.item_num = config['item_num']
        self.max_len = config.get('max_seq_length', 50)
        
        # Create model parameters based on config
        args = self._create_diffurec_args(config)
        
        # 创建扩散模型
        self.diffu = DiffuRecModel(args)
        
        # 为Att_Diffuse_model创建组件
        self.item_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
        self.embed_dropout = nn.Dropout(config.get('emb_dropout', 0.3))
        self.position_embeddings = nn.Embedding(self.max_len, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
    
    def _create_diffurec_args(self, config):
        """创建DiffuRec需要的参数对象"""
        class Args:
            def __init__(self):
                pass
        
        args = Args()
        
        # 核心参数转移
        args.hidden_size = config['hidden_size']
        args.dropout = config.get('dropout', 0.1)
        args.num_blocks = config.get('num_blocks', 4)
        args.schedule_sampler_name = config.get('schedule_sampler_name', 'lossaware')
        args.diffusion_steps = config.get('diffusion_steps', 32)
        args.noise_schedule = config.get('noise_schedule', 'trunc_lin')
        args.lambda_uncertainty = config.get('lambda_uncertainty', 0.001)
        args.rescale_timesteps = config.get('rescale_timesteps', True)
        
        return args
    
    def get_embeddings(self, items):
        """获取物品嵌入"""
        return self.item_embeddings(items)
    
    def get_all_embeddings(self, device=None):
        """获取所有物品嵌入"""
        if device:
            return self.item_embeddings.weight.data.to(device)
        return self.item_embeddings.weight.data
    
    def get_current_embeddings(self, device=None):
        """获取当前选择池中的物品嵌入"""
        if device:
            return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]].to(device)
        return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]
    
    def get_representation(self, batch):
        """从序列获取表示"""
        sequence = batch['item_seqs']
        seq_length = sequence.size(1)
        
        # 位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 物品嵌入
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)
        
        # 合并位置编码和物品嵌入
        item_embeddings = item_embeddings + position_embeddings
        item_embeddings = self.LayerNorm(item_embeddings)
        
        # 创建掩码
        mask_seq = (sequence > 0).float()
        
        return item_embeddings, mask_seq
    
    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        """运行扩散预测过程"""
        seq_rep_diffu, item_rep_out, weights, t = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t
    
    def reverse(self, item_rep, noise_x_t, mask_seq):
        """运行扩散反向过程"""
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        return reverse_pre
    
    def loss_diffu_ce(self, rep_diffu, labels):
        """使用交叉熵损失计算扩散损失"""
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))
    
    def diffu_rep_pre(self, rep_diffu):
        """基于扩散表示预测得分"""
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    
    def forward(self, batch):
        """前向传播计算损失"""
        sequence = batch['item_seqs']
        labels = batch['labels']
        
        # 获取表示
        item_embeddings, mask_seq = self.get_representation(batch)
        
        # 获取标签嵌入
        tag_emb = self.item_embeddings(labels.squeeze(-1))
        
        # 扩散预测
        rep_diffu, _, weights, t = self.diffu_pre(item_embeddings, tag_emb, mask_seq)
        
        # 计算损失
        loss_diffu = self.loss_diffu_ce(rep_diffu, labels)
        
        return {'loss': loss_diffu}
    
    def predict(self, batch, n_return_sequences=1):
        """预测下一个物品"""
        sequence = batch['item_seqs']
        
        # 获取表示
        item_embeddings, mask_seq = self.get_representation(batch)
        
        # 生成噪声输入
        noise_x_t = torch.randn_like(item_embeddings[:,-1,:])
        
        # 扩散反向采样
        rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq)
        
        # 计算与候选池中物品的相似度
        test_item_emb = self.get_all_embeddings(rep_diffu.device)
        scores = torch.matmul(rep_diffu, test_item_emb.transpose(0, 1))[:,
                 self.config['select_pool'][0]: self.config['select_pool'][1]]
        
        # 返回topk预测
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds