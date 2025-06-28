import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base import AbstractModel
from .Modules_ori import MultiHeadAttention, PositionwiseFeedForward
from .utility import extract_axis_1
from .diffusion import diffusion

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DreamRec(AbstractModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.state_size = config.get('max_seq_length', 10)
        self.hidden_size = config['hidden_size']
        self.item_num = int(config['item_num'])
        self.dropout_rate = config['dropout']
        self.diffuser_type = config.get('diffuser_type', 'mlp1')
        
        # 项目嵌入
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_size,
            padding_idx=0
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        
        # 空嵌入（用于条件控制）
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        
        # 位置嵌入
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.state_size,
            embedding_dim=self.hidden_size
        )
        
        # Transformer层
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.ln_3 = nn.LayerNorm(self.hidden_size)
        self.mh_attn = MultiHeadAttention(
            self.hidden_size, 
            self.hidden_size, 
            config.get('num_heads', 2), 
            self.dropout_rate
        )
        self.feed_forward = PositionwiseFeedForward(
            self.hidden_size, 
            self.hidden_size, 
            self.dropout_rate
        )
        
        # 扩散模型组件
        self.diff = diffusion(config)
        
        # 时间步骤编码
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )
        
        # 扩散去噪网络
        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size*2),
                nn.GELU(),
                nn.Linear(self.hidden_size*2, self.hidden_size)
            )

    def get_embeddings(self, items):
        """获取项目嵌入 - 框架接口要求"""
        return self.item_embeddings(items)
        
    def get_all_embeddings(self, device=None):
        """获取所有项目嵌入 - 框架接口要求"""
        return self.item_embeddings.weight.data
        
    def get_current_embeddings(self, device=None):
        """获取当前候选集的项目嵌入 - 框架接口要求"""
        return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]
    
    def get_representation(self, batch):
        """获取序列表示 - 框架接口要求"""
        # 基于原始cacu_h方法实现，但调整输入格式
        states = batch['item_seqs']
        len_states = batch['seq_lengths']
        
        # 获取项目和位置嵌入
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(inputs_emb.device))
        
        # 应用dropout
        seq = self.emb_dropout(inputs_emb)
        
        # 创建掩码
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(inputs_emb.device)
        seq *= mask
        
        # 应用Transformer层
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        
        # 获取最后一个有效位置的表示
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()
        
        # 添加随机dropout（原始cacu_h中的p参数）
        p = self.config.get('p', 0.1)
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(h.device)
        
        h = h * mask + self.none_embedding(torch.tensor([0]).to(h.device)) * (1-mask)
        
        return h
    
    def forward(self, batch):
        """前向传播函数 - 训练时调用"""
        state_hidden = self.get_representation(batch)
        labels = batch['labels'].view(-1)
        x_start = self.get_embeddings(labels)
        
        # 随机时间步
        n = torch.randint(0, self.config['timesteps'], (labels.shape[0],), device=state_hidden.device).long()
        
        # 计算扩散损失
        loss, _ = self.diff.p_losses(self, x_start, state_hidden, n, loss_type=self.config['loss_type'])
        
        return {'loss': loss}
    
    def denoise(self, x, h, step):
        """去噪函数 - 被扩散模型调用"""
        t = self.step_mlp(step)
        
        if len(x.shape) < 3:
            return self.diffuser(torch.cat((x, h, t), dim=1))
        else:
            B, N, D = x.shape
            x = x.view(-1, D)
            h_expanded = h.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            t_expanded = t.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            input_tensor = torch.cat((x, h_expanded, t_expanded), dim=1)
            return self.diffuser(input_tensor).view(B, N, D)
    
    def denoise_uncon(self, x, step):
        """无条件去噪函数 - 用于引导采样"""
        h = self.none_embedding(torch.tensor([0]).to(x.device))
        h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)
        return self.denoise(x, h, step)
    
    def predict(self, batch, n_return_sequences=1):
        """预测函数 - 框架接口要求"""
        state_hidden = self.get_representation(batch)
        
        # 使用扩散模型采样
        x = self.diff.sample(self.denoise, self.denoise_uncon, state_hidden)
        
        # 计算与所有候选项的相似度
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))[:,
                 self.config['select_pool'][0]: self.config['select_pool'][1]]
        
        # 获取topk结果
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        
        return preds