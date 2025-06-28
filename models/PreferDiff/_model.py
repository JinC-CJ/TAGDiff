import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from modules import SinusoidalPositionEmbeddings, diagonalize_and_scale, in_batch_negative_sampling_sample, in_batch_negative_sampling
from diffusion import PreferenceDiffusion
from models.SASRec._model import SASRec

class PreferDiff(SASRec):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=config['hidden_size'],
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.diff = PreferenceDiffusion(config=config)
        self.step_nn = nn.Sequential(
            SinusoidalPositionEmbeddings(config['hidden_size']),
            nn.Linear(config['hidden_size'], config['hidden_size'] * 2),
            nn.GELU(),
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
        )
        self.denoise_nn = nn.Sequential(
            nn.Linear(config['hidden_size'] * 3, config['hidden_size'])
        )
        if self.config['predict'] == 'Y':
            # 时间embedding预测模块 - 修改为预测embedding而不是标量

            self.time_embedding_predictor = nn.Sequential(
                nn.Linear(config['hidden_size'] * 2, config['hidden_size'] * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
                nn.LayerNorm(config['hidden_size'])  # 归一化有助于稳定训练
            )
            
            # 用户特征与时间特征融合模块 - 保持不变
            self.feature_fusion = nn.Sequential(
                nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
                nn.ReLU(),
                nn.LayerNorm(config['hidden_size'])
            )

    def get_embeddings(self, items):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings(items)
        else:
            return self.item_embeddings[items].to(items.device)

    def get_all_embeddings(self, device=None):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings.weight.data
        else:
            return self.item_embeddings.to(device)

    def get_current_embeddings(self, device=None):
        if self.config.get('ab', None) == 'iids':
            return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]
        else:
            return self.item_embeddings.to(device)

    def load_item_embeddings(self):
        import pickle
        if self.config.get('ab', None) == 'iids':
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.config['item_num'] + 1,
                embedding_dim=self.config['hidden_size'],
                padding_idx=0
            )
            nn.init.normal_(self.item_embeddings.weight, 0, 1)
        elif self.config.get('ab', None) == 'single':
            single_domain = self.config['sd']
            self.item_embeddings = diagonalize_and_scale(diagonalize_and_scale(torch.tensor(
                pickle.load(
                    open(
                        f"./data/{self.config['source_dict'][single_domain]}/{self.config['embedding']}_item_embedding.pkl",
                        'rb'))
            ).float()))
            random_embedding = torch.randn_like(self.item_embeddings)[0, :].reshape(1, -1)
            self.item_embeddings = torch.cat([random_embedding, self.item_embeddings], dim=0)

        else:
            self.item_embeddings = diagonalize_and_scale(torch.cat([
                diagonalize_and_scale(torch.tensor(
                    pickle.load(
                        open(
                            f"./data/{self.config['source_dict'][domain]}/{self.config['embedding']}_item_embedding.pkl",
                            'rb'))
                ).float())
                for domain in self.config['source_dict']
            ], dim=0))


            random_embedding = torch.randn_like(self.item_embeddings)[0, :].reshape(1, -1)
            self.item_embeddings = torch.cat([random_embedding, self.item_embeddings], dim=0)
    def calcu_h(self, logits, p):
        B, D = logits.shape[0], logits.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(logits.device)
        h = logits * mask + self.none_embedding(torch.tensor([0]).to(logits.device)) * (1 - mask)
        return h

    def denoise_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(x.device))
        h = torch.cat([h.view(1, x.shape[1])] * x.shape[0], dim=0)
        return self.denoise(x, h, step)

    def denoise(self, x, h, step):
        t = self.step_nn(step)
        if len(x.shape) < 3:
            return self.denoise_nn(torch.cat((x, h, t), dim=1))
        else:
            B, N, D = x.shape
            x = x.view(-1, D)
            h_expanded = h.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            t_expanded = t.unsqueeze(1).repeat(1, N, 1).view(-1, D)
            input = torch.cat((x, h_expanded, t_expanded), dim=1)
            return self.denoise_nn(input).view(B, N, D)

    def forward(self, batch):
        if self.config['predict'] == 'Y':
            state_hidden = self.get_representation(batch)
            # 获取历史交互序列中最后的时间戳embedding
            last_timestamp_emb = self.get_last_timestamp_embedding(batch)  # [B, hidden_size]
            
            # 预测目标时间的embedding
            # 输入：用户表示 + 最后时间戳的embedding
            time_pred_input = torch.cat([state_hidden, last_timestamp_emb], dim=1)  # [B, hidden_size*2]
            predicted_time_emb = self.time_embedding_predictor(time_pred_input)  # [B, hidden_size]
            
            # 融合用户特征和预测的时间特征
            fused_feature = self.feature_fusion(
                torch.cat([state_hidden, predicted_time_emb], dim=1)
            )

            # 使用融合后的特征替代原始用户特征
            #enhanced_state_hidden = fused_feature

            # 使用残差连接增强用户特征
            #residual_weight = self.config.get('r', 0.5)
            #enhanced_state_hidden = state_hidden + residual_weight * (fused_feature - state_hidden)
            # 用函数
            enhanced_state_hidden = self.apply_fusion(state_hidden, fused_feature)
            labels_neg = self._generate_negative_samples(batch)
            labels = batch['labels'].view(-1)
            x_start = self.get_embeddings(labels)
            x_start_neg = self.get_embeddings(labels_neg)
            #h = self.calcu_h(state_hidden, self.config['p'])，用结合了target item的时间戳信息的user feature
            h = self.calcu_h(enhanced_state_hidden, self.config['p'])
            n = torch.randint(0, self.config['timesteps'], (labels.shape[0],), device=h.device).long()
            rec_loss, _ = self.diff.p_losses(self, x_start, x_start_neg, h, n, loss_type=self.config['loss_type'])
            
            # 时间embedding预测损失
            # 获取真实的目标时间embedding
            target_time_emb = self.get_target_timestamp_embedding(batch)  # [B, hidden_size]
            
            # 使用余弦相似度损失（类似于PreferDiff中的cosine error）
            time_loss = self.compute_time_embedding_loss(predicted_time_emb, target_time_emb)
            
            # 合并损失
            time_loss_weight = self.config.get('a', 0.2)
            total_loss = rec_loss + time_loss_weight * time_loss
            
            return {'loss': total_loss, 'rec_loss': rec_loss.detach(), 'time_loss': time_loss.detach()}
        else:
            state_hidden = self.get_representation(batch)
            labels_neg = self._generate_negative_samples(batch)
            labels = batch['labels'].view(-1)
            x_start = self.get_embeddings(labels)
            x_start_neg = self.get_embeddings(labels_neg)
            h = self.calcu_h(state_hidden, self.config['p'])
            n = torch.randint(0, self.config['timesteps'], (labels.shape[0],), device=h.device).long()
            loss, _ = self.diff.p_losses(self, x_start, x_start_neg, h, n, loss_type=self.config['loss_type'])
            return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        if self.config['predict'] == 'Y':
            state_hidden = self.get_representation(batch)
            
            # 获取最后时间戳的embedding
            last_timestamp_emb = self.get_last_timestamp_embedding(batch)
            
            # 预测目标时间的embedding
            time_pred_input = torch.cat([state_hidden, last_timestamp_emb], dim=1)
            predicted_time_emb = self.time_embedding_predictor(time_pred_input)
            
            # 融合特征
            fused_feature = self.feature_fusion(
                torch.cat([state_hidden, predicted_time_emb], dim=1)
            )
            # 使用融合后的特征替代原始用户特征
            #enhanced_state_hidden = fused_feature
            # 使用残差连接增强用户特征
            #residual_weight = self.config.get('r', 0.5)
            #enhanced_state_hidden = state_hidden + residual_weight * (fused_feature - state_hidden)
             # 用函数
            enhanced_state_hidden = self.apply_fusion(state_hidden, fused_feature)
            # 使用融合后的特征生成推荐
            x = self.diff.sample(self, enhanced_state_hidden)
            test_item_emb = self.get_all_embeddings(fused_feature.device)
            scores = torch.matmul(x, test_item_emb.transpose(0, 1))[:,
                    self.config['select_pool'][0]: self.config['select_pool'][1]]
            preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
            
            return preds

        else:    
            state_hidden = self.get_representation(batch)
            x = self.diff.sample(self, state_hidden)
            test_item_emb = self.get_all_embeddings(state_hidden.device)
            scores = torch.matmul(x, test_item_emb.transpose(0, 1))[:,
                    self.config['select_pool'][0]: self.config['select_pool'][1]]
            preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
            if self.config['exp_type'] == 'check':
                self.samples = self.get_samples(batch)
                self.target_embedding = self.get_embeddings(torch.tensor([30724]).to(x.device))
                self.predict_embeddings = self.get_embeddings(preds)
            return preds

    def get_samples(self, batch):
        state_hidden = self.get_representation(batch)
        samples = []
        for i in tqdm.tqdm(range(1000)):
            x = self.diff.sample(self, state_hidden)
            samples.append(x.detach().cpu().numpy())
        return samples

    def _generate_negative_samples(self, batch):
        if self.config['sample_func'] == 'batch':
            return in_batch_negative_sampling(batch['labels'])
        elif self.config['sample_func'] == 'random':
            return in_batch_negative_sampling_sample(batch['labels'], self.config['neg_samples'])
        labels_neg = []
        for index in range(len(batch['labels'])):
            import numpy as np
            neg_samples = np.random.choice(range(self.config['select_pool'][0], self.config['select_pool'][1]), size=1,
                                           replace=False)
            neg_samples = neg_samples[neg_samples != batch['labels'][index]]
            labels_neg.append(neg_samples.tolist())
        return torch.LongTensor(labels_neg).to(batch['labels'].device).reshape(-1, 1)

    def get_last_timestamp_embedding(self, batch):
        """获取序列中最后一个时间戳的embedding"""
        # 获取时间戳序列
        if self.config['encoding_type'] == 'sinusoidal':
            timestamps = batch['timestamps_seqs']  # [B, seq_len]
        elif self.config['encoding_type'] in ['RFF', 'Gaussian']:
            timestamps = batch['norm_timestamps_seqs']  # [B, seq_len]
        
        seq_lengths = batch['seq_lengths']
        batch_size = timestamps.shape[0]
        device = timestamps.device
        
        # 获取每个序列的最后一个时间戳
        last_indices = (seq_lengths - 1).clamp(min=0)
        row_indices = torch.arange(batch_size, device=device)
        last_timestamps = timestamps[row_indices, last_indices].unsqueeze(1)  # [B, 1]
        
        # 将时间戳转换为embedding
        last_timestamp_emb = self.temporal_embeddings(last_timestamps).squeeze(1)  # [B, hidden_size]
        return last_timestamp_emb
    
    def get_target_timestamp_embedding(self, batch):
        # 获取时间戳序列和长度信息
        if self.config['encoding_type'] == 'sinusoidal':
            target_timestamp = batch['timestamps_label'].unsqueeze(1)  # [B, 1]
        elif self.config['encoding_type'] in ['RFF', 'Gaussian']:
            target_timestamp = batch['norm_timestamps_label'].unsqueeze(1)  # [B, 1]
        
        # 转换为embedding
        target_time_emb = self.temporal_embeddings(target_timestamp).squeeze(1)  # [B, hidden_size]
        return target_time_emb
    
    def compute_time_embedding_loss(self, pred_emb, target_emb):
        # 与原始代码计算item embedding的损失一致来计算时间损失
        pred_norm = F.normalize(pred_emb, p=2, dim=1)
        gt_norm = F.normalize(target_emb, p=2, dim=1)

            # 计算点积
        dot_product = torch.sum(pred_norm * gt_norm, dim=1)

            # 计算与1之间的均方误差损失
        loss = torch.mean((dot_product - 1) ** 2)
        return loss
    
    def compute_time_embedding_loss11(self, pred_emb, target_emb):
        """计算时间embedding的损失 使用余弦相似度"""
        # 方法1：余弦相似度损失（推荐）
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=1)
        loss = 1 - cos_sim.mean()  # 将相似度转换为损失
        
        # 方法2：MSE损失（备选）
        # loss = F.mse_loss(pred_emb, target_emb)
        
        # 方法3：对比学习损失（更复杂但可能效果更好）
        # 可以考虑将批次内的其他时间embedding作为负样本
        
        return loss
    def apply_fusion(self, state_hidden, fused_feature):
        residual_weight = self.config.get('r', 0.0)
        if residual_weight == 0.0:
            return state_hidden  # 完全使用原始表示
        elif residual_weight == 1.0:
            return fused_feature  # 完全使用融合特征，避免计算残差
        else:
            return state_hidden + residual_weight * (fused_feature - state_hidden)
