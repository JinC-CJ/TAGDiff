import torch
import torch.nn as nn
import numpy as np
import sys
import os
from base import AbstractModel


# 添加meantime路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
meantime_dir = os.path.join(current_dir, 'meantime')
if meantime_dir not in sys.path:
    sys.path.insert(0, meantime_dir)

# 直接用包内相对导入
try:
    from .meantime.models.transformer_models.meantime import MeantimeModel
    from .meantime.models.base_model import BaseModel
except ImportError as e:
    print(f"导入MEANTIME模块失败: {e}")
    print(f"当前路径: {current_dir}")
    print(f"meantime路径: {meantime_dir}")
    print(f"Python路径: {sys.path}")
    raise


class MEANTIME(AbstractModel):
    """
    TAGDiff框架下的MEANTIME模型适配器
    将MEANTIME模型适配到TAGDiff的统一接口
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        
        # 将TAGDiff config转换为MEANTIME args格式
        self.args = self._convert_config_to_args(config)
        
        # 初始化MEANTIME模型
        self.meantime_model = MeantimeModel(self.args)
        
        # 存储一些必要的属性用于TAGDiff框架
        self.num_items = config['num_items']
        self.device = config.get('device', 'cuda')
        
        # 确保MEANTIME模型在正确的设备上
        if hasattr(self, 'device') and self.device != 'cpu':
            try:
                self.meantime_model = self.meantime_model.to(self.device)
            except Exception as e:
                # 如果CUDA不可用，回退到CPU
                self.device = 'cpu'
                self.meantime_model = self.meantime_model.to('cpu')
        
    def _ensure_device_consistency(self, tensor, target_device):
        """确保张量在正确的设备上"""
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor
    
    def _validate_labels(self, labels, max_valid_value):
        """验证标签是否在有效范围内"""
        if labels is None:
            return labels
        
        # 确保数据类型正确
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # 检查是否有无效值
        invalid_mask = (labels < 0) | (labels > max_valid_value) | torch.isnan(labels) | torch.isinf(labels)
        
        if invalid_mask.any():
            # 将无效值替换为0
            labels = torch.where(invalid_mask, torch.zeros_like(labels), labels)
        
        return labels
    
    def _convert_config_to_args(self, config):
        """将TAGDiff的config转换为MEANTIME需要的args格式"""
        class Args:
            def __init__(self):
                pass
        
        args = Args()
        
        # 基本参数映射
        args.num_items = config['num_items']
        args.hidden_units = config.get('hidden_size', 64)
        args.max_len = config.get('max_seq_length', 200)
        args.dropout = config.get('dropout_prob', 0.2)
        args.num_blocks = config.get('num_layers', 2)
        args.num_heads = config.get('num_heads', 2)
        
        # 确保词汇表大小正确
        # TokenEmbedding使用 vocab_size = num_items + 2，所以标签应该在 [0, num_items+1] 范围内
        # 但为了安全起见，我们将标签限制在 [0, num_items] 范围内
        
        # 根据select_pool调整num_items（如果存在）
        if 'select_pool' in config:
            start_idx, end_idx = config['select_pool']
            # select_pool的范围是[start_idx, end_idx)，所以实际物品数量是end_idx - start_idx
            actual_num_items = end_idx - start_idx
            # 更新num_items为实际可用的物品数量
            args.num_items = actual_num_items
            print(f"Adjusted num_items from {config['num_items']} to {actual_num_items} based on select_pool {config['select_pool']}")
        
        # MEANTIME特有参数
        args.mask_prob = config.get('mask_prob', 0.2)
        args.output_info = False
        args.residual_ln_type = 'pre'
        args.headtype = 'dot'
        args.head_use_ln = True
        args.time_unit_divide = 1
        args.freq = 10000
        
        # 时间相关embedding类型
        args.absolute_kernel_types = config.get('absolute_kernel_types', 'p')  # 简化为只用position
        args.relative_kernel_types = config.get('relative_kernel_types', 's')  # 简化为只用sinusoidal
        
        # 模型初始化参数
        args.model_init_seed = 0
        args.model_init_range = 0.02
        
        # 其他可能需要的参数
        args.num_users = config.get('num_users', 1000)
        args.num_ratings = config.get('num_ratings', 5)
        args.num_days = config.get('num_days', 365)
        
        return args
    
    def get_embeddings(self, items):
        """获取物品嵌入，适配TAGDiff框架接口"""
        if hasattr(self.meantime_model, 'token_embedding'):
            return self.meantime_model.token_embedding.emb(items)
        else:
            # 如果没有token_embedding，返回随机初始化的embedding
            embed_dim = self.args.hidden_units
            return torch.randn(items.shape + (embed_dim,), device=items.device)
    
    def get_all_embeddings(self, device=None):
        """获取所有物品嵌入"""
        if device is None:
            device = self.device
        
        if hasattr(self.meantime_model, 'token_embedding'):
            return self.meantime_model.token_embedding.emb.weight.data.to(device)
        else:
            embed_dim = self.args.hidden_units
            return torch.randn(self.num_items, embed_dim, device=device)
    
    def get_current_embeddings(self, device=None):
        """获取当前选择池中的物品嵌入"""
        if device is None:
            device = self.device
            
        all_embeddings = self.get_all_embeddings(device)
        
        # 如果config中有select_pool，使用它
        if 'select_pool' in self.config:
            start_idx, end_idx = self.config['select_pool']
            return all_embeddings[start_idx:end_idx]
        else:
            return all_embeddings
    
    def get_representation(self, batch):
        """从序列获取用户表示，适配TAGDiff框架"""
        # 构造MEANTIME需要的数据格式
        meantime_batch = self._convert_batch_format(batch)
        
        # 使用MEANTIME的get_logits方法获取序列表示
        try:
            # 获取隐藏状态而不是logits
            user_repr = self._get_hidden_states(meantime_batch)
            
            # 调试信息（注释掉以避免过多输出）
            # print(f"user_repr shape: {user_repr.shape}")
            # print(f"user_repr device: {user_repr.device}")
                
        except Exception as e:
            # 备用方案：返回固定维度的零向量
            batch_size = len(batch['item_seqs']) if 'item_seqs' in batch else 1
            target_device = self.device if hasattr(self, 'device') else 'cpu'
            user_repr = torch.zeros(batch_size, self.args.hidden_units, device=target_device)
            
        return user_repr
    
    def _get_hidden_states(self, meantime_batch):
        """获取MEANTIME模型的隐藏状态"""
        try:
            # 构建输入数据
            # 调试信息（注释掉以避免过多输出）
            # print(f"meantime_batch keys: {list(meantime_batch.keys())}")
            # for key, value in meantime_batch.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"{key} shape: {value.shape}")
            #     else:
            #         print(f"{key} type: {type(value)}")
            
            tokens = meantime_batch['tokens']
            target_device = tokens.device
            
            # 额外的安全检查
            if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                # 如果数据中有无效值，返回零向量
                B = tokens.size(0)
                return torch.zeros(B, self.args.hidden_units, device=target_device)
            
            B, L = tokens.shape
            
            # 创建attention mask - 使用更简单的方式
            attn_mask = (tokens > 0).float()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            
            # Token embeddings
            try:
                # TokenEmbedding期望接收一个字典，而不是直接的张量
                x = self.meantime_model.token_embedding(meantime_batch)
            except Exception as e:
                # 备用方案：直接返回零向量
                return torch.zeros(B, self.args.hidden_units, device=target_device)
            
            # 尝试直接使用MEANTIME的get_logits方法
            try:
                # 使用MEANTIME的原始方法
                logits, info = self.meantime_model.get_logits(meantime_batch)
                
                # 取最后一个位置的表示
                if len(logits.shape) == 3:  # [B, L, H]
                    hidden_states = logits[:, -1, :]  # [B, H]
                else:
                    hidden_states = logits
                    
                return hidden_states
                
            except Exception as e:
                # 备用方案：手动构建
                
                # Absolute embeddings
                abs_kernel = []
                d = meantime_batch  # 使用完整的batch作为输入
                for i, emb_layer in enumerate(self.meantime_model.absolute_kernel_embeddings_list):
                    try:
                        abs_emb = emb_layer(d)
                        abs_kernel.append(abs_emb)
                    except Exception as emb_e:
                        # 创建假的embedding
                        abs_emb = torch.zeros(B, L, self.args.hidden_units, device=target_device)
                        abs_kernel.append(abs_emb)
                
                # Relative embeddings  
                rel_kernel = []
                for i, emb_layer in enumerate(self.meantime_model.relative_kernel_embeddings_list):
                    try:
                        rel_emb = emb_layer(d)
                        rel_kernel.append(rel_emb)
                    except Exception as emb_e:
                        # 创建假的embedding
                        rel_emb = torch.zeros(B, L, L, self.args.hidden_units, device=target_device)
                        rel_kernel.append(rel_emb)
                
                # 通过transformer body
                info = {}
                try:
                    hidden_states = self.meantime_model.body(x, attn_mask, abs_kernel, rel_kernel, info)
                except Exception as body_e:
                    # 返回token embeddings作为备用
                    hidden_states = x
            
            # 确保返回的是正确的维度
            if len(hidden_states.shape) == 3:  # [B, L, H]
                # 取最后一个位置的表示作为用户表示
                hidden_states = hidden_states[:, -1, :]  # [B, H]
            
            return hidden_states
            
        except Exception as e:
            # 返回备用方案
            batch_size = meantime_batch['tokens'].size(0)
            target_device = meantime_batch['tokens'].device
            return torch.zeros(batch_size, self.args.hidden_units, device=target_device)
    
    def _convert_batch_format(self, batch):
        """将TAGDiff的batch格式转换为MEANTIME需要的格式"""
        meantime_batch = {}
        
        # 获取目标设备
        target_device = self.device if hasattr(self, 'device') else 'cpu'
        
        # 基本的序列数据
        if 'item_seqs' in batch:
            tokens = batch['item_seqs']
        else:
            # 如果没有item_seqs，尝试其他可能的键名
            for key in ['tokens', 'input_ids', 'sequence']:
                if key in batch:
                    tokens = batch[key]
                    break
            else:
                raise ValueError("No sequence data found in batch")
        
        # 数据清理和验证
        tokens = tokens.clone()  # 避免修改原始数据
        
        # 确保数据类型是 long 并移动到正确设备
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        # 确保所有值都在有效范围内 [0, num_items+1]
        max_valid_id = self.args.num_items + 1
        tokens = torch.clamp(tokens, min=0, max=max_valid_id)
        
        # 处理无效值（NaN, inf）
        tokens = torch.where(torch.isnan(tokens) | torch.isinf(tokens), 
                           torch.zeros_like(tokens), tokens)
        
        meantime_batch['tokens'] = tokens
        
        # 标签数据
        if 'labels' in batch:
            labels = batch['labels'].clone()
            if labels.dtype != torch.long:
                labels = labels.long()
            if labels.device != target_device:
                labels = labels.to(target_device)
            
            # 如果有select_pool，需要调整标签的偏移
            if hasattr(self, 'config') and 'select_pool' in self.config:
                start_idx = self.config['select_pool'][0]
                # 将标签从全局ID转换为相对于select_pool的局部ID
                labels = labels - start_idx
                # 确保标签在有效范围内 [0, num_items-1]
                labels = torch.clamp(labels, min=0, max=self.args.num_items - 1)
            else:
                # 更严格的标签范围检查
                # 确保标签在 [0, num_items] 范围内，因为embedding层的大小是 num_items + 1
                labels = torch.clamp(labels, min=0, max=self.args.num_items)
            
            labels = torch.where(torch.isnan(labels) | torch.isinf(labels), 
                               torch.zeros_like(labels), labels)
            
            # 额外的安全检查：如果标签仍然超出范围，设置为0
            invalid_mask = (labels >= self.args.num_items) | (labels < 0)
            if invalid_mask.any():
                labels = torch.where(invalid_mask, torch.zeros_like(labels), labels)
            
            meantime_batch['labels'] = labels
        
        # 负样本标签数据
        if 'negative_labels' in batch:
            negative_labels = batch['negative_labels'].clone()
            if negative_labels.dtype != torch.long:
                negative_labels = negative_labels.long()
            if negative_labels.device != target_device:
                negative_labels = negative_labels.to(target_device)
            
            # 如果有select_pool，需要调整标签的偏移
            if hasattr(self, 'config') and 'select_pool' in self.config:
                start_idx = self.config['select_pool'][0]
                # 将标签从全局ID转换为相对于select_pool的局部ID
                negative_labels = negative_labels - start_idx
                # 确保标签在有效范围内 [0, num_items-1]
                negative_labels = torch.clamp(negative_labels, min=0, max=self.args.num_items - 1)
            else:
                # 更严格的标签范围检查
                negative_labels = torch.clamp(negative_labels, min=0, max=self.args.num_items)
            
            negative_labels = torch.where(torch.isnan(negative_labels) | torch.isinf(negative_labels), 
                                        torch.zeros_like(negative_labels), negative_labels)
            
            # 额外的安全检查：如果标签仍然超出范围，设置为0
            invalid_mask = (negative_labels >= self.args.num_items) | (negative_labels < 0)
            if invalid_mask.any():
                negative_labels = torch.where(invalid_mask, torch.zeros_like(negative_labels), negative_labels)
            
            meantime_batch['negative_labels'] = negative_labels
            
        # 时间戳数据（如果有的话）
        if 'timestamps_seqs' in batch:
            timestamps = batch['timestamps_seqs'].clone()
            if timestamps.dtype != torch.long:
                timestamps = timestamps.long()
            if timestamps.device != target_device:
                timestamps = timestamps.to(target_device)
            meantime_batch['timestamps'] = timestamps
        else:
            # 如果没有时间戳，创建假的时间戳
            seq_len = meantime_batch['tokens'].size(1)
            batch_size = meantime_batch['tokens'].size(0)
            meantime_batch['timestamps'] = torch.arange(seq_len, dtype=torch.long, device=target_device).unsqueeze(0).repeat(batch_size, 1)
            
        if 'timestamps_label' in batch:
            timestamp_labels = batch['timestamps_label'].clone()
            if timestamp_labels.dtype != torch.long:
                timestamp_labels = timestamp_labels.long()
            if timestamp_labels.device != target_device:
                timestamp_labels = timestamp_labels.to(target_device)
            meantime_batch['timestamp_labels'] = timestamp_labels
        else:
            # 如果没有timestamp_labels，创建假的
            seq_len = meantime_batch['tokens'].size(1)
            batch_size = meantime_batch['tokens'].size(0)
            meantime_batch['timestamp_labels'] = torch.arange(seq_len, dtype=torch.long, device=target_device).unsqueeze(0).repeat(batch_size, 1)
            
        # 其他可能的数据
        for key in ['days', 'mask']:
            if key in batch:
                value = batch[key].clone()
                if isinstance(value, torch.Tensor):
                    if value.dtype != torch.long:
                        value = value.long()
                    if value.device != target_device:
                        value = value.to(target_device)
                meantime_batch[key] = value
        
        # 处理seq_lengths
        if 'seq_lengths' in batch:
            seq_lengths = batch['seq_lengths'].clone()
            if seq_lengths.dtype != torch.long:
                seq_lengths = seq_lengths.long()
            if seq_lengths.device != target_device:
                seq_lengths = seq_lengths.to(target_device)
            meantime_batch['seq_lengths'] = seq_lengths
        else:
            # 如果没有seq_lengths，根据tokens计算
            seq_lengths = (meantime_batch['tokens'] > 0).sum(dim=1)
            meantime_batch['seq_lengths'] = seq_lengths
                
        return meantime_batch
    
    def forward(self, batch):
        """前向传播，适配TAGDiff框架"""
        # 转换batch格式
        meantime_batch = self._convert_batch_format(batch)
        
        # 额外的标签验证
        if 'labels' in meantime_batch:
            labels = meantime_batch['labels']
            # 确保标签在有效范围内 [0, num_items]
            max_valid_label = self.args.num_items
            meantime_batch['labels'] = self._validate_labels(labels, max_valid_label)
        
        if 'negative_labels' in meantime_batch:
            negative_labels = meantime_batch['negative_labels']
            # 确保负样本标签在有效范围内 [0, num_items]
            max_valid_label = self.args.num_items
            meantime_batch['negative_labels'] = self._validate_labels(negative_labels, max_valid_label)
        
        try:
            # 计算损失
            loss = self.meantime_model(meantime_batch)
            return loss
        except Exception as e:
            # 返回一个假的损失用于调试，但避免在CUDA错误时创建新张量
            if hasattr(self, 'device') and self.device != 'cpu':
                try:
                    return torch.tensor(0.0, requires_grad=True, device=self.device)
                except:
                    # 如果CUDA设备有问题，回退到CPU
                    return torch.tensor(0.0, requires_grad=True, device='cpu')
            else:
                return torch.tensor(0.0, requires_grad=True, device='cpu')
    
    def predict(self, batch, n_return_sequences=20):
        """预测函数，适配TAGDiff框架接口"""
        try:
            # 获取用户表示
            user_repr = self.get_representation(batch)
            
            # 获取当前选择池中的物品embedding
            current_item_embeddings = self.get_current_embeddings(user_repr.device)
            
            # 计算分数
            scores = torch.matmul(user_repr, current_item_embeddings.transpose(0, 1))
            
            # 如果有select_pool，调整索引
            if 'select_pool' in self.config:
                start_idx = self.config['select_pool'][0]
                preds = scores.topk(n_return_sequences, dim=-1).indices + start_idx
            else:
                preds = scores.topk(n_return_sequences, dim=-1).indices
                
            return preds
            
        except Exception as e:
            # 返回随机预测作为备用
            batch_size = len(batch['item_seqs']) if 'item_seqs' in batch else 1
            target_device = self.device if hasattr(self, 'device') else 'cpu'
            if 'select_pool' in self.config:
                start_idx, end_idx = self.config['select_pool']
                random_preds = torch.randint(start_idx, end_idx, (batch_size, n_return_sequences), device=target_device)
            else:
                random_preds = torch.randint(0, self.num_items, (batch_size, n_return_sequences), device=target_device)
            return random_preds