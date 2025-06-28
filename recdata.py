import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler

class SequenceDataset(Dataset):
    def __init__(self, config, sequences, indexes, timestamps=None):
        self.sequences = sequences
        self.indexes = indexes
        self.config = config
        self.timestamps = timestamps  # 添加时间戳数据属性
        self.index_to_samples = defaultdict(list)
        for i, index in enumerate(indexes):
            self.index_to_samples[index].append(i)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        index = self.indexes[idx]
        item_seq = seq[:-1]
        labels = seq[-1]
        seq_length = len(item_seq)
        padding_length = self.config['max_seq_length'] - len(item_seq)
        if padding_length > 0:
            item_seq = item_seq + [0] * padding_length  # 在后面填充0
                
        result = {
            'item_seqs': torch.tensor(item_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_lengths': seq_length,
            'idx': index
        }
        
        # 如果有时间戳数据，添加到结果中，并分开处理序列时间戳和标签时间戳
        if self.timestamps is not None:
            if idx >= len(self.timestamps):
                print(f"Warning: No timestamp found for sequence idx {idx}")
                # 创建零时间戳
                time_seq = [0] * len(item_seq)
                time_label = 0
            else:
                time_data = self.timestamps[idx]
                if len(time_data) < len(seq):
                    print(f"Warning: Timestamp sequence is shorter than item sequence for idx {idx}")
                    # 如果时间戳序列比物品序列短，用0填充
                    if len(time_data) == 0:
                        time_seq = [0] * len(item_seq)
                        time_label = 0
                    else:
                        time_seq = time_data[:-1] if len(time_data) > 1 else [0]
                        time_label = time_data[-1] if time_data else 0
                else:
                    time_seq = time_data[:-1]  # 序列时间戳（除了最后一个）
                    time_label = time_data[-1]  # 标签时间戳（最后一个）
            
            # 对序列时间戳进行填充
            if len(time_seq) < len(item_seq):
                time_seq = time_seq + [0] * (len(item_seq) - len(time_seq))
            elif len(time_seq) > len(item_seq):
                time_seq = time_seq[:len(item_seq)]
            
            # 获取配置中的最小时间戳（已经是天数）
            min_timestamp = self.config.get('min_timestamp', 0)
            
            # 时间戳转换为整数天：先转换为天数，然后取整
            # 对于原始Unix时间戳（秒），除以86400转换为天，然后减去最小天数，最后取整
            time_seq_days = []
            for t in time_seq:
                if t > 0:
                    days = int(t / 86400.0)  # 先转换为天数并取整
                    relative_days = days - int(min_timestamp)  # 减去最小天数（也要确保是整数）
                    time_seq_days.append(relative_days)
                else:
                    time_seq_days.append(0)
            
            # 同样处理标签时间戳
            if time_label > 0:
                time_label_days = int(time_label / 86400.0) - int(min_timestamp)
            else:
                time_label_days = 0
            
            # 存储整数天的时间戳
            result['timestamps_seqs'] = torch.tensor(time_seq_days, dtype=torch.long)
            result['timestamps_label'] = torch.tensor(time_label_days, dtype=torch.long)
            
            # 归一化时间戳（保持浮点数）
            range_timestamp = self.config.get('range_timestamp', 1)
            # 使用整数天进行归一化，但结果保持为浮点数
            norm_time_seq = [float(t) / range_timestamp if range_timestamp > 0 else 0.0 for t in time_seq_days]
            norm_time_label = float(time_label_days) / range_timestamp if range_timestamp > 0 else 0.0
            
            result['norm_timestamps_seqs'] = torch.tensor(norm_time_seq, dtype=torch.float)
            result['norm_timestamps_label'] = torch.tensor(norm_time_label, dtype=torch.float)
        
        return result


class NormalRecData:
    def __init__(self, config: dict):
        self.config = config
        
    def load_data(self):
        from pathlib import Path
        # 完整的 source_dict
        full_source_dict = {
            'B': 'Beauty',
            'T': 'Toys_and_Games',
            'O': 'Sports_and_Outdoors',
            'M1': 'MovieLens-1M',
            'M2': 'MovieLens-10M',
        }
        
        # 如果是单一数据集模式，只使用指定的数据集
        if self.config.get('single_domain_mode', False):
            single_key = self.config['sd']
            if single_key not in full_source_dict:
                raise ValueError(f"Invalid domain key: {single_key}")
            
            # 创建只包含单一数据集的 source_dict
            source_dict = {single_key: full_source_dict[single_key]}
            # 为了保持数据处理的一致性，使用固定的索引映射
            domain_index_map = {single_key: 0}
        else:
            # 使用完整的 source_dict
            source_dict = full_source_dict
            # 创建固定的域索引映射，确保一致性
            domain_index_map = {'C': 0, 'O': 1, 'T': 2}
        
        self.config['source_dict'] = source_dict

        def read_timestamp_from_file(domain, mode=''):
            if self.config['partition'] == 'split':
                base_path = Path('811data/')
            elif self.config['partition'] == 'LOO':
                base_path = Path('LOOdata/')
            file_path = base_path / source_dict[domain] / '{}data_timestamps.txt'.format(mode)
            
            if not file_path.exists():
                print(f"Timestamp file not found: {file_path}")
                return None
                
            with file_path.open('r') as file:
                timestamp_seqs = [list(map(int, line.split())) for line in file]
            
            return timestamp_seqs
        
        def read_data_from_file(domain, mode=''):
            if self.config['partition'] == 'split':
                base_path = Path('811data/')
            elif self.config['partition'] == 'LOO':
                base_path = Path('LOOdata/')
            file_path = base_path / source_dict[domain] / '{}data.txt'.format(mode)
            with file_path.open('r') as file:
                item_seqs = [list(map(int, line.split())) for line in file]
            if mode == '':
                flat_list = [item for sublist in item_seqs for item in sublist]
                import numpy as np
                item_num = np.max(flat_list)
                return item_seqs, item_num
            else:
                return item_seqs

        total_item_num = 0
        all_data = []
        train_data = []
        valid_data = []
        test_data = []
        train_index = []
        valid_index = []
        test_index = []
                
        # 时间戳数据变量
        train_timestamps = []
        valid_timestamps = []
        test_timestamps = []
        
        select_pool = None
        
        # 使用排序后的键来确保处理顺序的一致性
        sorted_keys = sorted(source_dict.keys())
        
        for key in sorted_keys:
            if self.config.get('ab', None) == 'single':
                if key != self.config['sd']:
                    continue
                        
            # 获取固定的域索引
            if self.config.get('single_domain_mode', False):
                index = domain_index_map.get(key, 0)
            else:
                index = domain_index_map.get(key, sorted_keys.index(key))
                    
            tmp_item_seqs, cur_item_num = read_data_from_file(key)
            all_data.extend([[item + total_item_num for item in sublist] for sublist in tmp_item_seqs])
            
            tmp_train_item_seqs, tmp_valid_item_seqs, tmp_test_item_seqs = (
                read_data_from_file(key, mode='train_'),
                read_data_from_file(key, mode='valid_'),
                read_data_from_file(key, mode='test_')
            )
            
            # 读取时间戳数据
            tmp_train_timestamps = read_timestamp_from_file(key, mode='train_')
            tmp_valid_timestamps = read_timestamp_from_file(key, mode='valid_')
            tmp_test_timestamps = read_timestamp_from_file(key, mode='test_')
            
            # 应用ID偏移
            tmp_train_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_train_item_seqs]
            tmp_valid_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_valid_item_seqs]
            tmp_test_item_seqs = [[item + total_item_num for item in sublist] for sublist in tmp_test_item_seqs]

            tmp_train_idx = [index] * len(tmp_train_item_seqs)
            tmp_valid_idx = [index] * len(tmp_valid_item_seqs)
            tmp_test_idx = [index] * len(tmp_test_item_seqs)
            
            if key == self.config['td']:
                valid_data.extend(tmp_valid_item_seqs)
                valid_index.extend(tmp_valid_idx)
                test_data.extend(tmp_test_item_seqs)
                test_index.extend(tmp_test_idx)
                                
                # 添加时间戳数据
                if tmp_valid_timestamps:
                    valid_timestamps.extend(tmp_valid_timestamps)
                if tmp_test_timestamps:
                    test_timestamps.extend(tmp_test_timestamps)

            if key in self.config['sd']:
                train_data.extend(tmp_train_item_seqs)
                train_index.extend(tmp_train_idx)
                        
                # 添加时间戳数据
                if tmp_train_timestamps:
                    train_timestamps.extend(tmp_train_timestamps)
                
            if key == self.config['td']:
                select_pool = [total_item_num + 1, total_item_num + cur_item_num + 1]
                
            total_item_num += cur_item_num

        # ✅ 在循环结束后统一计算时间戳范围（基于整数天）
        all_timestamps = []
        if train_timestamps:
            for seq in train_timestamps:
                all_timestamps.extend([t for t in seq if t > 0])
        if valid_timestamps:
            for seq in valid_timestamps:
                all_timestamps.extend([t for t in seq if t > 0])
        if test_timestamps:
            for seq in test_timestamps:
                all_timestamps.extend([t for t in seq if t > 0])
        
        # 计算最小和最大时间戳（转换为整数天）
        if all_timestamps:
            # 将所有时间戳转换为天数并取整
            all_days = [int(t / 86400.0) for t in all_timestamps]
            min_timestamp = min(all_days)  # 最小天数（整数）
            max_timestamp = max(all_days)  # 最大天数（整数）
            range_timestamp = max_timestamp - min_timestamp  # 天数范围（整数）
        else:
            min_timestamp = 0
            max_timestamp = 0
            range_timestamp = 1  # 避免除以0
        
        print(f"Final Timestamp Range: {range_timestamp} days, Minimum Day: {min_timestamp}, Maximum Day: {max_timestamp}")
        
        # 将时间戳信息添加到配置中（存储的是天数，不是秒数）
        self.config['min_timestamp'] = min_timestamp
        self.config['max_timestamp'] = max_timestamp
        self.config['range_timestamp'] = range_timestamp

        # 创建数据集
        train_dataset = SequenceDataset(self.config, train_data, train_index, train_timestamps)
        valid_dataset = SequenceDataset(self.config, valid_data, valid_index, valid_timestamps)
        test_dataset = SequenceDataset(self.config, test_data, test_index, test_timestamps)
        
        return (train_dataset, valid_dataset, test_dataset, select_pool, total_item_num)