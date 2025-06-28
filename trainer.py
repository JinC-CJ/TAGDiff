import os
import torch
import numpy as np
import time  # Add this import for timing
from tqdm import tqdm
from torch.optim import AdamW
from base import AbstractModel
from transformers.optimization import get_scheduler
from collections import defaultdict, OrderedDict
from utils import get_file_name, get_total_steps
from evaluator import Evaluator


class BaseTrainer(object):
    def __init__(self, config: dict, model: AbstractModel):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config)
        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        self.best_metric = 0
        self.best_epoch = 0
        self.count = 0
        self.checkpoints_deque = []
        # 用于记录所有epoch的训练时间
        self.epoch_times = []

    def train(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        total_n_steps = get_total_steps(self.config, train_dataloader)

        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader)
        self.config.pop('accelerator')
        self.accelerator.init_trackers(
            project_name="PreferDiff",
            config=self.config
        )

        n_epochs = np.ceil(
            total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)
        ).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            # 开始计时
            epoch_start_time = time.time()
            # Training
            self.model.train()
            sum_loss = 0.0
            sum_rec = 0.0
            sum_time = 0.0
            train_pb = tqdm(train_dataloader,
                            total=len(train_dataloader),
                            desc=f"Training - [Epoch {epoch+1}]",
                            disable=True
                            )
            for batch in train_pb:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                rec_loss = outputs.get('rec_loss', torch.tensor(0.0, device=loss.device))
                time_loss = outputs.get('time_loss', torch.tensor(0.0, device=loss.device))

                self.accelerator.backward(loss)
                optimizer.step()

                sum_loss += loss.item()
                sum_rec += rec_loss.item()
                sum_time += time_loss.item()

            avg_loss = sum_loss / len(train_dataloader)
            avg_rec = sum_rec / len(train_dataloader)
            avg_time = sum_time / len(train_dataloader)
            # 结束计时并计算时长
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.accelerator.log({
                "Loss/train_loss": avg_loss,
                "Loss/train_rec_loss": avg_rec,
                "Loss/train_time_loss": avg_time
            }, step=epoch + 1)

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key, val in all_results.items():
                        self.accelerator.log({f"Val_Metric/{key}": val}, step=epoch + 1)

                    # 在输出评估结果的同时输出当前epoch的训练时间
                    print(f"[Epoch {epoch + 1}] Metrics: {all_results}, Training Time: {epoch_time:.2f} seconds")

                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']:
                            unwrapped = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        print(f'[Epoch {epoch + 1}] Saved checkpoint to {self.saved_model_ckpt}')
                else:
                    print('Patience for {} Times'.format(epoch + 1 - best_epoch))

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
                
        # 输出训练完成的总结信息
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        print(f'Average training time per epoch: {avg_epoch_time:.2f} seconds')
        print(f'Total training time: {sum(self.epoch_times):.2f} seconds ({sum(self.epoch_times)/60:.2f} minutes)')

    def evaluate(self, dataloader, split='test'):
        self.model.eval()
        all_results = defaultdict(list)
        eval_pb = tqdm(dataloader,
                       total=len(dataloader),
                       desc=f"Eval - {split}",
                       disable=True
                       )
        for batch in eval_pb:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']:
                    preds = self.model.module.predict(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.predict(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)

        output = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output[key] = torch.cat(all_results[key]).mean().item()
        return output

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()


"""
    def save_embeddings_after_training(self, test_dataloader, save_dir):

        print(f"Loading best model from {self.saved_model_ckpt}")
        if not self.accelerator.is_main_process:
            return

        # 加载最佳模型
        if self.config['use_ddp']:
            self.model.module.load_state_dict(torch.load(self.saved_model_ckpt))
        else:
            self.model.load_state_dict(torch.load(self.saved_model_ckpt))

        self.model.eval()

"""

"""
        # 1. 保存所有 item embeddings
        print("Saving all item embeddings...")
        if self.config['use_ddp']:
            all_item_embeddings = self.model.module.get_all_embeddings(self.accelerator.device)
        else:
            all_item_embeddings = self.model.get_all_embeddings(self.accelerator.device)

        # 2. 收集测试集预测
        print("Collecting test set predictions...")
        predicted_embeddings = []
        true_labels = []
        true_item_embeddings = []
        user_representations = []

        for batch in tqdm(test_dataloader, desc="Processing test set"):
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                # 1) 用户表示
                if self.config['use_ddp']:
                    model_ref = self.model.module
                else:
                    model_ref = self.model

                state_hidden = model_ref.get_representation(batch)

                # 2) 如果启用时间预测，构造 fused feature
                if self.config.get('predict') == 'Y':
                    last_ts_emb = model_ref.get_last_timestamp_embedding(batch)       # [B, D]
                    pred_ts_emb = model_ref.time_embedding_predictor(
                        torch.cat([state_hidden, last_ts_emb], dim=1)
                    )                                                                  # [B, D]
                    state_hidden = model_ref.feature_fusion(
                        torch.cat([state_hidden, pred_ts_emb], dim=1)
                    )                                                                  # [B, D]

                # 3) 通过扩散模型生成预测 embedding
                pred_emb = model_ref.diff.sample(model_ref, state_hidden)            # [B, D]
                true_items = batch['labels'].view(-1)
                true_item_emb = model_ref.get_embeddings(true_items)                 # [B, D]

                predicted_embeddings.append(pred_emb.cpu())
                true_labels.append(true_items.cpu())
                true_item_embeddings.append(true_item_emb.cpu())
                user_representations.append(state_hidden.cpu())

        # 合并并保存
        predicted_embeddings = torch.cat(predicted_embeddings, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        true_item_embeddings = torch.cat(true_item_embeddings, dim=0)
        user_representations = torch.cat(user_representations, dim=0)

        os.makedirs(save_dir, exist_ok=True)
        save_data = {
            'all_item_embeddings': all_item_embeddings.cpu(),
            'predicted_embeddings': predicted_embeddings,
            'true_labels': true_labels,
            'true_item_embeddings': true_item_embeddings,
            'user_representations': user_representations,
            'select_pool': self.config['select_pool'],
            'item_num': self.config['item_num'],
            'config': self.config
        }
        save_path = os.path.join(save_dir, 'test_embeddings.pt')
        torch.save(save_data, save_path)
        print(f"Saved embeddings to {save_path}")
        print(f"Saved data info:")
        print(f"  - All item embeddings shape: {all_item_embeddings.shape}")
        print(f"  - Predicted embeddings shape: {predicted_embeddings.shape}")
        print(f"  - True labels shape: {true_labels.shape}")
        print(f"  - True item embeddings shape: {true_item_embeddings.shape}")
        print(f"  - User representations shape: {user_representations.shape}")
"""