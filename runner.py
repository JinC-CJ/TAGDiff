import torch
import time
from typing import Union
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler

from base import AbstractModel
from recdata import NormalRecData
from utils import get_config, init_device, init_seed, get_model
from trainer import BaseTrainer


class Runner:
    def __init__(
            self,
            model_name: Union[str, AbstractModel],
            config_dict: dict = None,
            config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            config_file=config_file,
            config_dict=config_dict
        )

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()
        self.accelerator = Accelerator(log_with='wandb')

        self.config['accelerator'] = self.accelerator

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        _ = NormalRecData(self.config).load_data()

        self.recdata = {
            'train': _[0],
            'valid': _[1],
            'test': _[2]
        }
        self.config['select_pool'] = _[3]
        self.config['item_num'] = _[4]
        self.config['eos_token'] = _[4] + 1


        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config)

        print(self.model)
        print(self.model.n_parameters)
        self.trainer = BaseTrainer(self.config, self.model)

    def run(self):
        import random
        train_dataloader = DataLoader(
                self.recdata['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
            )
        val_dataloader = DataLoader(
            self.recdata['valid'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )
        test_dataloader = DataLoader(
            self.recdata['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )
        self.trainer.train(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

        if self.config.get('steps', None) != 0:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
        else:
            pass

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process:
            print(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        if self.config.get('step', None) != 0:
            # 添加测试集推理时间统计
            self.test_with_timing(test_dataloader)
        
            test_results = self.trainer.evaluate(test_dataloader)
            if self.accelerator.is_main_process:
                print("Test Results:", test_results)
                for key in test_results:
                    self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        """
        # 保存embeddings（新增）
        if self.config.get('save_embeddings', False) and self.accelerator.is_main_process:
            import os
            print("Saving embeddings for offline evaluation...")
            embedding_save_dir = os.path.join(
                self.config['ckpt_dir'], 
                f"embeddings_{self.config['model']}_{self.config['sd']}_{self.config['td']}"
            )
            self.trainer.save_embeddings_after_training(test_dataloader, embedding_save_dir)
        """
        if self.accelerator.is_main_process:
            if self.config['save'] is False:
                import os
                if os.path.exists(self.trainer.saved_model_ckpt):
                    os.remove(self.trainer.saved_model_ckpt)
                    print(f"{self.trainer.saved_model_ckpt} has been deleted.")
                else:
                    print(f"{self.trainer.saved_model_ckpt} not found.")
        self.trainer.end()

    def test_with_timing(self, test_dataloader):
        """
        在测试集上评估模型并记录平均推理时间
        """
        print("\n" + "="*50)
        print("Testing model on test dataset with timing...")
        
        self.model.eval()
        
        # 获取评估指标
        all_test_results = self.trainer.evaluate(test_dataloader, split='test')
        
        # 计算每个样本的推理时间
        total_inference_time = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                batch_size = batch['labels'].size(0)
                
                # 批量处理的推理时间
                start_time = time.time()
                _ = self.model.predict(batch, n_return_sequences=self.trainer.evaluator.maxk)
                inference_time = time.time() - start_time
                
                # 计算平均每个样本的时间
                total_inference_time += inference_time
                total_samples += batch_size
        
        # 计算平均推理时间
        avg_inference_time = total_inference_time / total_samples
        
        # 输出测试结果和推理时间
        print(f"Test Results: {all_test_results}")
        print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
        print(f"Total inference time for {total_samples} samples: {total_inference_time:.2f} seconds")
        
        # 计算理论上每秒可处理的用户数
        throughput = 1.0 / avg_inference_time
        print(f"Theoretical throughput: {throughput:.2f} users/second")
        
        # 输出平均训练时间总结
        if hasattr(self.trainer, 'epoch_times') and len(self.trainer.epoch_times) > 0:
            avg_epoch_time = sum(self.trainer.epoch_times) / len(self.trainer.epoch_times)
            print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")
        
        print("="*50 + "\n")
        
        # 返回结果以便可能的后续使用
        return {
            'test_results': all_test_results,
            'avg_inference_time': avg_inference_time,
            'total_samples': total_samples,
            'total_inference_time': total_inference_time
        }