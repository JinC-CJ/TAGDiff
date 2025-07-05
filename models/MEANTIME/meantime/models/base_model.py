import torch.nn as nn
from abc import *

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_init_seed = args.model_init_seed
        self.model_init_range = args.model_init_range

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def init_weights(self):
        # 延迟导入，避免循环导入
        from ..utils import fix_random_seed_as
        fix_random_seed_as(self.args.model_init_seed)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 延迟导入，避免循环导入
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif hasattr(module, 'rel_position_bias'):  # MixedAttention
            for param in [module.rel_position_bias]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif hasattr(module, 'r_bias'):  # RelAttention
            for param in [module.r_bias]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif hasattr(module, 'w'):  # BertDiscriminatorHead
            for param in [module.w]:
                param.data.normal_(mean=0.0, std=self.model_init_range)
        elif hasattr(module, 'bias'):  # BertDotProductPredictionHead
            for param in [module.bias]:
                param.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load(self, path):
        # 延迟导入，避免循环导入
        from ..utils import load_pretrained_weights
        load_pretrained_weights(self, path) 