from inspect import Parameter
from typing import Iterator, Tuple
from transformers import GPT2LMHeadModel, GPT2Config


import torch
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn as nn

class GPT2WithMFU(nn.Module):
    def __init__(self, config):
        """
        初始化模型类，包含模型的配置和GPT2LMHeadModel实例。
        
        :param config: GPT2Config配置对象
        """
        super(GPT2WithMFU, self).__init__()
        self.config = config
        self.model = GPT2LMHeadModel(
            config
        )

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self):
        """
        获取模型的参数数量。
        
        :return: 模型的总参数数量
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def forward(self, input_ids, labels=None):
        """
        执行前向传播，并返回模型输出。
        
        :param input_ids: 输入的 ID 张量
        :param labels: 如果存在标签，计算损失
        
        :return: 模型输出
        """
        # 使用 GPT2LMHeadModel 的 forward 方法进行前向传播
        outputs = self.model(input_ids, labels=labels)
        
        return outputs

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估算模型的 FLOPS 利用率 (MFU)，单位为 A100 bfloat16 理论峰值 FLOPS。

        :param fwdbwd_per_iter: 每次迭代的前向和反向传播次数
        :param dt: 每次迭代的时间，单位为秒
        
        :return: 模型的 FLOPS 利用率
        """
        # 获取参数数量
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        
        # 计算每个 token 的 FLOPS 数量
        flops_per_token = 6 * N + 12 * L * H * Q * T
        
        # 计算每次前向+反向传播的 FLOPS 数量
        flops_per_fwdbwd = flops_per_token * T
        
        # 计算每次迭代的总 FLOPS 数量
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # 计算每秒实现的 FLOPS 数量
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        
        # A100 GPU 的 bfloat16 理论峰值 FLOPS：312 TFLOPS
        flops_promised = 312e12  # A100 bfloat16 peak FLOPS is 312 TFLOPS
        
        # 计算内存/计算资源利用率
        mfu = flops_achieved / flops_promised
        return mfu
    
    def custom_named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """
        返回模块参数的迭代器，生成参数名称和参数对象。

        Args:
            prefix (str): 为所有参数名称添加的前缀。
            recurse (bool): 是否递归子模块的参数。
            remove_duplicate (bool): 是否移除重复的参数。

        Yields:
            Tuple[str, Parameter]: 参数名称和参数对象。
        """
        seen = set()  # 记录已经访问过的参数
        
        def get_parameters(module: torch.Module, prefix: str) -> Iterator[Tuple[str, Parameter]]:
            for name, param in module._parameters.items():
                if param is not None:
                    full_name = f"{prefix}.{name}" if prefix else name
                    if not remove_duplicate or param not in seen:
                        seen.add(param)
                        yield full_name, param

        yield from get_parameters(self, prefix)
        
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = f"{prefix}.{name}" if prefix else name
                    yield from module.custom_named_parameters(prefix=submodule_prefix, recurse=recurse)

    
def get_model(config):
    return GPT2WithMFU(config)

if __name__ == '__main__':

    model = get_model(GPT2Config.from_pretrained('gpt2'))