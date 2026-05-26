import torch.nn as nn
from torch import Tensor


# register_forward_hook 是 PyTorch 中用于向网络模型中的指定层注册一个回调函数的方法
# 在模型进行前向传播时，当指定层的前向传播结束时，该回调函数就会被自动调用，并且将该层的输入、输出和模块本身的引用作为参数传入
# 这个回调函数可以用于在网络前向传播过程中获取输出每个子模块的输出结果、中间特征图、记录梯度信息等操作
class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layers, _, outputs: print(f"{layers.__name__}：{outputs.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
