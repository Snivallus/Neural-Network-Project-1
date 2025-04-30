from abc import abstractmethod
import numpy as np
from typing import Literal

class Optimizer:
    def __init__(self, model, init_lr) -> None:
        self.model = model
        self.init_lr = init_lr     # 初始学习率, 保持不变
        self.current_lr = init_lr  # 当前学习率, 由调度器动态调整

    @abstractmethod
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    """
    随机梯度下降优化器, 支持:
      - regular
      - momentum
      - nesterov
      - adam  (RMSprop lr + momentum)
      - nadam (RMSprop lr + Nesterov)
    并集成了线性 Warmup。

    参数
    ----
    model : 模型实例
        需要有 model.layers, 每层需提供 .params (dict)、.grads (dict) 和 .optimizable (bool)
    init_lr : float
        初始学习率
    type: {'regular','momentum','nesterov','adam','nadam'}
        指定优化器类型.
    rho : float, optional
        动量因子 (仅对 momentum/nesterov/adam/nadam 有效)
    beta : float, default=0.9
        RMSprop 的衰减率 (仅对 adam/nadam 有效)
    epsilon : float, default=1e-8
        防止除 0 (仅对 adam/nadam 有效)
    warmup_iter : int, default=0
        线性预热的迭代步数 (0 表示不做预热)
    """
    def __init__(
        self,
        model,
        init_lr: float,
        type: Literal['regular','momentum','nesterov','adam','nadam']='regular',
        rho: float = None,            # 用于 momentum/Nesterov/adam/nadam
        beta: float = 0.9,            # 用于 Adam/Nadam 的 RMSprop 权重
        epsilon: float = 1e-8,        # 防止除零
        warmup_iter: int = 0
    ):
        super().__init__(model, init_lr)
        self.type = type.lower()

        # Warmup 相关
        self.warmup_iter  = warmup_iter
        self.warmup_count = 0

        # —— 为所有需要动量的分支初始化 rho 和 velocity —— 
        if self.type in ['momentum','nesterov','adam','nadam']:
            self.rho = rho if rho is not None else 0.9
            # 每层每个参数都要有一个 velocity
            self.velocity = {
                id(layer): {k: np.zeros_like(v) 
                        for k,v in layer.params.items()}
                for layer in self.model.layers if layer.optimizable
            }
            # —— 为 Adam/Nadam 初始化 RMSprop 累积项 —— 
            if self.type in ['adam','nadam']:
                self.beta    = beta
                self.epsilon = epsilon
                self.sum_grad_sq = { # 保存梯度平方的指数加权平均
                    layer: {k: np.zeros_like(v)
                            for k,v in layer.params.items()}
                    for layer in self.model.layers if layer.optimizable
                }
        # regular 分支不需要 rho
        elif self.type == 'regular':
            if rho is not None:
                print("[WARNING] Parameter rho is redundant for regular SGD.")
            self.rho = None

        else:
            raise ValueError(f"Unsupported optimizer type: {self.type}")

        # 全局 current_lr 仅供 regular/momentum/nesterov 使用
        self.current_lr = init_lr

    def zero_grad(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for k in layer.grads:
                    layer.grads[k].fill(0)

    def step(self):
        # —— 计算本步的 base_lr （线性 Warmup 或外部 scheduler 更新后的 lr） —— 
        if self.warmup_iter > 0 and self.warmup_count < self.warmup_iter:
            self.warmup_count += 1
            base_lr = self.init_lr * (self.warmup_count / self.warmup_iter)
        else:
            base_lr = self.current_lr        
        
        # —— 统一遍历所有可训练层和参数 —— 
        for layer in self.model.layers:
            if not layer.optimizable:
                continue

            for key in layer.params.keys():
                g = layer.grads[key].copy()

                # —— regular SGD —— 
                if self.type == 'regular':
                    layer.params[key] -= base_lr * g
                    continue

                # —— 其它都需要 velocity —— 
                v_prev = self.velocity[id(layer)][key]

                # —— 若是 Adam/Nadam，还要先算 RMSprop 样的 lr_t —— 
                if self.type in ['adam','nadam']:
                    # 更新指数加权平方梯度
                    self.sum_grad_sq[layer][key] = (
                        self.beta * self.sum_grad_sq[layer][key] 
                        + (1 - self.beta) * (g * g)
                    )
                    # per-parameter 学习率
                    lr_t = base_lr / (np.sqrt(self.sum_grad_sq[layer][key]) + self.epsilon)
                else:
                    # momentum/nesterov 用全局 lr
                    lr_t = base_lr

                # —— 计算新的动量 v_t = ρ v_{t-1} - lr_t * g —— 
                v_new = self.rho * v_prev - lr_t * g
                self.velocity[id(layer)][key] = v_new

                # —— 根据 type 做参数更新 —— 
                if self.type in ['momentum','adam']: # 动量修正的 SGD
                    # 直接把动量加上去, θ ← θ + Δθ_k
                    layer.params[key] += v_new

                elif self.type in ['nesterov', 'nadam']: # 近似 Nesterov 动量修正的 SGD
                    # θ_lookahead = θ + ρ Δθ_{k-1}
                    # θ ← θ_lookahead - α g^{(k)}
                    layer.params[key] += self.rho * v_prev - lr_t * g
                else:
                    raise ValueError(f"Unsupported type of SGD optimizer: {self.type}")
                
class DummyLayer:
    def __init__(self):
        self.optimizable = True
        self.params = {'w': np.array([1.0, 2.0])}
        self.grads = {'w': np.array([0.1, 0.2])}

model = type('DummyModel', (), {'layers': [DummyLayer()]})
optimizer = SGD(model, init_lr=0.01, type='adam')
optimizer.zero_grad()
optimizer.step()