from abc import abstractmethod
import numpy as np
import bisect

class Scheduler():
    def __init__(self, optimizer) -> None:
        if optimizer.type in ['adam', 'nadam']:
            print(
                f"[WARNING] Detected optimizer type '{optimizer.type.capitalize()}'. "
                "This type uses an internal RMSprop-style adaptive learning rate, "
                "so an external scheduler is unnecessary and will be ignored."
            )
        self.optimizer = optimizer
        self.step_count = 0
        self.model = optimizer.model  # 以便访问模型的梯度等信息
        self.init_lr = optimizer.init_lr  # 保存初始学习率

    def in_warmup(self): # 检查 optimizer 是否处于预热期间
        return (
            self.optimizer.warmup_iter > 0 and 
            self.optimizer.warmup_count < self.optimizer.warmup_iter
        )

    @abstractmethod
    def step():
        raise NotImplementedError

#========================================================================
class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        if self.in_warmup():
            return  # 如果处于 Warmup 期间，则不更新 step_count 和 current_lr
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.current_lr *= self.gamma
            self.step_count = 0

#========================================================================
class PiecewiseConstantLR(Scheduler):
    def __init__(self, optimizer, milestones, gamma_list):
        """
        milestones : list of int
            分段区间端点（左闭右开）, 例如 [100, 200] 表示区间 [0,100), [100,200), [200, ∞)
        gamma_list : list of float
            每个区间内的学习率缩放系数，长度需为 len(milestones) + 1
        """
        super().__init__(optimizer)
        assert len(gamma_list) == len(milestones) + 1, "gamma_list 长度应为 milestones 长度 +1"
        self.milestones = sorted(milestones)
        self.gamma_list = gamma_list
        self.init_lr = optimizer.init_lr  # 保存初始学习率

    def step(self):
        if self.optimizer.warmup_iter > 0 and self.optimizer.warmup_count < self.optimizer.warmup_iter:
            return  # 如果处于 Warmup 期间，则不更新 step_count 和 current_lr
        stage = bisect.bisect_right(self.milestones, self.step_count) # 二分查找对应的小区间
        self.optimizer.current_lr = self.init_lr * self.gamma_list[stage] # 更新当前学习率
        self.step_count += 1

class InverseTimeLR(Scheduler):
    def __init__(self, optimizer, beta):
        super().__init__(optimizer)
        self.beta = beta
        self.init_lr = optimizer.init_lr  # 保存初始学习率

    def step(self):
        if self.in_warmup():
            return  # 如果处于 Warmup 期间，则不更新 step_count 和 current_lr
        self.optimizer.current_lr = self.init_lr / (1 + self.beta * self.step_count)
        self.step_count += 1

class ExponentialLR(Scheduler):
    def __init__(self, optimizer, beta=0.999, use_exp=False):
        """
        use_exp : bool
            True 时使用 α = α0 * exp(-beta * step)
            False 时使用 α = α0 * beta^step
        """
        super().__init__(optimizer)
        self.beta = beta
        self.use_exp = use_exp
        self.init_lr = optimizer.init_lr

    def step(self):
        if self.optimizer.warmup_iter > 0 and self.optimizer.warmup_count < self.optimizer.warmup_iter:
            return  # 如果处于 Warmup 期间，则不更新 step_count 和 current_lr
        if self.use_exp:
            self.optimizer.current_lr = self.init_lr * np.exp(-self.beta * self.step_count)
        else:
            self.optimizer.current_lr = self.init_lr * (self.beta ** self.step_count)
        self.step_count += 1

class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, max_iter):
        super().__init__(optimizer)
        self.max_iter = max_iter
        self.init_lr = optimizer.init_lr

    def step(self):
        if self.in_warmup():
            return  # 如果处于 Warmup 期间，则不更新 step_count 和 current_lr
        if self.step_count >= self.max_iter:
            # 超过最大迭代次数后保持一个很小的值
            self.optimizer.current_lr = 1e-2 * self.init_lr # TODO: 不过可能不够灵活
        else:
            cosine = np.cos(np.pi * self.step_count / self.max_iter)
            self.optimizer.current_lr = 0.5 * self.init_lr * (1 + cosine)
        self.step_count += 1

#========================================================================
# 非典型的 "全局" AdaGrad 学习率衰减, 效果不一定好
# class AdaGradScheduler(Scheduler):
#     def __init__(self, optimizer, epsilon: float = 1e-8):
#         super().__init__(optimizer)
#         self.epsilon = epsilon
#         self.grad_sq_sum = 0.0 # 累计所有参数的梯度平方和

#     def step(self):
#         self.step_count += 1
#         # 将当前所有层的梯度平方累积到全局和
#         total = 0.0
#         for layer in self.optimizer.model.layers:
#             if not layer.optimizable:
#                 continue
#             for g in layer.grads.values():
#                 total += np.sum(g * g)
#         self.grad_sq_sum += total

#         # 更新全局学习率
#         # α_t = α₀ / (sqrt(G_t) + ε)
#         self.optimizer.current_lr = self.optimizer.init_lr / (np.sqrt(self.grad_sq_sum) + self.epsilon)

# 非典型的 "全局" RMSprop 学习率衰减, 效果不一定好
# class RMSpropScheduler(Scheduler):
#     def __init__(self, optimizer, beta: float = 0.9, epsilon: float = 1e-8):
#         super().__init__(optimizer)
#         self.beta     = beta
#         self.epsilon  = epsilon
#         self.accum = 0.0 # 累计所有参数的梯度平方和的指数加权平均

#     def step(self):
#         self.step_count += 1
#         # 将当前所有层的梯度平方加权后累积到全局和
#         total, count = 0.0, 0
#         for layer in self.optimizer.model.layers:
#             if not layer.optimizable:
#                 continue
#             for g in layer.grads.values():
#                 total += np.sum(g * g)
#                 count += g.size
#         mean_sq = total / max(1, count)

#         # 指数加权平均： E[g²]_t = β·E[g²]_{t-1} + (1−β)·mean_sq
#         self.accum = self.beta * self.accum + (1 - self.beta) * mean_sq

#         # 更新学习率
#         # α_t = α₀ / (sqrt(E[g²]_t) + ε)
#         self.optimizer.current_lr = self.optimizer.init_lr / (np.sqrt(self.accum) + self.epsilon)