from .op import *
import pickle
from typing import List, Literal, Tuple

class ModelMLP(Layer):
    """
    多层感知机模型 (MLP)，网络结构由 size_list 定义，
    激活函数通过 act_func 决定 (支持 'relu'、'sigmoid'、'tanh')。
    支持 L2 正则化 (L2_regularization)。
    """
    def __init__(
        self,
        size_list: List[int] =[784, 600, 10],
        act_func: Literal['relu', 'leaky_relu', 'elu', 'sigmoid','tanh'] = 'relu',
        lambda_list: List[float] = None
    ):
        super().__init__()  # 初始化父类 Layer
        assert len(size_list) >= 2, "size_list 至少包含输入和输出两个维度"
        self.size_list = size_list
        self.act_func = act_func.lower()
        self.layers = []

        if lambda_list is not None:
            assert len(lambda_list) == len(size_list) - 1, (
                "lambda_list 的长度应为 Linear 层数，即 len(size_list) - 1"
            )

        # 搭建全连接 + 激活层
        for i in range(len(size_list)-1):
            in_dim, out_dim = size_list[i], size_list[i+1]
            # Linear 层
            lin = Linear(
                in_dim=in_dim,
                out_dim=out_dim,
                L2_regularization = (lambda_list is not None),
                L2_regularization_lambda = (lambda_list[i] if lambda_list else 0.0)
            )
            self.layers.append(lin)

            # 如果不是最后一层，则插入激活层
            if i < len(size_list)-2:
                if self.act_func in ['relu','leaky_relu','elu']:
                    self.layers.append(ReLU(type=self.act_func))
                elif self.act_func in ['sigmoid', 'tanh']:
                    self.layers.append(Sigmoid(type=self.act_func))
                else:
                    raise ValueError(f"Unsupported type of activation function: {self.act_func}")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        grad = loss_grad
        # 逆序传递梯度
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        """清空所有可训练层的 grads"""
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()
            elif hasattr(layer, 'grads'):
                for k in layer.grads:
                    layer.grads[k] = None

    def save_model(self, save_path: str):
        """
        将模型参数列表保存为 pickle 文件。
        param_list 格式：
        [ size_list, act_func,
          { 'W':..., 'b':..., 'L2_regularization':bool, 'L2_regularization_lambda':float },  # for each Linear
          ...
        ]
        """
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            # 只保存 Linear 层的可训练参数
            if isinstance(layer, Linear):
                param_list.append({
                    'W': layer.W.copy(),
                    'b': layer.b.copy(),
                    'L2_regularization': layer.L2_regularization,
                    'L2_regularization_lambda': layer.L2_regularization_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, load_path: str):
        """
        从 pickle 文件加载参数，**严格按照保存内容重建模型结构**。
        """
        with open(load_path, 'rb') as f:
            params = pickle.load(f)

        self.size_list = params[0]
        self.act_func = params[1].lower()
        layer_params_list = params[2:]

        self.layers = []  # 严格重建

        for i in range(len(self.size_list) - 1):
            in_dim = self.size_list[i]
            out_dim = self.size_list[i + 1]
            layer_params = layer_params_list[i]

            # 重新构造 Linear 层
            lin = Linear(
                in_dim=in_dim,
                out_dim=out_dim,
                L2_regularization=layer_params['L2_regularization'],
                L2_regularization_lambda=layer_params['L2_regularization_lambda']
            )
            lin.W = layer_params['W']
            lin.b = layer_params['b']
            lin.params['W'] = lin.W
            lin.params['b'] = lin.b
            self.layers.append(lin)

            # 插入激活函数层（除了最后一层）
            if i < len(self.size_list) - 2:
                if self.act_func in ['relu', 'leaky_relu', 'elu']:
                    self.layers.append(ReLU(type=self.act_func))
                elif self.act_func in ['sigmoid', 'tanh']:
                    self.layers.append(Sigmoid(type=self.act_func))
                else:
                    raise ValueError(f"Unsupported activation function: {self.act_func}")

class ModelCNN(Layer):
    """
    通用 CNN 模型：
      - input_shape: (in_channels, H, W)
      - conv_channels: 每个卷积层输出通道列表
      - kernel_sizes:  每个卷积层的卷积核大小列表
      - pool_sizes:    每个卷积层后的池化窗口大小列表
      - fc_sizes:      展平后全连接层的输出维度列表（最后一个通常是类别数）
      - act_func:      激活函数类型 ('relu','leaky_relu','elu','sigmoid','tanh')
      - lambda_conv:   每个卷积层的 L2 正则化系数列表（可选）
      - lambda_fc:     每个全连接层的 L2 正则化系数列表（可选）
    """
    def __init__(
        self,
        input_shape: Tuple[int,int,int]=(1, 28, 28),
        conv_channels: List[int]=[8],
        kernel_sizes:  List[int]=[3, 3],
        pool_sizes:    List[int]=[2, 2],
        fc_sizes:      List[int]=[128, 10],
        act_func:      Literal['relu','leaky_relu','elu','sigmoid','tanh']='relu',
        lambda_conv:   List[float]=[1e-4],
        lambda_fc:     List[float]=[1e-4, 0.0]
    ):
        super().__init__()
        C0, H, W = input_shape
        self.act_func = act_func.lower()

        # 验证激活函数类型
        valid_acts = ['relu','leaky_relu','elu','sigmoid','tanh']
        if self.act_func not in valid_acts:
            raise ValueError(f"Unsupported activation function: {self.act_func}")

        # 默认正则系数为 0
        if lambda_conv is None:
            lambda_conv = [0.0]*len(conv_channels)
        if lambda_fc is None:
            lambda_fc = [0.0]*len(fc_sizes)

        # 1) 卷积 + 激活 + 池化
        self.conv_blocks = []
        curr_C, curr_H, curr_W = C0, H, W
        for out_ch, k, p, lam in zip(conv_channels, kernel_sizes, pool_sizes, lambda_conv):
            pad = k // 2
            conv = conv2D(
                in_channels=curr_C, out_channels=out_ch,
                kernel_size=k, stride=1, padding=pad,
                L2_regularization=(lam>0),
                L2_regularization_lambda=lam
            )
            self.conv_blocks.append(conv)

            # 插入激活层
            if self.act_func in ['relu','leaky_relu','elu']:
                self.conv_blocks.append(ReLU(type=self.act_func))
            else:
                # sigmoid 或 tanh
                self.conv_blocks.append(Sigmoid(type=self.act_func))

            # 池化层
            self.conv_blocks.append(MaxPooling(kernel_size=p, stride=p))

            # 计算下一个 feature map 大小
            h1 = (curr_H + 2*pad - k)//1 + 1
            w1 = (curr_W + 2*pad - k)//1 + 1
            curr_H = (h1 - p)//p + 1
            curr_W = (w1 - p)//p + 1
            curr_C = out_ch

        # 2) 全连接层
        flat_dim = curr_C * curr_H * curr_W
        self.fc_blocks = []
        dims = [flat_dim] + fc_sizes
        for i in range(len(dims)-1):
            lam = lambda_fc[i]
            lin = Linear(
                in_dim=dims[i], out_dim=dims[i+1],
                L2_regularization=(lam>0),
                L2_regularization_lambda=lam
            )
            self.fc_blocks.append(lin)

            # 最后一层不加激活
            if i < len(dims)-2:
                if self.act_func in ['relu','leaky_relu','elu']:
                    self.fc_blocks.append(ReLU(type=self.act_func))
                else:
                    self.fc_blocks.append(Sigmoid(type=self.act_func))

        # 用于 backward 恢复形状
        self._flatten_shape = None

        # 卷积层 + 全连接层
        self.layers = self.conv_blocks + self.fc_blocks

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.conv_blocks:
            out = layer.forward(out)
        # 记录形状并展平
        self._flatten_shape = out.shape
        batch = out.shape[0]
        out = out.reshape(batch, -1)
        for layer in self.fc_blocks:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        grad = loss_grad
        # 全连接反向
        for layer in reversed(self.fc_blocks):
            grad = layer.backward(grad)
        # 恢复到卷积前的四维
        grad = grad.reshape(self._flatten_shape)
        # 卷积反向
        for layer in reversed(self.conv_blocks):
            grad = layer.backward(grad)
        return grad

    def save_model(self, save_path: str):
        """
        先存 meta 再存所有卷积/全连接层的参数.
        """
        meta = {
            'input_shape': self._flatten_shape[1:],  # C,H,W
            'conv_channels': [l.out_channels for l in self.conv_blocks if isinstance(l, conv2D)],
            'kernel_sizes':  [l.kernel_size[0] for l in self.conv_blocks if isinstance(l, conv2D)],
            'pool_sizes':    [l.stride[0]      for l in self.conv_blocks if isinstance(l, MaxPooling)],
            'fc_sizes':      [l.b.shape[1]     for l in self.fc_blocks if isinstance(l, Linear)],
            'act_func':      self.act_func,
            'lambda_conv':   [l.L2_regularization_lambda for l in self.conv_blocks if isinstance(l, conv2D)],
            'lambda_fc':     [l.L2_regularization_lambda for l in self.fc_blocks if isinstance(l, Linear)],
        }
        params = [meta]
        for layer in self.conv_blocks + self.fc_blocks:
            if hasattr(layer, 'W'):
                params.append({
                    'W': layer.W.copy(),
                    'b': layer.b.copy(),
                    'L2_regularization': layer.L2_regularization,
                    'lambda': layer.L2_regularization_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, load_path: str):
        """
        重建网络并加载参数.
        """
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
        meta = params[0]
        # 重构
        self.__init__(
            input_shape=tuple(meta['input_shape']),
            conv_channels=meta['conv_channels'],
            kernel_sizes=meta['kernel_sizes'],
            pool_sizes=meta['pool_sizes'],
            fc_sizes=meta['fc_sizes'],
            act_func=meta['act_func'],
            lambda_conv=meta['lambda_conv'],
            lambda_fc=meta['lambda_fc']
        )
        # 加载权重
        idx = 1
        for layer in self.conv_blocks + self.fc_blocks:
            if hasattr(layer, 'W'):
                p = params[idx]
                layer.W = p['W'].copy()
                layer.b = p['b'].copy()
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.L2_regularization = p['L2_regularization']
                layer.L2_regularization_lambda = p['lambda']
                idx += 1