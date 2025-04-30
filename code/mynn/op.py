from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Literal

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

#=====================================================================================================
class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, 
                 initialize_method=np.random.normal, 
                 L2_regularization=False, L2_regularization_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.
        self.params = {'W' : self.W, 'b' : self.b}
        self.L2_regularization = L2_regularization # whether using L2 regularization
        self.L2_regularization_lambda = L2_regularization_lambda # control the intensity of L2 regularization
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        Weight: [in_dim, out_dim]
        Bias: [1, out_dim]
        output: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grads : np.ndarray):
        """
        input: [batch_size, out_dim] the grads passed by the next layer.
        output: [batch_size, in_dim] the grads to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        X = self.input
        # Compute gradient for W: X^T @ grads, with possible L2 regularization
        # Gradiant ∂L/∂W = ∂Z/∂W · ∂L/∂Z = Xᵀ · grads
        dW = np.dot(X.T, grads)
        if self.L2_regularization: # L2 regularization
            dW += self.L2_regularization_lambda * self.W

        # Compute gradient for b: sum over the batch dimension
        # Gradiant ∂L/∂b = ∂Z/∂b · ∂L/∂Z = 1ᵀ · grads
        db = np.sum(grads, axis=0, keepdims=True)
        
        # Compute gradient for the previous layer: grads @ W^T
        # Gradiant ∂L/∂X = ∂L/∂Z · ∂Z/∂X = grads · Wᵀ
        grads_prev = np.dot(grads, self.W.T)

        # Save gradients
        self.grads['W'] = dW
        self.grads['b'] = db

        return grads_prev

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

#=====================================================================================================
# ReLU 类激活函数
class ReLU(Layer):
    """
    ReLU 型激活函数, 支持 'relu'、'leaky_relu'、'elu' 三种类型.
    type : {'relu', 'leaky_relu', 'elu'}
        指定激活函数类型.
    alpha : float
        对于 LeakyReLU 和 ELU, 在输入 <=0 时的缩放系数 alpha, 默认为 0.01.
    """
    def __init__(
        self, 
        type: Literal['relu','leaky_relu','elu']='relu', 
        alpha: float = None
    ) -> None:
        super().__init__()
        self.type = type.lower()
        self.input = None      # 用于 backward 时保存 forward 的输入
        self.optimizable = False
        if alpha is not None:
            if self.type == 'relu':
                print("[WARNING] Parameter alpha is redundant.")
                self.alpha = None
            elif self.type in ['leaky_relu', 'elu']:
                self.alpha = alpha
            else:
                raise ValueError(f"Unsupported type of ReLU-type activation: {self.type}")
        else:
            if self.type == 'relu':
                self.alpha = None
            elif self.type == 'leaky_relu':
                self.alpha = 0.01
            elif self.type == 'elu':
                self.alpha = 1
            else:
                raise ValueError(f"Unsupported type of ReLU-type activation: {self.type}")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向计算：根据 self.type 选择不同公式.
        relu:       f(x)=max(0, x)
        leaky_relu: f(x)=max(x, alpha*x)
        elu:        f(x)= x                 if x>0
                          alpha*(exp(x)-1)  if x<=0
        """
        self.input = X
        if self.type == 'relu':
            return np.where(X > 0, X, 0)
        elif self.type == 'leaky_relu':
            return np.where(X > 0, X, self.alpha * X)
        elif self.type == 'elu':
            return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
        else:
            raise ValueError(f"Unsupported type of ReLU-type activation: {self.type}")

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """
        反向传播：根据 self.type 计算导数，再乘以上游梯度 grads = ∂L/∂f.        
        relu':        1 (x>0), 0 (x<=0)
        leaky_relu':  1 (x>0), alpha (x<=0)
        elu':         1 (x>0), alpha*exp(x) (x<=0)
        """
        assert self.input is not None, "Must call forward() before backward()"
        assert grads.shape == self.input.shape
        
        X = self.input
        if self.type == 'relu':
            grad_input = np.where(X > 0, 1, 0)
        elif self.type == 'leaky_relu':
            grad_input = np.where(X > 0, 1, self.alpha)
        elif self.type == 'elu':
            # 对于 ELU，d/dx [alpha*(exp(x)-1)] = alpha * exp(x)
            grad_input = np.where(X > 0, 1, self.alpha * np.exp(X))
        else:
            raise ValueError(f"Unsupported type of ReLU-type activation: {self.type}")
        
        return grad_input * grads

# Sigmoid 型激活函数
class Sigmoid(Layer):
    """
    Sigmoid 型激活函数, 支持 'sigmoid' 和 'tanh' 两种类型.
    type : {'sigmoid', 'tanh'}
        指定激活函数类型.
    threshold: 输入裁剪阈值
    """

    def __init__(
        self, 
        type: Literal['sigmoid','tanh']='sigmoid',
        threshold: float = 20.0
    ) -> None:
        super().__init__()
        self.type = type.lower()
        self.threshold = threshold
        self.output = None     # 保存 output 以便 backward 使用
        self.optimizable = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向计算：根据 self.type 选择不同公式.
        sigmoid:    σ(x) = 1 / (1 + exp(-x))
        tanh:       tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
                            = np.tanh(x)
        """
        # 为了数值稳定性 (防止上溢或下溢), 先对 X 做裁剪
        X = np.clip(X, -self.threshold, self.threshold)
        if self.type == 'sigmoid':
            self.output = 1.0 / (1.0 + np.exp(-X))
        elif self.type == 'tanh':
            self.output = np.tanh(X)
        else:
            raise ValueError(f"Unsupported type of Sigmoid-type activation: {self.type}")
        return self.output

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """
        反向传播：根据 self.type 计算导数，再乘以上游梯度 grads = ∂L/∂f.    
        'sigmoid':     dσ/dx = σ(x) * (1 - σ(x))
        'tanh':        dtanh(x)/dx  = 1 - tanh(x)^2
        """
        assert self.output is not None, "Must call forward() before backward()"
        assert grads.shape == self.output.shape

        if self.type == 'sigmoid':
            grad_input = (self.output * (1.0 - self.output))
        elif self.type == 'tanh':
            grad_input = (1.0 - np.square(self.output))
        else:
            raise ValueError(f"Unsupported type of Sigmoid-type activation: {self.type}")

        return grad_input * grads

#=====================================================================================================
class CrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True # 默认使用 Softmax 激活函数
        self.grads = None
        self.probs = None
        self.labels = None

    def __call__(self, logits, labels):
        return self.forward(logits, labels)
    
    def softmax(self, X):
        """
        Numerically stable softmax function.
        
        Parameters:
            X: np.ndarray of shape (batch_size, num_classes)
        
        Returns:
            probs: np.ndarray of shape (batch_size, num_classes), where each row sums to 1
        """
        # Subtract max for numerical stability
        x_max = np.max(X, axis=1, keepdims=True)
        x_exp = np.exp(X - x_max)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        return x_exp / partition
    
    def forward(self, logits, labels):
        """
        logits: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        batch_size = logits.shape[0]
        
        if self.has_softmax:
            self.probs = self.softmax(logits)
        else:
            self.probs = logits
        
        # Compute cross-entropy loss with numerical stability
        correct_log_probs = np.log(self.probs[np.arange(batch_size), labels] + 1e-8)
        loss = -np.mean(correct_log_probs)
        
        return loss

    def backward(self):
        """
        Backward pass to compute gradient of the loss w.r.t. inputs to softmax (logits).
        """
        batch_size = self.probs.shape[0]
        num_classes = self.probs.shape[1]
        labels = self.labels
        
        # Create one-hot encoded labels
        one_hot = np.zeros((batch_size, num_classes))
        one_hot[np.arange(batch_size), labels] = 1.0
        
        if self.has_softmax:
            # Gradient when softmax is included: (probs - one_hot) / batch_size
            self.grads = (self.probs - one_hot) / batch_size
        else:
            # Gradient when softmax is excluded: (-one_hot / (probs + 1e-8)) / batch_size
            self.grads = (-one_hot / (self.probs + 1e-8)) / batch_size
        
        # Propagate the gradient backward through the model
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        """
        Cancel internal softmax computation.
        """
        self.has_softmax = False
        return self

#=====================================================================================================
class conv2D(Layer):
    """
    The 2D convolutional layer.
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=0, 
                 initialize_method=np.random.normal, 
                 L2_regularization=False, L2_regularization_lambda=1e-8,
                 dropout_rate=0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.L2_regularization = L2_regularization
        self.L2_regularization_lambda = L2_regularization_lambda
        self.dropout_rate = dropout_rate  # Dropout rate

        # Initialize weights and bias
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.params = {'W': self.W, 'b': self.b}
        
        # Storage for backward pass
        self.input = None
        self.X_shape = None
        self.X_padded = None
        self.X_cols = None
        self.dropout_mask = None  # For dropout implementation

    def im2col(self, X):
        """
        Convert X_padded [N, C, H, W] to columns [N*height_new*width_new, C*k_h*k_w]
        """
        N, C, H, W = X.shape
        k_h, k_w   = self.kernel_size
        stride_h, stride_w = self.stride

        height_new = (H - k_h)//stride_h + 1
        width_new  = (W - k_w)//stride_w + 1

        shape = (N, C, height_new, width_new, k_h, k_w)
        strides = (
            X.strides[0],
            X.strides[1],
            stride_h * X.strides[2],
            stride_w * X.strides[3],
            X.strides[2],
            X.strides[3]
        )
        windows = as_strided(X, shape=shape, strides=strides)
        cols = windows.reshape(N, C, height_new*width_new, k_h*k_w)
        cols = cols.transpose(0,2,1,3).reshape(-1, C*k_h*k_w)
        return cols, height_new, width_new

    def col2im(self, cols, X_padded_shape, height_new, width_new):
        """
        Convert cols [N*height_new*width_new, C*k_h*k_w] back to dX_padded [N, C, H, W].
        """
        N, C, H, W = X_padded_shape
        k_h, k_w   = self.kernel_size
        stride_h, stride_w = self.stride

        cols = cols.reshape(N, height_new*width_new, C, k_h*k_w)
        cols = cols.transpose(0,2,1,3).reshape(N, C, height_new, width_new, k_h, k_w)

        dX_padded = np.zeros((N, C, H, W), dtype=cols.dtype)
        for i in range(k_h):
            for j in range(k_w):
                dX_padded[
                    :,
                    :,
                    i : i + stride_h*height_new : stride_h,
                    j : j + stride_w*width_new : stride_w
                ] += cols[..., i, j]
        return dX_padded

    def forward(self, X, training=True):
        """
        input X: [batch, in_channels, height, width]
        W : [1, out_channels, in_channels, k_h, k_w]
        """
        self.input = X
        self.X_shape = X.shape
        batch_size, in_channels, height, width = X.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Pad input
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        self.X_padded = X_padded

        # im2col
        X_cols, height_new, width_new = self.im2col(X_padded)
        self.X_cols = X_cols
        
        # convolution as matrix multiply
        W_col = self.W.reshape(self.out_channels, -1)
        output = X_cols.dot(W_col.T) + self.b  # (N*height_new*width_new, out_channels)
        
        # reshape to 4D
        output = output.reshape(batch_size, height_new, width_new, self.out_channels)
        output = output.transpose(0,3,1,2)

        # # Compute output dimensions
        # height_padded, width_padded = X_padded.shape[2], X_padded.shape[3]
        # height_new = (height_padded - k_h) // stride_h + 1
        # width_new = (width_padded - k_w) // stride_w + 1

        # # Initialize output
        # output = np.zeros((batch_size, self.out_channels, height_new, width_new))

        # # Perform convolution
        # for b in range(batch_size):
        #     for out_c in range(self.out_channels):
        #         for in_c in range(in_channels):
        #             for i in range(height_new):
        #                 for j in range(width_new):
        #                     h_start = i * stride_h
        #                     h_end = h_start + k_h
        #                     w_start = j * stride_w
        #                     w_end = w_start + k_w
        #                     window = X_padded[b, in_c, h_start:h_end, w_start:w_end]
        #                     output[b, out_c, i, j] += np.sum(window * self.W[out_c, in_c])
        #         output[b, out_c] += self.b[out_c]  # Add bias

        # Apply dropout during training
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*output.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            output *= self.dropout_mask

        return output

    def backward(self, grad):
        """
        grads : [batch_size, out_channels, height_new, width_new]
        """
        # Apply dropout mask to gradient
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            grad *= self.dropout_mask

        # reshape grad to cols
        batch_size, out_channels, height_new, width_new = grad.shape
        grad_cols = grad.transpose(0,2,3,1).reshape(-1, out_channels)
        
        # dW, db
        dW_col = self.X_cols.T.dot(grad_cols)
        dW     = dW_col.T.reshape(self.W.shape)
        db     = grad_cols.sum(axis=0)

        # dX_cols -> dX_padded
        W_col   = self.W.reshape(self.out_channels, -1)
        dX_cols = grad_cols.dot(W_col)
        dX_padded = self.col2im(dX_cols, self.X_padded.shape, height_new, width_new)

        # 去掉 padding 得到 dX
        pad_h, pad_w = self.padding
        dX = dX_padded[
            :,
            :,
            pad_h : pad_h + self.X_shape[2],
            pad_w : pad_w + self.X_shape[3]
        ]

        # X_padded = self.X_padded
        # batch_size, out_channels, height_new, width_new = grad.shape
        # in_channels = self.in_channels
        # k_h, k_w = self.kernel_size
        # stride_h, stride_w = self.stride
        # pad_h, pad_w = self.padding

        # # Compute dW
        # dW = np.zeros_like(self.W)
        # for b in range(batch_size):
        #     for out_c in range(out_channels):
        #         for in_c in range(in_channels):
        #             for i in range(height_new):
        #                 for j in range(width_new):
        #                     h_start = i * stride_h
        #                     h_end = h_start + k_h
        #                     w_start = j * stride_w
        #                     w_end = w_start + k_w
        #                     window = X_padded[b, in_c, h_start:h_end, w_start:w_end]
        #                     dW[out_c, in_c] += window * grad[b, out_c, i, j]

        # Apply L2 regularization
        if self.L2_regularization:
            dW += self.L2_regularization_lambda * self.W

        # # Compute db
        # db = np.sum(grad, axis=(0, 2, 3))

        # # Compute dX_padded
        # dX_padded = np.zeros_like(X_padded)
        # for b in range(batch_size):
        #     for out_c in range(out_channels):
        #         for in_c in range(in_channels):
        #             for i in range(height_new):
        #                 for j in range(width_new):
        #                     h_start = i * stride_h
        #                     h_end = h_start + k_h
        #                     w_start = j * stride_w
        #                     w_end = w_start + k_w
        #                     kernel = self.W[out_c, in_c]
        #                     grad_val = grad[b, out_c, i, j]
        #                     dX_padded[b, in_c, h_start:h_end, w_start:w_end] += kernel * grad_val

        # # Remove padding to get dX
        # dX = dX_padded[:, :, pad_h:X_padded.shape[2]-pad_h, pad_w:X_padded.shape[3]-pad_w]

        # Save gradients
        self.grads['W'] = dW
        self.grads['b'] = db

        return dX

    def clear_grad(self):
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.dropout_mask = None

#=====================================================================================================
class MaxPooling(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.optimizable = False 
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.mask = None  # Track max positions
        self.X_padded = None

    def forward(self, X):
        """
        X: [batch_size, channels, height, width]
        """
        self.input = X
        batch_size, channels, height, width = X.shape
        k_h, k_w   = self.kernel_size
        s_h, s_w   = self.stride
        p_h, p_w   = self.padding

        # 1) Pad
        X_padded = np.pad(
            X,
            ((0,0),(0,0),(p_h,p_h),(p_w,p_w)),
            mode='constant'
        )
        self.X_padded = X_padded

        # 2) as_strided windows
        H_p, W_p = X_padded.shape[2], X_padded.shape[3]
        out_h = (H_p - k_h)//s_h + 1
        out_w = (W_p - k_w)//s_w + 1

        shape = (batch_size, channels, out_h, out_w, k_h, k_w)
        strides = (
            X_padded.strides[0],
            X_padded.strides[1],
            s_h * X_padded.strides[2],
            s_w * X_padded.strides[3],
            X_padded.strides[2],
            X_padded.strides[3],
        )
        windows = as_strided(X_padded, shape=shape, strides=strides)

        # 3) Forward: max + mask
        # output: [batch_size, channels, out_h, out_w]
        output = windows.max(axis=(4,5))

        # mask: one-hot normalized over k_h*k_w
        mask = (windows == output[:,:,:, :, None, None])
        count = mask.sum(axis=(4,5), keepdims=True)
        mask = mask / count
        # store for backward
        self.mask = mask

        return output

    # def forward(self, X):
    #     self.input = X
    #     batch_size, channels, height, width = X.shape
    #     k_h, k_w = self.kernel_size
    #     stride_h, stride_w = self.stride
    #     pad_h, pad_w = self.padding

    #     # Pad input
    #     X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    #     self.X_padded = X_padded

    #     # Compute output dimensions
    #     height_padded, width_padded = X_padded.shape[2], X_padded.shape[3]
    #     height_new = (height_padded - k_h) // stride_h + 1
    #     width_new = (width_padded - k_w) // stride_w + 1

    #     output = np.zeros((batch_size, channels, height_new, width_new))
    #     self.mask = np.zeros_like(X_padded, dtype=np.float32)

    #     for b in range(batch_size):
    #         for c in range(channels):
    #             for i in range(height_new):
    #                 for j in range(width_new):
    #                     h_start = i * stride_h
    #                     h_end = h_start + k_h
    #                     w_start = j * stride_w
    #                     w_end = w_start + k_w
    #                     window = X_padded[b, c, h_start:h_end, w_start:w_end]
    #                     max_val = np.max(window)
    #                     output[b, c, i, j] = max_val

    #                     # Create mask (normalized by number of max positions)
    #                     max_mask = (window == max_val)
    #                     count = np.sum(max_mask)
    #                     if count > 0:
    #                         max_mask = (max_mask.astype(np.float32) / count)
    #                     self.mask[b, c, h_start:h_end, w_start:w_end] += max_mask

    #     return output

    def backward(self, grad):
        """
        grad: [batch_size, channels, out_h, out_w]
        """
        # expand grad to window shape and multiply by mask
        # grad_windows: [batch_size, channels, out_h, out_w, k_h, k_w]
        grad_windows = grad[:,:,:, :, None, None] * self.mask

        # 反向聚合到 padded 输入
        batch_size, channels, height_p, width_p = self.X_padded.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        out_h, out_w = grad.shape[2], grad.shape[3]

        # 初始化 dX_padded
        dX_padded = np.zeros_like(self.X_padded)

        # 用向量化方法把每个 window 的梯度累加回原位
        for i in range(k_h):
            i_end = i + s_h * out_h
            for j in range(k_w):
                j_end = j + s_w * out_w
                dX_padded[
                    :, :, i:i_end:s_h, j:j_end:s_w
                ] += grad_windows[..., i, j]

        # 去 padding
        p_h, p_w = self.padding
        dX = dX_padded[
            :,
            :,
            p_h:p_h + self.input.shape[2],
            p_w:p_w + self.input.shape[3]
        ]
        return dX

    # def backward(self, grad):
    #     batch_size, channels, height_new, width_new = grad.shape
    #     k_h, k_w = self.kernel_size
    #     stride_h, stride_w = self.stride
    #     pad_h, pad_w = self.padding

    #     dX_padded = np.zeros_like(self.X_padded)
    #     for b in range(batch_size):
    #         for c in range(channels):
    #             for i in range(height_new):
    #                 for j in range(width_new):
    #                     h_start = i * stride_h
    #                     h_end = h_start + k_h
    #                     w_start = j * stride_w
    #                     w_end = w_start + k_w
    #                     grad_val = grad[b, c, i, j]
    #                     dX_padded[b, c, h_start:h_end, w_start:w_end] += self.mask[b, c, h_start:h_end, w_start:w_end] * grad_val

    #     # Remove padding
    #     dX = dX_padded[:, :, pad_h:dX_padded.shape[2]-pad_h, pad_w:dX_padded.shape[3]-pad_w]
    #     return dX