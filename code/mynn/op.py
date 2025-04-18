from abc import abstractmethod
import numpy as np

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
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, L2_regularization=False, L2_regularization_lambda=1e-8) -> None:
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
# TODO: 实现 ReLU 类激活函数
class ReLU(Layer):
    """
    ReLU activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

# TODO: 实现 Sigmoid 类激活函数

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
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, L2_regularization=False, L2_regularization_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out_channels, in_cha, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, height_new, width_new]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

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
        self.X_padded = None
        self.dropout_mask = None  # For dropout implementation

    def forward(self, X, training=True):
        """
        input X: [batch, in_channels, height, width]
        W : [1, out_channels, in_channels, k_h, k_w]
        """
        self.input = X
        batch_size, in_channels, height, width = X.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Pad input
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        self.X_padded = X_padded

        # Compute output dimensions
        height_padded, width_padded = X_padded.shape[2], X_padded.shape[3]
        height_new = (height_padded - k_h) // stride_h + 1
        width_new = (width_padded - k_w) // stride_w + 1

        # Initialize output
        output = np.zeros((batch_size, self.out_channels, height_new, width_new))

        # Perform convolution
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(in_channels):
                    for i in range(height_new):
                        for j in range(width_new):
                            h_start = i * stride_h
                            h_end = h_start + k_h
                            w_start = j * stride_w
                            w_end = w_start + k_w
                            window = X_padded[b, in_c, h_start:h_end, w_start:w_end]
                            output[b, out_c, i, j] += np.sum(window * self.W[out_c, in_c])
                output[b, out_c] += self.b[out_c]  # Add bias

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

        X_padded = self.X_padded
        batch_size, out_channels, height_new, width_new = grad.shape
        in_channels = self.in_channels
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Compute dW
        dW = np.zeros_like(self.W)
        for b in range(batch_size):
            for out_c in range(out_channels):
                for in_c in range(in_channels):
                    for i in range(height_new):
                        for j in range(width_new):
                            h_start = i * stride_h
                            h_end = h_start + k_h
                            w_start = j * stride_w
                            w_end = w_start + k_w
                            window = X_padded[b, in_c, h_start:h_end, w_start:w_end]
                            dW[out_c, in_c] += window * grad[b, out_c, i, j]

        # Apply L2 regularization
        if self.L2_regularization:
            dW += self.L2_regularization_lambda * self.W

        # Compute db
        db = np.sum(grad, axis=(0, 2, 3))

        # Compute dX_padded
        dX_padded = np.zeros_like(X_padded)
        for b in range(batch_size):
            for out_c in range(out_channels):
                for in_c in range(in_channels):
                    for i in range(height_new):
                        for j in range(width_new):
                            h_start = i * stride_h
                            h_end = h_start + k_h
                            w_start = j * stride_w
                            w_end = w_start + k_w
                            kernel = self.W[out_c, in_c]
                            grad_val = grad[b, out_c, i, j]
                            dX_padded[b, in_c, h_start:h_end, w_start:w_end] += kernel * grad_val

        # Remove padding to get dX
        dX = dX_padded[:, :, pad_h:X_padded.shape[2]-pad_h, pad_w:X_padded.shape[3]-pad_w]

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
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.mask = None  # Track max positions
        self.X_padded = None

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # Pad input
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        self.X_padded = X_padded

        # Compute output dimensions
        height_padded, width_padded = X_padded.shape[2], X_padded.shape[3]
        height_new = (height_padded - k_h) // stride_h + 1
        width_new = (width_padded - k_w) // stride_w + 1

        output = np.zeros((batch_size, channels, height_new, width_new))
        self.mask = np.zeros_like(X_padded, dtype=np.float32)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_new):
                    for j in range(width_new):
                        h_start = i * stride_h
                        h_end = h_start + k_h
                        w_start = j * stride_w
                        w_end = w_start + k_w
                        window = X_padded[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val

                        # Create mask (normalized by number of max positions)
                        max_mask = (window == max_val)
                        count = np.sum(max_mask)
                        if count > 0:
                            max_mask = (max_mask.astype(np.float32) / count)
                        self.mask[b, c, h_start:h_end, w_start:w_end] += max_mask

        return output

    def backward(self, grad):
        batch_size, channels, height_new, width_new = grad.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        dX_padded = np.zeros_like(self.X_padded)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_new):
                    for j in range(width_new):
                        h_start = i * stride_h
                        h_end = h_start + k_h
                        w_start = j * stride_w
                        w_end = w_start + k_w
                        grad_val = grad[b, c, i, j]
                        dX_padded[b, c, h_start:h_end, w_start:w_end] += self.mask[b, c, h_start:h_end, w_start:w_end] * grad_val

        # Remove padding
        dX = dX_padded[:, :, pad_h:dX_padded.shape[2]-pad_h, pad_w:dX_padded.shape[3]-pad_w]
        return dX