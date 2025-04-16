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


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
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

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        X = self.input
        # Compute gradient for W: X^T @ grad, with possible weight decay
        # Gradiant ∂L/∂W = ∂Z/∂W · ∂L/∂Z = Xᵀ · grad
        dW = np.dot(X.T, grad)
        if self.weight_decay: # L2 regularization
            dW += self.weight_decay_lambda * self.W
        
        # Compute gradient for b: sum over the batch dimension
        # Gradiant ∂L/∂b = ∂Z/∂b · ∂L/∂Z = 1ᵀ · grad
        db = np.sum(grad, axis=0, keepdims=True)

        # Compute gradient for the previous layer: grad @ W^T
        # Jacobian ∂L/∂X = ∂L/∂Z · ∂Z/∂X = grad · Wᵀ
        grad_prev = np.dot(grad, self.W.T)

        # Save gradients
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return grad_prev

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
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

def softmax(X):
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

class MultiCrossEntropyLoss(Layer):
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

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels
        batch_size = predicts.shape[0]
        
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        
        # Compute cross-entropy loss with numerical stability
        correct_log_probs = np.log(self.probs[np.arange(batch_size), labels] + 1e-8)
        loss = -np.mean(correct_log_probs)
        
        return loss

    def backward(self):
        """
        Backward pass to compute gradient of the loss w.r.t. inputs to softmax (logits).
        """
        # first compute the grads from the loss to the input
        y_true = np.zeros_like(self.probs)  # Initialize one-hot label matrix
        y_true[np.arange(self.batch_size), self.labels] = 1
        self.grads = (self.probs - y_true) / self.batch_size # Gradient of loss w.r.t. input logits

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        """
        Cancel internal softmax computation.
        """
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass