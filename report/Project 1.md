# Project 1

Due: May 1, 2025  
姓名: 雍崔扬  
学号: 21307140051

## 1. Problem Setting

In this problem we will investigate handwritten digit classification.   
MNIST (Modified National Institute of Standards and Technology database)   
is a large database of handwritten digits commonly used for training various image processing systems.   
The database is also widely used for training and testing in the field of machine learning.   
It was created by "re-mixing" the samples from NIST's original datasets.   
The dataset contains $60,000$ training images and $10,000$ testing images.   
Each image is a $28\times 28$ pixel grayscale image and is labeled with the correct digit ($0\sim9$) it represents.   
You need to implement one or more neural network to recognize the handwritten digit,   
and conduct experiment to test your model and conclude the ability of your model.   
After that, you may implement several modifications to your model and test whether the model acts better.   
Do not worry about whether your mode's performance is better than others',   
the effort you’ve made to improve your model matters.   

Please refer to following instruction of writing the report:

- ① The goal of your write-up is to document the experiments you’ve done and your main findings.   
  So be sure to explain the results.   
  Hand in a single PDF file of your report.   
  Enclose a Github link to your codes in your submitted file.   
  You should also provide a link to your dataset and your trained model weights in your report.   
  You may upload the dataset and model into Google Drive or other Netdisk service platform.   
  Also put the name and Student ID in your paper.   
  Lack of code link or model weights link will lead to a penalization of scores.  
- ② You may use Mindspore to do some visualization   
  or do some experiments if your implemented version runs too slow.  
  But you must implement your own version first.

- ③ Note that the goal of this project is to let the students do the practice,   
  and write some basic components of a neural network or CNN.   
  So do not invoke the deep learning functions/modules   
  that can directly give the results of the following questions (e,g, the PyTorch package).   
  The project is very easy, and you do not really need GPUs;   
  just running it on CPU will be enough.



## 2. Implementation

1. `op.py`   
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py`   
   You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py`   
   You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.

### 2.1 `op.py`

#### (1) `Linear`

Implement the forward and backward function of class `Linear`.

- **① 前向传播:**  
  $$
  X \in \mathbb R^{\text{batch\_size}\times \text{in\_dim}}\\
  W \in \mathbb R^{\text{in\_dim}\times \text{out\_dim}}\\
  b \in \mathbb R^{1\times \text{out\_dim}}\\
  \hline
  Z = X W + 1_{\text{batch\_size}}\cdot b \in \mathbb R^{\text{batch\_size}\times \text{out\_dim}}
  $$

  ```python
  def forward(self, X):
      """
      input: [batch_size, in_dim]
      Weight: [in_dim, out_dim]
      Bias: [1, out_dim]
      output: [batch_size, out_dim]
      """
      self.input = X
      return np.dot(X, self.W) + self.b
  ```

- **② 反向传播:**
  $$
  \text{grads}=\frac{\partial \mathcal L}{\partial Z} \in \mathbb R^{\text{batch\_size}\times \text{out\_dim}}\\
  X \in \mathbb R^{\text{batch\_size}\times \text{in\_dim}}\\
  W \in \mathbb R^{\text{in\_dim}\times \text{out\_dim}}\\
  b \in \mathbb R^{1\times \text{out\_dim}}\\
  \hline
  \frac{\partial \mathcal L}{\partial W} = \frac{\partial Z}{\partial W}\cdot \frac{\partial \mathcal L}{\partial Z} = X^{\mathrm T} \cdot \text{grads}\in \mathbb R^{\text{in\_dim}\times \text{out\_dim}}\\
  
  \frac{\partial \mathcal L}{\partial b} = \frac{\partial Z}{\partial b}\cdot \frac{\partial \mathcal L}{\partial Z} = 1_{\text{batch\_size}}^{\mathrm T} \cdot \text{grads}\in \mathbb R^{1\times \text{out\_dim}}\\
  
  \frac{\partial \mathcal L}{\partial X} = \frac{\partial Z}{\partial X}\cdot \frac{\partial \mathcal L}{\partial Z} = \text{grads}\cdot W^{\mathrm T}\in \mathbb R^{\text{batch\_dim}\times \text{in\_dim}}\\
  $$

  同时我们还可以引入对权重 $W$ 的 $l_2$ 正则化:
  
  ```python
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
  ```



#### (2) `CrossEntropyLoss`

Implement the `CrossEntropyLoss`.   
Note that the `Softmax` layer could be included in the `CrossEntropyLoss`.

- **① Softmax 函数:**    
  Softmax 函数可以视为 $\text{argmax}(\cdot)$ 函数的平滑版本 (如果我们认为 $\text{argmax}(\cdot)$ 的输出是 one-hot 向量的话)  
  同时它也可以视为 Sigmoid 函数在高维情况的推广: 
  $$
  \begin{align}
  \text{softmax}(z) 
  &:= \frac{\exp(z)}{1_d^{\mathrm T} \exp(z)} \\
  &=
  \frac{1}{\sum_{i=1}^d \exp(z_i)} 
  \begin{bmatrix}
  \exp(z_1)\\
  \vdots\\
  \exp(z_d)
  \end{bmatrix}
  \end{align}\quad (\forall\ x\in \mathbb R^d)
  $$
  它可以用来表示一个具有 $n$ 个可能取值的离散型随机变量的分布.  
  引入 $\text{batch\_size}$ 后，对于 $X\in \mathbb R^{\text{batch\_size}\times D}$ 计算 $\text{softmax}(X)\in \mathbb R^{\text{batch\_size}\times D}$ 的代码如下:  
  (同时引入了增加数值稳定性的技巧)

  ```python
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
  ```

- **② 前向传播:**
  $$
  \text{labels} = y\in \mathbb R^{\text{batch\_size}}\\
  \text{logits} = X\in \mathbb{R}^{\text{batch\_size} \times D}\\
  \hline
  \text{probs} = \text{softmax}(X) \in \mathbb{R}^{\text{batch\_size} \times D}\\
  \mathcal{L} = -\frac{1}{\text{batch\_size}} \sum_{i=1}^{\text{batch\_size}} \log (\text{probs}_{i, y_i} + \varepsilon) \in \mathbb R
  $$
  其中 $\varepsilon>0$ 是一个小的正数，用于避免出现 $\log(0)$ 的错误.

  ```python
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
  ```

- **③ 反向传播:**
  $$
  \text{labels} = y\in \mathbb R^{\text{batch\_size}}\\
  \text{logits} = X\in \mathbb{R}^{\text{batch\_size} \times D}\\
  \hline
  \text{one\_hot} = [\delta_{y_i,j}] \in \mathbb R^{\text{batch\_size}\times D}\\
  \text{probs} = \text{softmax}(X) \in \mathbb{R}^{\text{batch\_size} \times D}\\
  \mathcal{L} = -\frac{1}{\text{batch\_size}} \sum_{i=1}^{\text{batch\_size}} \log (\text{probs}_{i, y_i} + \varepsilon) \in \mathbb R\\
  \frac{\partial \mathcal{L}}{\partial X} = \frac{1}{\text{batch\_size}} (\text{probs} - \text{one\_hot}) \in \mathbb R^{\text{batch\_size}\times D}
  $$
  其中 $\delta_{i,j}:= \begin{cases}
  1,&\text{if }i=j\\
  0,&\text{otherwise}\end{cases}$ 为 Dirac Delta 函数的一种表示形式.  
  
  ```python
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
  ```



#### (3) `conv2D`

Try to implement `conv2D`, do not worry about the efficiency.











## 3. Questions

Below are additional features you could try to incorporate into your neural network   
to improve performance (the options are approximately in order of increasing difficulty).   
The modifications you make in trying to improve performance are up to you   
and you can even try things that are not on the question list.   
But, let's stick with neural networks models and only use one neural network (no ensembles).   
Remember to write in your reports about what modifications you’ve made and the effectiveness of the modification.













