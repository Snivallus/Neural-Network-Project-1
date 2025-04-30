# 读取 MNIST 数据并训练 MLP 模型，模型定义在 mynn.models.ModelMLP 中
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import gzip
import pickle
from draw_tools.plot import plot

# 导入自己实现的模块
from mynn.models import ModelMLP, ModelCNN
from mynn.optimizer import SGD
from mynn.lr_scheduler import ExponentialLR
from mynn.op import CrossEntropyLoss
from mynn.metric import accuracy
from mynn.runner import RunnerM

#========================================================================================
import sys
import atexit
class Logger:
    def __init__(self, filename, max_line_length=256):
        self.terminal = sys.stdout  # 终端输出
        self.log = open(filename, "w", encoding="utf-8")  # 日志文件，防止编码问题
        self.max_line_length = max_line_length  # 最大行长度

    def write(self, message):
        # 自动处理过长的行
        lines = self._split_long_lines(message)
        
        # 将每一行分别输出
        for line in lines:
            self.terminal.write(line)  # 终端输出
            self.log.write(line)  # 写入文件
            self.log.flush()  # 立刻写入文件，防止缓冲区丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):  # 关闭文件
        self.log.close()

    def _split_long_lines(self, message):
        """将长行拆分成多个适合显示的短行"""
        lines = []
        while len(message) > self.max_line_length:
            # 找到最接近max_line_length的位置进行换行
            split_point = message.rfind(" ", 0, self.max_line_length)
            if split_point == -1:
                # 如果找不到空格，直接在max_line_length处拆分
                split_point = self.max_line_length
            lines.append(message[:split_point] + '\n')
            message = message[split_point:].lstrip()  # 去掉前面的空格
        lines.append(message)  # 添加最后剩余的部分
        return lines

#========================================================================================
# 启用 Logger 并确保程序退出时关闭
sys.stdout = Logger('./log/test_train.log')
sys.stderr = sys.stdout  # 让错误也记录
atexit.register(sys.stdout.close)  # 确保程序退出时关闭日志文件

#========================================================================================
from PIL import Image
def rotate_images(X, angles):
    """
    离线将 X 中的每张图像按 angles 列表中的角度旋转,
    返回一个新的大数组, shape = (len(angles)*N, 1, H, W)
    """
    X = X.reshape(-1, 1, 28, 28)
    B, C, H, W = X.shape
    all_imgs = [X]  # 原始不动
    for a in angles:
        aug = np.empty_like(X)
        for i in range(B):
            arr = (X[i,0] * 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode='L')
            pil2 = pil.rotate(a)
            aug[i,0] = np.array(pil2, dtype=np.float32) / 255.0
        all_imgs.append(aug)
    # 垂直拼接
    X = np.concatenate(all_imgs, axis=0)
    X = X.reshape(-1, 28*28)
    return X

#========================================================================================
# 设置随机种子
np.random.seed(51)

# 数据路径
train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

# 加载图像数据（28x28 的手写数字图像）
with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

# 加载标签数据
with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 打乱数据，并划分出验证集（前 10000 个作为验证集）
idx = np.random.permutation(num)
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 数据归一化处理, 将像素值缩放到 [0, 1] 区间
train_imgs = train_imgs.astype(np.float32) / 255.0
valid_imgs = valid_imgs.astype(np.float32) / 255.0

choice = 'CNN'
augmentation = True # 是否进行数据增广
if choice == 'MLP':
    # 初始化 MLP 模型
    model = ModelMLP(
        size_list=[784, 600, 10],        # 输入层 784，隐藏层 600，输出层 10（分类）
        act_func='relu',                 # 使用 ReLU 激活函数
        lambda_list=[1e-2, 1e-2]         # 对每个 Linear 层使用 L2 正则
    )
elif choice == 'CNN':
	if augmentation: # 数据增广
		angles = [-15, 0, 15] # 扩增 ±15°
		augmented_imgs = rotate_images(train_imgs, angles)
		augmented_labs = np.tile(train_labs, len(angles)+1)

		# 打乱并拆分
		idx = np.random.permutation(augmented_imgs.shape[0])
		train_imgs = augmented_imgs[idx]
		train_labs = augmented_labs[idx]

	train_imgs = train_imgs.reshape(-1, 1, 28, 28)
	valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

	model = ModelCNN(
	input_shape   = (1, 28, 28),
	conv_channels = [8],            # 卷积层输出通道
	kernel_sizes  = [3, 3],         # 卷积核 3×3
	pool_sizes    = [2, 2],         # 池化窗口 2×2
	fc_sizes      = [128, 10],      # 全连接层：128 → 10
	act_func      = 'relu',         # 激活：ReLU
	lambda_conv   = [1e-4],   		# 卷积层的 L2 正则
	lambda_fc     = [1e-4, 0.0]     # 全连接层 L2: 只对第一层作用
	)
else:
    raise(f"Invalid choice: {choice}, should be \'MLP\' or \'CNN\'")

# 优化器 SGD + 动态学习率调整
optimizer = SGD(init_lr=0.01, model=model, type='adam')
scheduler = ExponentialLR(
    optimizer=optimizer,
    beta=0.999,
    use_exp=False
)

# 定义损失函数（交叉熵损失 + 自动支持 softmax + L2 正则项）
loss_fn = CrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

# 构建训练器 RunnerM（封装了 forward + backward + optimizer.step + metric + loss + scheduler）
runner = RunnerM(
    model=model,
    optimizer=optimizer,
    metric=accuracy,
    loss_fn=loss_fn,
    batch_size=1024,
    scheduler=scheduler
)

# 训练模型
runner.train(
    [train_imgs, train_labs],   # train_set
    [valid_imgs, valid_labs],   # dev_set
    num_epochs=3,
    log_iters=100,
    save_dir='./saved_models'
)

# 绘制训练过程图像（loss / accuracy）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.reshape(-1)
fig.tight_layout()
plot(runner, axes)
plt.savefig("./figs/training_curves.png", dpi=300)
plt.show()