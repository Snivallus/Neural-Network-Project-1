import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

choice = 'CNN'
if choice == 'MLP':
	model = nn.models.ModelMLP()
elif choice == 'CNN':
	model = nn.models.ModelCNN()
else:
	raise(f"Invalid choice: {choice}, should be \'MLP\' or \'CNN\'")

model.load_model(r'./saved_models/best_model_8.pickle')

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
if choice == 'CNN': # 对于 CNN, 输入需要 reshape 成四维张量
	test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))