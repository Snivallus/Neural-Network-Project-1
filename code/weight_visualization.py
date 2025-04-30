# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import math

# test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
# test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# with gzip.open(test_images_path, 'rb') as f:
#         magic, num, rows, cols = unpack('>4I', f.read(16))
#         test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
# with gzip.open(test_labels_path, 'rb') as f:
#         magic, num = unpack('>2I', f.read(8))
#         test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# model = nn.models.Model_MLP()
# model.load_model(r'./saved_models/best_model_8.pickle')

# test_imgs = test_imgs / test_imgs.max()

# # logits = model(test_imgs)

# mats = []
# mats.append(model.layers[0].params['W'])
# mats.append(model.layers[2].params['W'])

# # _, axes = plt.subplots(30, 20)
# # _.set_tight_layout(1)
# # axes = axes.reshape(-1)
# # for i in range(600):
# #         axes[i].matshow(mats[0].T[i].reshape(28,28))
# #         axes[i].set_xticks([])
# #         axes[i].set_yticks([])

# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

#=============================================================================
model = nn.models.ModelCNN()
model.load_model('./saved_models/best_model_8.pickle')  # your path here

# Grab the first conv-layer’s weights
# shape = (out_channels, in_channels, k_h, k_w)
W0 = model.conv_blocks[0].W

# Since MNIST is single-channel, we can plot each 3×3 kernel directly
n_filters, _, k_h, k_w = W0.shape
cols = int(math.ceil(math.sqrt(n_filters)))
rows = int(math.ceil(n_filters/cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
for idx, ax in enumerate(axes.flat):
    if idx < n_filters:
        filt = W0[idx,0]  # take the (3×3) kernel for channel-0
        # Normalize to [0,1] for better contrast
        fmin, fmax = filt.min(), filt.max()
        ax.imshow((filt - fmin)/(fmax - fmin), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("First Conv Layer Kernels")
plt.tight_layout()
plt.savefig("./figs/weight_visualization.png", dpi=300)
plt.show()