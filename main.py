# test block
import matplotlib.pyplot as plt
import numpy as np

# used for correcting the initialization of SpaceMAP (will be removed soon)
from temp import temporary_fn

temporary_fn()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib

from _spacemap import SpaceMAP


# 1) Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
x_train = torchvision.datasets.MNIST(root='../TUDelft/Datasets/data_multi/', train=True, transform=transform, download=True)
x_test = torchvision.datasets.MNIST(root='../TUDelft/Datasets/data_multi/', train=False, transform=transform, download=True)
x_targets = np.concatenate((x_train.targets, x_test.targets), axis=0)
x = np.concatenate((np.array(x_train.data).astype('float32'), np.array(x_test.data).astype('float32')), axis=0)
x = x / 255.0
x = x.reshape(x.shape[0], -1)

# 2) instantiate SpaceMAP
spacemap = SpaceMAP(n_near_field=21,
                    n_middle_field=700,
                    d_local=0,
                    d_global=0,
                    n_epochs=200,
                    init='spectral',
                    metric='euclidean',
                    verbose=True,
                    plot_results=False,
                    num_plots=50,
                    )
y = spacemap.fit_transform(x)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet', 'peru', 'olive', 'midnightblue',
              'palevioletred']
cmap = matplotlib.colors.ListedColormap(colors[::-1])
scatter1 = ax.scatter(y[:, 0], y[:, 1], s=0.1, cmap=cmap, c=x_targets[:], alpha=1)
ax.set_aspect('equal')
plt.axis('off')
plt.show()



