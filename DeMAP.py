import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)  # after this, 'import torch' will not change the property!

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import time
import math
import matplotlib

from _spacemap import SpaceMAP

import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def DEMaP(data, embedding, knn=30, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation


def geodesic_distance(data, knn=30, distance="data"):
    G = graphtools.Graph(data, knn=knn, decay=None)
    return G.shortest_path(distance=distance)


points_per_line = 500
num_of_lines = 11
xs = np.array(np.repeat(np.linspace(-10,10,num_of_lines), points_per_line)).reshape(1, -1)
print(xs.shape)
t = np.array(np.linspace(0, 2*np.pi, points_per_line))
A = np.array(np.linspace(10, 20, points_per_line))
# print(t)
ys = A * np.cos(t)
zs = A * np.sin(t)
ys = np.tile(ys, (1,num_of_lines))
zs = np.tile(zs, (1,num_of_lines))
color = np.tile(t, (1, num_of_lines))
x = np.hstack((xs.T,ys.T,zs.T))
print(x.shape)

import umap

umap_f = umap.UMAP(verbose=True)

y = umap_f.fit_transform(x)
print('done')
demap = DEMaP(x, y)
print(demap)


