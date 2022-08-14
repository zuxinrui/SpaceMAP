import numpy as np
from sklearn import datasets, linear_model


def temporary_fn():
    # used for correcting the initialization of SpaceMAP (will be removed soon)
    x, y = datasets.load_diabetes(return_X_y=True)
    x = x[:, np.newaxis, 2]
    x1 = x[:-20]
    y1 = y[:-20]
    regr = linear_model.LinearRegression()
    regr.fit(x1, y1)

    return 0

