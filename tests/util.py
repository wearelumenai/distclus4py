import numpy as np


def sample(*shape):
    data = np.concatenate(((
        np.array(np.random.rand(*shape), dtype=np.float64) + np.array([2, 4])),
        np.array(
            np.random.rand(*shape), dtype=np.float64) + np.array([30, -15])
    ))
    np.random.shuffle(data)
    return data


def rmse(data, centroids, labels):
    mse = 0.
    for i, label in enumerate(labels):
        mse += np.linalg.norm(centroids[label] - data[i]) / len(data)
    return np.math.sqrt(mse)


def nan():
    return np.array([1, np.nan, 2])
