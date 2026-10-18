import numpy as np


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def sign_accuracy(y, yhat):
    """Share of examples where the sign matches (used for topspin vs backspin)."""
    return float(np.mean(np.sign(y) == np.sign(yhat)))
