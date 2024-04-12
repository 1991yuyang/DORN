from numpy import random as rd
import numpy as np
import torch as t


def SID(alpha, beta, K):
    ti = np.arange(0, K + 1)
    ti = t.from_numpy(np.exp(np.log(alpha) + np.log(beta / alpha) * ti / K)).view((1, 1, -1))  # [1, 1, K + 1]
    return ti


if __name__ == "__main__":
    SID(1, 10, 100)