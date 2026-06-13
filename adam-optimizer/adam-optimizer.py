import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.array(param, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    m2 = beta1 * m + (1 - beta1) * grad
    v2 = beta2 * v + (1 - beta2) * (grad ** 2)
    m3 = m2 / ( 1 - beta1 ** t)
    v3 = v2 / (1 - beta2 ** t)
    param2 = param - lr * m3 / (np.sqrt(v3) + eps)
    return param2, m2, v2