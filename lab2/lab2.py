import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


class RandomGenerator:
    def __init__(self, m, z, mod):
        self.m = m
        self.z = z
        self.mod = mod

    def next_float(self):
        self.z = self.m * self.z % self.mod
        return self.z / self.mod


def normalize(data: np.array):
    res = np.zeros_like(data)
    maximum = np.max(data)
    minimum = np.min(data)
    for idx, val in np.ndenumerate(data):
        res[idx[0]] = (val - minimum) / (maximum - minimum)
    return res


def sma(data: np.array):
    res = np.zeros_like(data)
    for i in range(data.shape[0]):
        up = sum([data[j] for j in range(i + 1)])
        low = (i + 1)
        res[i] = up / low
    return res


def weighted_sma(arr: np.array, weights=None):
    if weights is None:
        weights = np.arange(1, arr.shape[0] + 1)
        weights = weights / np.sum(weights)
    res = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        up = sum([weights[j] * arr[j] for j in range(i + 1)])
        low = sum([weights[k] for k in range(i + 1)])
        res[i] = up / low
    return res


def holt(data: np.array, alpha, beta):
    s = np.zeros_like(data)
    m = np.zeros_like(data)
    res = np.zeros_like(data)
    s[0] = data[0]
    m[0] = data[1] - data[0]
    for i in range(1, data.shape[0]):
        s[i] = alpha * data[i] + (1 - alpha) * (s[i - 1] + m[i - 1])
        m[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * m[i - 1]
        res[i] = s[i] + m[i]
    return res


def winters(data: np.array, alpha, beta, gamma, L: int):
    s = np.zeros_like(data)
    m = np.zeros_like(data)
    q = np.copy(data)
    s[0] = data[0]
    m[0] = data[1] - data[0]
    for i in range(1, data.shape[0]):
        s[i] = alpha * data[i] / q[i - L] + (1 - alpha) * (s[i - 1] + m[i - 1])
        m[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * m[i - 1]
        q[i] = gamma * data[i] / s[i] + (1 - gamma) * q[i - L]
    return q


def trig_licha(data: np.array, alpha_0, beta):
    s = np.zeros_like(data)
    d = np.zeros_like(data)
    g = np.zeros_like(data)
    e = np.zeros_like(data)
    alpha = np.zeros_like(data)
    res = np.zeros_like(data)
    alpha[0] = alpha_0
    d[0] = data[0]
    g[0] = data[0]
    s[0] = data[0]
    s[1] = alpha_0 * data[0] + (1 - alpha_0) * s[0]
    res[0] = s[1]
    for i in range(1, data.shape[0] - 1):
        e[i] = data[i] - res[i]
        g[i] = beta * abs(e[i]) + (1 - beta) * g[i - 1]
        d[i] = beta * e[i] + (1 - beta) * d[i - 1]
        alpha[i] = abs(d[i] / g[i])
        s[i] = alpha[i] * data[i] + (1 - alpha[i]) * s[i - 1]
        res[i + 1] = s[i]
    return res


def exps(data: np.array, alpha):
    s = np.zeros_like(data)
    res = np.zeros_like(data)
    s[0] = data[0]
    for i in range(1, data.shape[0]):
        s[i] = alpha * data[i] + (1 - alpha) * s[i - 1]
        res[i] = s[i - 1]
    return res


def sign(data: np.array, alpha, index):
    if index == 0 or index == data.shape[0]:
        return 0
    delta_z = np.zeros_like(data)
    for i in range(data.shape[0] - 1):
        delta_z[i] = data[i + 1] - data[i]

    k = np.zeros_like(delta_z)
    for i in range(delta_z.shape[0]):
        if delta_z[i] > 0:
            k[i] = 1
        elif delta_z[i] < 0:
            k[i] = -1
        else:
            k[i] = 0

    m = np.zeros_like(k)
    for i in range(1, k.shape[0]):
        m[i] = k[i] * k[i - 1]

    s = exps(m, alpha)
    m_new = np.zeros_like(s)
    for t in range(s.shape[0] - 1):
        if s[t] > 0:
            m_new[t + 1] = 1
        elif s[t] < 0:
            m_new[t + 1] = -1
        else:
            m_new[t + 1] = 0

    sign_ = np.zeros_like(m_new)
    for t in range(m_new.shape[0] - 1):
        sign_[t] = m_new[t + 1] * k[t]
    if sign_[index] > 0:
        return 1
    elif sign_[index] < 0:
        return -1
    else:
        return 0


def plot_plt(plot: plt):
    plot.legend()
    plot.xlabel('Index')
    plot.ylabel('Price')
    plot.title('Data')
    plot.show()


def task(z1: np.array, z2: np.array):
    x = np.arange(0, 500, 1)
    z1_sma = sma(z1)
    plt.plot(x, z1, label='Real z1', alpha=0.5)
    plt.plot(x, z1_sma, label='SMA z1')
    plot_plt(plt)

    z2_sma = sma(z2)
    plt.clf()
    plt.plot(x, z2, label='Random z2', alpha=0.5)
    plt.plot(x, z2_sma, label='SMA z2')
    plot_plt(plt)

    z1_weighted_sma = weighted_sma(z1)
    plt.clf()
    plt.plot(x, z1, label='Real z1', alpha=0.5)
    plt.plot(x, z1_weighted_sma, label='Weighted z1')
    plot_plt(plt)

    z2_weighted_sma = weighted_sma(z2)
    plt.clf()
    plt.plot(x, z2, label='Random z2', alpha=0.5)
    plt.plot(x, z2_weighted_sma, label='Weighted z2')
    plot_plt(plt)

    z1_holt = holt(z1, 0.1, 0.1)
    plt.clf()
    plt.plot(x, z1, label='Real z1', alpha=0.5)
    plt.plot(x, z1_holt, label='Holt z1')
    plot_plt(plt)

    z2_holt = holt(z1, 0.1, 0.1)
    plt.clf()
    plt.plot(x, z2, label='Random z2', alpha=0.5)
    plt.plot(x, z2_holt, label='Holt z2')
    plot_plt(plt)

    z1_winters = winters(z1, 0.1, 0.5, 0.1, 250)
    plt.clf()
    plt.plot(x, z1, label='Real z1', alpha=0.5)
    plt.plot(x, z1_winters, label='Winters z1')
    plot_plt(plt)

    z2_winters = winters(z2, 0.1, 0.5, 0.1, 250)
    plt.clf()
    plt.plot(x, z2, label='Random z2', alpha=0.5)
    plt.plot(x, z2_winters, label='Winters z2')
    plot_plt(plt)

    z1_trig_licha = trig_licha(z1, 0.1, 0.1)
    plt.clf()
    plt.plot(x, z1, label='Real z1', alpha=0.5)
    plt.plot(x, z1_trig_licha, label='Trig_licha z1')
    plot_plt(plt)

    z2_trig_licha = trig_licha(z2, 0.1, 0.1)
    plt.clf()
    plt.plot(x, z2, label='Random z2', alpha=0.5)
    plt.plot(x, z2_trig_licha, label='Trig_licha z2')
    plot_plt(plt)

    z1_sign = sign(z1, 0.1, 250)
    z2_sign = sign(z2, 0.1, 250)
    print(z1_sign)
    print(z2_sign)


def main():
    # Download real data
    prices = fetch_california_housing()
    price_row = prices.target[:500]
    price_row_norm = normalize(price_row)

    # Generate random data
    rand = RandomGenerator(477211307, 123456789, 2**32)
    random_row = np.array([rand.next_float() for _ in range(500)])
    print(random_row[:10])
    task(price_row_norm, random_row)


main()
