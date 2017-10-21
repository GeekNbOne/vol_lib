import numpy as np
import pandas as pd
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class VolatilityCalibration(object):
    def __init__(self, option_price, future_price, strike, expiration, option_type, N=3, init_vol=0.75):
        op_len = len(option_price)
        ft_len = len(future_price)
        strike_len = len(strike)
        exp_len = len(expiration)
        type_len = len(option_type)
        self.N = N
        self.init_vol = init_vol

        if not (op_len == ft_len == strike_len == exp_len == type_len):
            raise ValueError('All parameters supplied shall have the same length')

        self.option_price = option_price
        self.future_price = future_price
        self.strike = strike
        self.expiration = expiration
        self.option_type = option_type

    class LocalVolatility(object):

        def __init__(self, w, mu, s, M, T, FT, N):
            M = M / FT

            M, T = np.meshgrid(M, T)
            self.M = np.reshape(M, (M.shape[0], M.shape[1], 1))
            self.T = np.reshape(T, (T.shape[0], T.shape[1], 1))

            self.w = np.reshape(w, (1, 1, N))
            self.mu = np.reshape(mu, (1, 1, N))
            self.s = np.reshape(s, (1, 1, N))

        def __str__(self):
            return self.value

        @property
        def f_i(self):
            return np.exp(self.mu * self.T) / (self.w * np.exp(self.mu * self.T)).sum(axis=2, keepdims=True)

        @property
        def d_i(self):
            return (np.log(self.f_i / self.M) + ((self.s ** 2) * self.T / 2)) / (self.s * np.sqrt(self.T))

        @property
        def mu_t(self):
            return (self.w * self.mu * np.exp(self.mu * self.T)).sum(axis=2, keepdims=True) / (
                self.w * np.exp(self.mu * self.T)).sum(axis=2, keepdims=True)

        @property
        def value(self):
            upper = (self.w * self.f_i * (norm.pdf(self.d_i) * self.s + 2 * np.sqrt(np.pi * self.T)
                                          * norm.cdf(self.d_i) * (self.mu - self.mu_t))).sum(axis=2)

            lower = (self.w * self.f_i * norm.pdf(self.d_i) / self.s).sum(axis=2)

            return upper / lower

        def plot(self):
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.plot_surface(self.M.squeeze(2), self.T.squeeze(2), self.value)

            plt.show()

    def local_volatility(self, X=None):
        if X is None:
            X = self._init_parameters()

        w, mu, s = self._split_parameter(X)
        return self.LocalVolatility(w, mu, s, self.moneyness, self.expiration, self.future_price, self.N)

    def _init_parameters(self):
        w = np.array([1 / self.N] * (self.N - 1))
        mu = np.array([0] * self.N)
        s = np.array([self.init_vol] * self.N)

        X = np.concatenate((w, mu, s), axis=0)

        return X

    def _split_parameter(self, X):
        w = X[0:(self.N - 1)]
        mu = X[(self.N - 1):2 * self.N - 1]
        s = X[2 * self.N - 1:]

        w = np.concatenate((w, [1 - w.sum()]))

        return w, mu, s

    @property
    def moneyness(self):
        return self.strike / self.future_price


def load_data():
    df = pd.read_excel(r'VIX OPTION DATA.xlsm', 'Data')
    df = df[df['OTM'] == 1]

    option_price = df['Option']
    future_price = df['FT']
    strike = df['Strike']
    expiration = df['T']
    option_type = df['Type']

    return option_price, future_price, strike, expiration, option_type


option_price, future_price, strike, expiration, option_type = load_data()

v = VolatilityCalibration(option_price, future_price, strike, expiration, option_type)

X = np.array([0.7, 0.25, 0, 0.3, -0.5, 0.3, 0.6, 1])

vol = v.local_volatility(X).mu_t

print(vol)
