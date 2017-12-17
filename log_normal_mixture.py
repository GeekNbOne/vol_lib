import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import pickle

class VolatilitySurface(object):

    def __init__(self, mixture_depth=3):
        self.mixture_depth = mixture_depth
        self.X = None

    def calibrate(self, y_true, F, K, r, T, w):
        X0 = self._generate_X0()

        res = minimize(self._lognormal_mixture_loss, X0, (F, K, r, T, w, y_true), bounds=self._bounds,
                       constraints=self._constrains)

        self.X = res.x

        return res.message

    def bs_volatility(self,m,T):

        y_true = self._lognormal_mixture_price_moneyness(m,T)
        X0 = np.random.uniform(0,1,y_true.shape[0])
        res = minimize(self._bs_loss,X0,(m,T,y_true))

        return res.x

    def save_X(self,file_name):
        pickle.dump(self.X,open(file_name,'wb'))

    def load_X(self,file_name):
        self.X = pickle.load(open(file_name,'rb'))

    def _bs_loss(self,n,m,T,y_true):
        y_pred = self._bs_price(n,m,T)

        return self._loss_mse(y_pred,y_true)


    def _bs_price(self,n,m,T):

        d1 = (m + 0.5 * n**2) / (n * np.sqrt(T))
        d2 = d1 - n * np.sqrt(T)

        return  norm.cdf(d1) - np.exp(-m) * norm.cdf(d2)


    def _lognormal_mixture_price_moneyness(self,m,T):

        l = self.X[:self.mixture_depth].reshape((1,self.mixture_depth))
        n = self.X[self.mixture_depth:].reshape((1, self.mixture_depth))

        m = np.expand_dims(m, 1)
        T = np.expand_dims(T, 1)


        d1 = ( m + 0.5 * n **2 ) / (n * np.sqrt(T))
        d2 = d1 - n * np.sqrt(T)

        return (l * (norm.cdf(d1) - np.exp(-m) * norm.cdf(d2))).sum(axis =1)


    def _lognormal_mixture_price(self, X, F, K, r, T, w):
        l = X[:self.mixture_depth].reshape((1, self.mixture_depth))
        n = X[self.mixture_depth:].reshape((1, self.mixture_depth))

        F = F.reshape((F.shape[0], 1))
        K = K.reshape((K.shape[0], 1))
        T = T.reshape((T.shape[0], 1))
        w = w.reshape((w.shape[0], 1))
        r = r.reshape((r.shape[0], 1))

        P = np.exp(-r * T)

        d1 = (np.log(F / K) + 0.5 * n ** 2 * T) / (n * np.sqrt(T))
        d2 = d1 - n * np.sqrt(T)

        inner_sum = l * (F * norm.cdf(d1 * w) - K * norm.cdf(d2 * w))

        return (w * P * inner_sum.sum(axis=1))[:, 0]

    def _lognormal_mixture_loss(self, X, F, K, r, T, w, y_true):
        y_pred = self._lognormal_mixture_price(X, F, K, r, T, w)
        return self._loss_mse(y_pred,y_true)


    def _generate_X0(self):
        N = self.mixture_depth
        X = np.random.uniform(0, 1, N * 2)
        X[:N] = X[:N] / X[:N].sum()

        return X

    def _loss_mse(self, y_pred, y_true):
        return np.inner(y_pred - y_true, y_pred - y_true) / y_pred.shape[0]

    @property
    def _bounds(self):
        return [(0, 1)] * self.mixture_depth + [(0.000001, None)] * self.mixture_depth

    @property
    def _constrains(self):
        N = self.mixture_depth
        fun = lambda X: np.inner(X, np.array([1] * N + [0] * N)) - 1

        return {'type': 'eq', 'fun': fun}




df = pd.read_excel('VIX OPTION DATA.xlsm', 1)

df['r'] = 0.0137

df['m'] = np.log(df['FT'] / df['Strike'])


n = df[['Option','FT', 'Strike', 'r', 'T', 'Type','m']].as_matrix()

vol = VolatilitySurface()

O = n[:, 0]
F = n[:, 1]
K = n[:, 2]
r = n[:, 3]
T =  n[:, 4]
w = n[:,5]
m = n[:,6]

print(vol.calibrate(O,F,K,r,T,w))
# vol.load_X('vol.pkl')

print(vol.X[-3:]/ np.sqrt(0.05))
# print(vol.bs_volatility(np.array([0,0.3]),0.05))

