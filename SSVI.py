import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

epsilon = 0.0000001


def SVI_raw(param, k):
    a, b, r, m, s = param

    return a + b * (r * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def option_price_svi_raw(param, F, K, r, T, w):
    P = np.exp(-r * T)
    k = np.log(K / F)
    vol= SVI_raw(param, k)

    vol = (vol / T)**0.5

    d1 = (np.log(F / K) + vol ** 2 * 0.5 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    return w * P * (F * norm.cdf(d1 * w) - K * norm.cdf(d2 * w))

def loss(param,F,K,r,T,w,y_true):

    y_pred = option_price_svi_raw(param,F,K,r,T,w)
    print( np.inner(y_pred - y_true,y_pred- y_true))
    return np.inner(y_pred - y_true,y_pred- y_true)

df = pd.read_excel('VIX OPTION DATA.xlsm', 1)
df = df[df.Group == 1]

df['r'] = 0.0137

df['m'] = np.log(df['FT'] / df['Strike'])

n = df[['Option','FT', 'Strike', 'r', 'T', 'Type','m']].as_matrix()

param = np.array([1, 1, 0.5, 1, 1])

O = n[:, 0]
F = n[:, 1]
K = n[:, 2]
r = n[:, 3]
T =  n[:, 4]
w = n[:,5]
m = n[:,6]

bounds = [(None, None), (0, None), (-1 + epsilon, 1 - epsilon), (None, None), (0 + epsilon, None)]

print(minimize(loss,param,(F,K,r,T,w,O),bounds = bounds))
