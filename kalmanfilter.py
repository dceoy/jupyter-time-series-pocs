#!/usr/bin/env python

import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class KalmanFilter(object):
    def __init__(self, x0=0.0, v0=1.0, q=1.0, r=1.0, keep_history=False):
        self.x = np.array([x0])             # estimate of x
        self.v = np.array([v0])             # error estimate
        self.q = q                          # process variance
        self.r = r                          # measurement variance
        self.y = np.array([np.nan])         # observation
        self.__keep_history = keep_history

    def filter(self, y, x0=None, v0=None, q=None, r=None):
        x0_ = x0 or self.x[-1]
        v0_ = v0 or self.v[-1]
        q_ = q or self.q
        r_ = r or self.r
        len_y = len(y)
        new_x = np.empty(len_y)
        new_v = np.empty(len_y)
        for i, y_n in enumerate(y):
            x_n_1 = (new_x[i - 1] if i else x0_)
            v_n_1 = (new_v[i - 1] if i else v0_) + q_
            k = v_n_1 / (v_n_1 + r_)
            new_x[i] = x_n_1 + k * (y_n - x_n_1)
            new_v[i] = (1 - k) * v_n_1
        if self.__keep_history:
            self.x = np.append(self.x, new_x)
            self.v = np.append(self.v, new_v)
            self.y = np.append(self.y, y)
        else:
            self.x = np.array([new_x[-1]])
            self.v = np.array([new_v[-1]])
        return pd.DataFrame(
            {'y': y, 'x': new_x, 'v': new_v},
            index=(y.index if hasattr(y, 'index') else range(len_y))
        )


class OptimizedKalmanFilter(object):
    def __init__(self, y, x0=0.0, v0=1.0, q0=1.0, r0=1.0, method='L-BFGS-B',
                 keep_history=False):
        self.__logger = logging.getLogger(__name__)
        self.y = y                      # observation
        self.x0 = x0                    # estimate of x
        self.v0 = v0                    # error estimate
        self.q0 = q0                    # process variance
        self.r0 = r0                    # measurement variance
        self.__method = method          # method for scipy.optimize.minimize()
        self.__keep_history = keep_history
        self.q = None
        self.r = None
        self.kf = None
        self.optimize_parameters()
        self.__logger.debug('vars(self): {}'.format(vars(self)))

    def optimize_parameters(self):
        res = minimize(
            fun=self._loss, x0=np.array([self.q0, self.r0]),
            args=(self.y, self.x0, self.v0), method=self.__method
        )
        self.__logger.info(f'{os.linesep}{res}')
        self.q = res.x[0]
        self.__logger.info(f'process variance: {self.q}')
        self.r = res.x[1]
        self.__logger.info(f'measurement variance: {self.r}')
        self.kf = KalmanFilter(
            x0=self.x0, v0=self.v0, q=self.q, r=self.r,
            keep_history=self.__keep_history
        )

    @staticmethod
    def _loss(x, *args):
        return KalmanFilter(
            x0=args[1], v0=args[2], q=x[0], r=x[1], keep_history=True
        ).filter(y=args[0]).pipe(
            lambda d: np.sum(np.square(d['y'] - d['x']))
        )

    def filter(self, y=None, **kwargs):
        return self.kf.filter(y=(y or self.y), **kwargs)
