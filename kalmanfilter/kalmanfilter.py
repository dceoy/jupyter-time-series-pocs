#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class KalmanFilter(object):
    def __init__(self, x0=0.0, v0=1.0, q=1.0, r=1.0, keep_history=False):
        self.x = np.array([x0])             # estimate of x
        self.v = np.array([v0])             # error estimate
        self.q = q                          # process variance
        self.r = r                          # measurement variance
        self.y = np.array([np.nan])         # observation
        self.__keep_history = keep_history

    def filter(self, y, x0=None, v0=None, q=None, r=None):
        x0_ = (self.x[-1] if x0 is None else x0)
        v0_ = (self.v[-1] if v0 is None else v0)
        assert (v0_ >= 0), f'negative variance: v0_ = {v0_}'
        q_ = (self.q if q is None else q)
        assert (q_ >= 0), f'negative variance: q_ = {q_}'
        r_ = (self.r if r is None else r)
        assert (r_ >= 0), f'negative variance: r_ = {r_}'
        len_y = len(y)
        new_x = np.empty(len_y)
        new_v = np.empty(len_y)
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
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
            self.y = np.array([y[-1]])
        return pd.DataFrame(
            {'y': y, 'x': new_x, 'v': new_v},
            index=(y.index if hasattr(y, 'index') else range(len_y))
        )

    def calculate_log_likelihood(self, *args, **kwargs):
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            try:
                loglik = self.filter(*args, **kwargs).pipe(
                    lambda d: np.sum(
                        np.log(
                            norm.pdf(
                                x=d['y'], loc=d['x'].shift(),
                                scale=np.sqrt(d['v']).shift()
                            )[1:]
                        )
                    )
                )
            except FloatingPointError:
                loglik = -np.inf
        return loglik


class OptimizedKalmanFilter(object):
    def __init__(self, y, x0=None, v0=None, q0=None, r0=None,
                 keep_history=False, q_bound=None, r_bound=None,
                 method='L-BFGS-B', **kwargs):
        self.__logger = logging.getLogger(__name__)
        assert (~np.isnan(y)).all(), f'y contains nan: {y}'
        self.y = y                                  # observation
        self.x0 = (y.mean() if x0 is None else x0)  # estimate of x
        y_var = y.var()
        self.v0 = (y_var if v0 is None else v0)     # error estimate
        self.q0 = (y_var if q0 is None else q0)     # process variance
        self.r0 = (y_var if r0 is None else r0)     # measurement variance
        self.__logger.info(
            f'self.x0: {self.x0}, self.v0: {self.v0}'
            + f', self.q0: {self.q0}, self.r0: {self.r0}'
        )
        self.q_bound = (q_bound or (0, np.inf))
        self.r_bound = (r_bound or (0, np.inf))
        self.method = method
        self.scipy_optimize_minimize_add_kwargs = kwargs
        self.__keep_history = keep_history
        self.q = None
        self.r = None
        self.kf = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_kf(self):
        np.seterr(all='raise')
        res = minimize(
            fun=self._loss, x0=np.array([self.q0, self.r0]),
            args=(self.y, self.x0, self.v0), method=self.method,
            bounds=[self.q_bound, self.r_bound],
            **self.scipy_optimize_minimize_add_kwargs
        )
        if res.success:
            self.__logger.info(f'{os.linesep}{res}')
        else:
            self.__logger.error(f'{os.linesep}{res}')
        assert res.success, res.message
        self.q, self.r = res.x
        self.__logger.info(
            f'process and measurement variances: {self.q}, {self.r}'
        )
        self.kf = KalmanFilter(
            x0=self.x0, v0=self.v0, q=self.q, r=self.r,
            keep_history=self.__keep_history
        )
        self.__logger.debug(
            'vars(self.kf):' + os.linesep + pformat(vars(self.kf))
        )

    @staticmethod
    def _loss(x, *args):
        if (x <= 0).any():
            return np.inf
        else:
            return -KalmanFilter(
                x0=args[1], v0=args[2], q=x[0], r=x[1], keep_history=False
            ).calculate_log_likelihood(y=args[0])

    def filter(self, y=None, optimize=False, **kwargs):
        new_y = (self.y if y is None else y)
        if optimize or not self.kf:
            self.optimize_kf()
        return self.kf.filter(y=new_y, **kwargs)
