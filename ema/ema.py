#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
from scipy.optimize import brute
from scipy.stats import norm


class OptimizedEma(object):
    def __init__(self, y, test_size=None, span_range=None, adjust=False,
                 ignore_na=True, **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert self.test_size > 0, f'invalid test size: {test_size}'
        self.__logger.info(f'self.test_size: {self.test_size}')
        assert y.size > self.test_size, f'too short y: {y.size}'
        self.ewm_span_range = slice(*(span_range or (2, int(y.size / 2))), 1)
        self.__logger.info(f'self.ewm_span_range: {self.ewm_span_range}')
        self.ewm_add_kwargs = {'adjust': adjust, 'ignore_na': ignore_na}
        self.scipy_optimize_brute_add_kwargs = kwargs
        self.ewm_span = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_ewm_span(self):
        np.seterr(all='raise')
        result = brute(
            func=self._loss, ranges=(self.ewm_span_range,),
            args=(self.y, self.test_size, self.ewm_add_kwargs),
            finish=None, **self.scipy_optimize_brute_add_kwargs
        )
        self.__logger.info(f'result: {result}')
        self.ewm_span = int(result)
        self.__logger.info(f'self.ewm_span: {self.ewm_span}')

    @staticmethod
    def _loss(x, *args):
        span = x[0]
        y, test_size, ewm_add_kwargs = args
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            ewm = y.ewm(span=span, **ewm_add_kwargs)
            try:
                loglik = np.sum(
                    np.log(
                        norm.pdf(
                            x=y.tail(test_size),
                            loc=ewm.mean().shift().tail(test_size),
                            scale=ewm.std(ddof=1).shift().tail(test_size)
                        )
                    )
                )
            except FloatingPointError:
                loglik = -np.inf
        return (-2 * loglik + 2 * span)

    def calculate_ema(self, y=None, optimize_ewm_span=False, **kwargs):
        if optimize_ewm_span or not self.ewm_span:
            self.optimize_ewm_span()
        y_obs = (self.y if y is None else y)
        y_name = (y_obs.name or 'y')
        return y_obs.to_frame(name=y_name).assign(
            **{
                f'{y_name}_ema': y_obs.ewm(
                    span=self.ewm_span, **self.ewm_add_kwargs, **kwargs
                ).mean()
            }
        )
