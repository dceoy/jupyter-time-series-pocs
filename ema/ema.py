#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
from scipy.optimize import brute
from scipy.stats import norm


class OptimizedEma(object):
    def __init__(self, y, test_size=None, span_range=None, ic='aic',
                 ewm_kw=None, **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert 0 < self.test_size <= y.size, f'invalid test size: {test_size}'
        self.__logger.info(f'self.test_size: {self.test_size}')
        self.ewm_span_range = slice(*(span_range or (2, int(y.size / 2))), 1)
        self.__logger.info(f'self.ewm_span_range: {self.ewm_span_range}')
        assert ic in {'aic', 'bic'}, f'invalid ic: {ic}'
        self.ic = ic
        self.ewm_kw = (ewm_kw or dict())
        self.scipy_optimize_brute_kw = kwargs
        self.ewm_span = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_ewm_span(self):
        result = brute(
            func=self._loss, ranges=(self.ewm_span_range,),
            args=(self.y.tail(self.test_size + 2), self.ic, self.ewm_kw),
            finish=None, **self.scipy_optimize_brute_kw
        )
        self.__logger.info(f'result: {result}')
        self.ewm_span = int(result)
        self.__logger.info(f'self.ewm_span: {self.ewm_span}')
        return self.ewm_span

    @staticmethod
    def _loss(x, *args):
        logger = logging.getLogger(__name__)
        span = x[0]
        y, ic, ewm_kw = args
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            ewm = y.ewm(span=span, **ewm_kw)
            try:
                loglik = np.log(
                    norm.pdf(
                        x=y, loc=ewm.mean().shift(),
                        scale=ewm.std(ddof=1).shift()
                    )[2:]
                ).sum()
            except FloatingPointError:
                loss = np.inf
            else:
                loss = (
                    -2 * loglik
                    + (2 if ic == 'aic' else np.log(y.size - 2)) * span
                )
        logger.info(f'x, loss: {x}, {loss}')
        return loss

    def create_ewm(self, y=None, optimize_ewm_span=False, **kwargs):
        if optimize_ewm_span or not self.ewm_span:
            self.optimize_ewm_span()
        return (self.y if y is None else y).ewm(
            span=self.ewm_span, **self.ewm_kw, **kwargs
        )

    def calculate_ema(self, y=None, **kwargs):
        y_obs = (self.y if y is None else y)
        y_name = (y_obs.name or 'y')
        y_ewm = self.create_ewm(y=y_obs, **kwargs)
        return y_obs.to_frame(name=y_name).assign(
            **{f'{y_name}_ema': y_ewm.mean()}
        )
