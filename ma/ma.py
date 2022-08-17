#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
from scipy.optimize import brute
from scipy.stats import norm


class OptimizedMa(object):
    def __init__(self, y, test_size=None, window_range=None, step_size=1,
                 rolling_kw=None, **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert 0 < self.test_size <= y.size, f'invalid test size: {test_size}'
        self.__logger.info(f'self.test_size: {self.test_size}')
        self.rolling_window_range = (
            slice(window_range[0], window_range[1] + 1, step_size)
            if window_range else slice(2, int(y.size / 2), step_size)
        )
        self.__logger.info(
            f'self.rolling_window_range: {self.rolling_window_range}'
        )
        self.rolling_kw = (rolling_kw or dict())
        self.scipy_optimize_brute_kw = kwargs
        self.rolling_window = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_rolling_window(self):
        if (self.rolling_window_range.start
                == self.rolling_window_range.stop - 1):
            self.rolling_window = self.rolling_window_range.start
        else:
            result = brute(
                func=self._loss, ranges=(self.rolling_window_range,),
                args=(
                    self.y.tail(
                        self.test_size + self.rolling_window_range.stop - 1
                    ),
                    self.test_size, self.rolling_kw
                ),
                finish=None, **self.scipy_optimize_brute_kw
            )
            self.__logger.info(f'result: {result}')
            self.rolling_window = int(result)
        self.__logger.info(f'self.rolling_window: {self.rolling_window}')
        return self.rolling_window

    @staticmethod
    def _loss(x, *args):
        logger = logging.getLogger(__name__)
        window = x[0]
        y, test_size, rolling_kw = args
        with np.errstate(divide='raise', over='raise', under='raise',
                         invalid='raise'):
            rolling = y.rolling(window=window, **rolling_kw)
            try:
                loglik = np.log(
                    norm.pdf(
                        x=y, loc=rolling.mean().shift(),
                        scale=rolling.std(ddof=1).shift()
                    )[-test_size:]
                ).sum()
            except FloatingPointError:
                loss = np.inf
            else:
                loss = -loglik
        logger.info(f'x, loss: {x}, {loss}')
        return loss

    def create_rolling(self, y=None, optimize_rolling_window=False, **kwargs):
        if optimize_rolling_window or not self.rolling_window:
            self.optimize_rolling_window()
        return (self.y if y is None else y).rolling(
            window=self.rolling_window, **self.rolling_kw, **kwargs
        )

    def calculate_ma(self, y=None, **kwargs):
        y_obs = (self.y if y is None else y)
        y_name = (y_obs.name or 'y')
        y_rolling = self.create_rolling(y=y_obs, **kwargs)
        return y_obs.to_frame(name=y_name).assign(
            **{f'{y_name}_ma': y_rolling.mean()}
        )
