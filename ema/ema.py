#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
from scipy.stats import norm


class OptimizedEma(object):
    def __init__(self, y, test_size=None, span_range=None, adjust=False,
                 ignore_na=True, **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert self.test_size > 0, f'invalid test size: {self.test_size}'
        self.__logger.info(f'self.test_size: {test_size}')
        assert y.size > self.test_size, f'too short y: {y.size}'
        self.ewm_span_range = (span_range or (2, int(y.size / 2)))
        self.__logger.info(f'self.ewm_span_range: {self.ewm_span_range}')
        self.ewm_add_kwargs = {
            'adjust': adjust, 'ignore_na': ignore_na, **kwargs
        }
        self.ewm_span = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_ewm_span(self):
        np.seterr(all='raise')
        loss = pd.Series({
            i: self._loss(
                span=i, y=self.y, test_size=self.test_size,
                **self.ewm_add_kwargs
            ) for i in range(*self.ewm_span_range)
        })
        self.__logger.info(f'loss: {loss}')
        self.ewm_span = loss.idxmin()
        self.__logger.info(f'self.ewm_span: {self.ewm_span}')

    @staticmethod
    def _loss(span, y, test_size, **kwargs):
        ewm = y.ewm(span=span, **kwargs)
        return (
            -2 * np.sum(
                np.log(
                    norm.pdf(
                        x=y.tail(test_size),
                        loc=ewm.mean().shift().tail(test_size),
                        scale=ewm.std(ddof=1).shift().tail(test_size)
                    )
                )
            ) + np.log(test_size) * span
        )

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
