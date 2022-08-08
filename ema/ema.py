#!/usr/bin/env python

import logging
import os
from pprint import pformat

import numpy as np
import pandas as pd
from scipy.optimize import brute
from scipy.stats import norm


class OptimizedEma(object):
    def __init__(self, y, test_size=None, ewm_span_range=None,
                 adjust=False, ignore_na=True, **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = int(test_size or (y.size / 2))
        assert self.test_size > 0, f'invalid test size: {self.test_size}'
        self.__logger.info(f'self.test_size: {test_size}')
        assert y.size > self.test_size, f'too short y: {y.size}'
        self.ewm_span_range = (ewm_span_range or slice(1, int(y.size / 2), 1))
        self.__logger.info(f'self.ewm_span_range: {ewm_span_range}')
        self.ewm_adjust = adjust
        self.ewm_ignore_na = ignore_na
        self.scipy_optimize_brute_add_kwargs = kwargs
        self.ewm_span = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_ewm_span(self):
        np.seterr(all='raise')
        # self.ewm_span = brute(
        #     func=self._loss, ranges=(self.ewm_span_range,),
        #     args=(
        #         self.y, self.test_size,
        #         {'adjust': self.ewm_adjust, 'ignore_na': self.ewm_ignore_na}
        #     ),
        #     finish=None, **self.scipy_optimize_brute_add_kwargs
        # )
        loss = pd.Series({
            i: self._loss(
                (i,), self.y, self.test_size,
                {'adjust': self.ewm_adjust, 'ignore_na': self.ewm_ignore_na}
            ) for i in range(self.ewm_span_range.stop)[self.ewm_span_range]
        })
        self.__logger.info(f'loss: {loss}')
        self.ewm_span = loss.idxmin()
        self.__logger.info(f'self.ewm_span: {self.ewm_span}')

    @staticmethod
    def _loss(x, *args):
        # return np.square(
        #     args[0].tail(args[1])
        #     - args[0].ewm(span=x[0], **args[2]).mean().shift().tail(args[1])
        # ).mean()
        ewm = args[0].ewm(span=x[0], **args[2])
        return -np.sum(
            np.log(
                norm.pdf(
                    x=args[0].tail(args[1]),
                    loc=ewm.mean().shift().tail(args[1]),
                    scale=ewm.std(ddof=1).shift().tail(args[1])
                )
            )
        )

    def calculate_ema(self, y=None, optimize_ewm_span=False, **kwargs):
        if optimize_ewm_span or not self.ewm_span:
            self.optimize_ewm_span()
        y_obs = (self.y if y is None else y)
        y_name = (y_obs.name or 'y')
        return y_obs.to_frame(name=y_name).assign(
            **{
                f'{y_name}_ema': y_obs.ewm(
                    span=self.ewm_span, adjust=self.ewm_adjust,
                    ignore_na=self.ewm_ignore_na, **kwargs
                ).mean()
            }
        )
