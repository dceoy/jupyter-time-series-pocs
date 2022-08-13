#!/usr/bin/env python

import logging
import os
from pprint import pformat
from warnings import catch_warnings, filterwarnings

import numpy as np
from scipy.optimize import brute
from statsmodels.tsa.arima.model import ARIMA


class OptimizedArima(object):
    def __init__(self, y, test_size=None, p_range=(0, 1), d_range=(0, 1),
                 q_range=(0, 1), ic='bic', model_kw=None, fit_kw=None,
                 **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.y = y      # observation
        self.test_size = (test_size or int(y.size / 2))
        assert 0 < self.test_size < y.size, f'invalid test size: {test_size}'
        self.__logger.info(f'self.test_size: {self.test_size}')
        self.arima_order_ranges = tuple(
            slice(t[0], t[1] + 1, 1) for t in [p_range, d_range, q_range]
        )
        self.__logger.info(
            f'self.arima_order_ranges: {self.arima_order_ranges}'
        )
        self.ic = ic
        self.model_kw = (model_kw or dict())
        self.fit_kw = (fit_kw or dict())
        self.scipy_optimize_brute_add_kwargs = kwargs
        self.arima_order = None
        self.arima = None
        self.__logger.debug('vars(self):' + os.linesep + pformat(vars(self)))

    def optimize_arima_order(self):
        result = brute(
            func=self._loss, ranges=self.arima_order_ranges,
            args=(self.y, self.test_size, self.model_kw, self.fit_kw, self.ic),
            **self.scipy_optimize_brute_add_kwargs
        )
        self.__logger.debug(f'result: {result}')
        self.arima_order = tuple(int(i) for i in result)
        self.__logger.info(f'self.arima_order: {self.arima_order}')
        return self.arima_order

    @staticmethod
    def _loss(x, *args):
        y, test_size, model_kw, fit_kw, ic = args
        with (np.errstate(divide='raise', over='raise', under='raise',
                          invalid='raise'),
              catch_warnings()):
            filterwarnings('ignore')
            try:
                mod = ARIMA(y.tail(test_size), order=x, **model_kw)
            except ValueError:
                loss = np.inf
            else:
                try:
                    res = mod.fit(**fit_kw)
                except np.linalg.LinAlgError:
                    loss = np.inf
                else:
                    loss = getattr(res, ic)
        logger = logging.getLogger(__name__)
        logger.debug(f'x, loss: {x}, {loss}')
        return loss

    def fit_parameters(self, y=None, optimize_arima_order=False, model_kw=None,
                       fit_kw=None):
        if optimize_arima_order or not self.arima_order:
            self.optimize_arima_order()
        self.arima = ARIMA(
            (self.y if y is None else y), order=self.arima_order,
            **self.model_kw, **(model_kw or dict())
        )
        self.__logger.debug(f'self.arima: {self.arima}')
        res = self.arima.fit(**self.fit_kw, **(fit_kw or dict()))
        self.__logger.debug(f'res: {res}')
        self.__logger.info('res.summary(): {}'.format(res.summary()))
        return res

    def predict_ci(self, y=None, optimize_arima_order=False, model_kw=None,
                   fit_kw=None, get_prediction_kw=None, **kwargs):
        return self.fit_parameters(
            y=y, optimize_arima_order=optimize_arima_order, model_kw=model_kw,
            fit_kw=fit_kw
        ).get_prediction(**get_prediction_kw).summary_frame(**kwargs)

    def forecast_ci(self, y=None, optimize_arima_order=False, model_kw=None,
                    fit_kw=None, get_forecast_kw=None, **kwargs):
        return self.fit_parameters(
            y=y, optimize_arima_order=optimize_arima_order, model_kw=model_kw,
            fit_kw=fit_kw
        ).get_forecast(**get_forecast_kw).summary_frame(**kwargs)