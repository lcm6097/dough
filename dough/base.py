# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:19:27 2020

@author: melul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import cached_property, partial

class DataManipulation:
    def _subset_by_dates(self, df, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.today()-relativedelta(years=5)
        if end_date is None:
            end_date = datetime.today()
        return df[(df.index >= start_date)&(df.index <= end_date)]
    
    def _calculate_returns(self, df, start_date=None, end_date=None, log=False):
        df = self._subset_by_dates(df, start_date, end_date)
        if log is True:
            return np.log(df.pct_change()+1)
        else:
            return df.pct_change()
    
    def _get_market_data(self, start_date=None, end_date=None, market_ticker="^GSPC"):
        return wb.DataReader(market_ticker, data_source='yahoo',
                                             start='1970-1-1')['Close']
        

        
        
        
        
        