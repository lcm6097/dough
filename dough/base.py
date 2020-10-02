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
        

class Stock(DataManipulation):
    def __init__(self,ticker):
        self.symbol = ticker
        self.historical_data = wb.DataReader(self.symbol, data_source='yahoo',
                                             start='1970-1-1')['Close']
    
    def volatility(self):
        # Assumes 5 year volatility
        five_years_ago = (datetime.today()-relativedelta(years=5)).strftime("%Y-%m-%d")
        tday = datetime.today().strftime("%Y-%m-%d")
        temp_df = self._subset_by_dates(self.historical_data, 
                                   five_years_ago, tday)
        temp_var = temp_df.values.std()
        return temp_var
    
    def plot(self, start_date=None, end_date=None):
        return self._subset_by_dates(self.historical_data, start_date, end_date)\
            .plot(figsize=(16,10))
    
    def plot_returns(self, start_date=None, end_date=None):
        return self._calculate_returns(
                self._subset_by_dates(
                    self.historical_data, start_date, end_date))\
            .plot(figsize=(16,10))