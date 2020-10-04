# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:14:47 2020

@author: melul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import cached_property, partial
from base import DataManipulation

class Stock(DataManipulation):
    def __init__(self,ticker):
        self.symbol = ticker
        self.historical_data = pd.DataFrame()
        self.historical_data[ticker] = wb.DataReader(self.symbol, data_source='yahoo',
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
            
    def compare_to_market(self, start_date=None, end_date=None, market_ticker="^GSPC"):
        market_df = self._subset_by_dates(self._get_market_data(market_ticker),start_date,end_date) 
        df = self._subset_by_dates(self.historical_data, start_date, end_date) 
        mx = df.join(market_df)#.plot(figsize=(16,10))
        return (mx/mx.iloc[0]).plot()
        
        
        
        
        