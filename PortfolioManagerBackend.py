# -*- coding: utf-8 -*-
"""
Portfolio manager backend

Author: Elias Melul Fresco
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

def import_stock_data(tickers, start_date = '2010-1-1', end_date = datetime.today().strftime('%Y-%m-%d')):
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start_date, end = end_date)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.DataReader(t, data_source='yahoo', start = start_date, end = end_date)['Adj Close']
    return data

def import_stock_dividends(tickers):
    """
    Fetches and returns the dividends of a given stock.
    Due to practical matters, this function will return all dividend history of a stock up until today.
    It cannot be subseted when imported.
    """
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.DataReader(tickers, data_source='yahoo-dividends')['value']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.DataReader(t, data_source='yahoo-dividends')['value']
    return data

def subset_by_dateindex(data, start_date=None, end_date=None, date_position="index"):
    """
    Subsets dataframe by dates (index)
    Inputs:
    start_date: (str) date FROM which to subset
    end_date: (str) date TO which to subset
    date_position: (str) index or name of column in which dates are 
    """
    if date_position != "index":
        data = data.set_index('Date')
        
    if (start_date == None) & (end_date == None):
        return data
    elif (start_date == None) & (end_date != None):
        return data[data.index <= end_date]
    elif (start_date != None) & (end_date == None):
        return data[data.index >= start_date]
    elif (start_date != None) & (end_date != None):
        return data[(data.index >= start_date) & (data.index <= end_date)]
    else:
        print("Wrong input")
        
def sell_order(purchase_history, sell_date, sell_quantity, order = "FIFO"):
    """
    Determines which stocks we are going to sell. 
    First in First Out? (FIFO)
    Last in First Out? (LIFO)
    Cheapest to Most Expensive? Low to High? (L2H)
    """
    def purchase_history_edit(purchase_history, sell_date, sell_quantity):
        row_num = 0
        while sell_quantity != 0:
            ith_row = purchase_history.iloc[row_num]
            ith_quantity = ith_row.Shares
            if sell_quantity == ith_quantity:
                purchase_history.iloc[row_num, 1] = 0
                sell_quantity = 0
            elif (sell_quantity - ith_quantity) < 0:
                ith_quantity_left = abs(sell_quantity - ith_quantity)
                purchase_history.iloc[row_num, 1] = ith_quantity_left
                sell_quantity = 0
                row_num += 1
            elif (sell_quantity - ith_quantity) > 0:
                sell_quantity = sell_quantity - ith_quantity
                purchase_history.iloc[row_num, 1] = 0
                row_num += 1
        return purchase_history
    
    if order == "FIFO":
        purchase_history.sort_values(by='Date', ascending = True, inplace=True)
        return purchase_history_edit(purchase_history, sell_date, sell_quantity)
    
    elif order == "L2H":
        purchase_history.sort_values(by='Price', ascending = True, inplace=True)
        return purchase_history_edit(purchase_history, sell_date, sell_quantity)
    
    elif order == "LIFO":
        purchase_history.sort_values(by='Date', ascending = False, inplace=True)
        return purchase_history_edit(purchase_history, sell_date, sell_quantity)

    elif order == "H2L":
        purchase_history.sort_values(by='Price', ascending = False, inplace=True)
        return purchase_history_edit(purchase_history, sell_date, sell_quantity)
    
class Stock:
    # This class will define the information of a stock
    def __init__(self, ticker):
        self.ticker = ticker
        self.historical_data = import_stock_data(self.ticker, start_date="1970-1-1")
        self.historical_daily_returns = ((self.historical_data/self.historical_data.shift(1))-1)
        try:
            self.dividends = import_stock_dividends(self.ticker)
        except:
            self.dividends = pd.DataFrame({"value":0}, index=['1970-1-1'])
        
    def __str__(self):
        return self.ticker
        
    def plot_stock(self, start_date=None, end_date=None):
        return subset_by_dateindex(self.historical_data, start_date=start_date, end_date=end_date).plot(figsize=(15,8))
    
class Position:
    def __init__(self, Stock):
        self.stock = Stock
        self.shares = 0
        self.equity = 0
        self.invested = 0
        self._max_invested_to_date = 0
        self.market_price = np.nan
        self.unrealized_gain = 0
        self.realized_gain = 0
        self.cash_on_hand = np.nan
        self.purchase_history = pd.DataFrame(columns=["Date","Shares","Price"])
        self.ownership_history = pd.DataFrame(columns=["Date","Ticker","Market_Price","Shares","Equity","Invested","Unrealized_Gains","Realized_Gains","Gains","Max_Invested"])
    
    def __str__(self):
        return "Position for {}".format(self.stock.ticker)
        
    def buy(self, quantity, date):
        self.shares = self.shares + quantity
        self.market_price = subset_by_dateindex(self.stock.historical_data, date, date).iloc[0][0]
        self.invested = self.invested + (quantity*self.market_price)
        if self.invested > self._max_invested_to_date:
            self._max_invested_to_date = self.invested
        self.equity = self.shares*self.market_price
        self.unrealized_gain = self.equity - self.invested
        self.purchase_history = self.purchase_history.append({"Date":date,"Shares":quantity,"Price":self.market_price}, ignore_index=True)
        
        #Realized gains including dividends
        if date in self.stock.dividends.index:
            self.realized_gain = self.realized_gain + (self.shares * subset_by_dateindex(self.stock.dividends, date, date).iloc[0][0])
        #Update ownership history - a dataframe with all relevant stats for every day
        self.ownership_history = self.ownership_history.append({"Date":date,"Ticker":self.stock.ticker,"Market_Price":self.market_price,
                                                               "Shares":self.shares,"Equity":self.equity,"Invested":self.invested,
                                                               "Unrealized_Gains":self.unrealized_gain, "Realized_Gains":self.realized_gain,
                                                               "Gains":(self.unrealized_gain + self.realized_gain),"Max_Invested":self._max_invested_to_date}, ignore_index=True)
        
    def sell(self, quantity, date):
        self.shares = self.shares - quantity
        self.market_price = subset_by_dateindex(self.stock.historical_data, date, date).iloc[0][0]
        self.equity = self.shares*self.market_price    
        self.purchase_history = sell_order(self.purchase_history, date, quantity)
        invested_prior_to_sell = self.invested
        self.invested = sum(self.purchase_history[['Shares','Price']].prod(axis=1))
        self.unrealized_gain = self.equity - self.invested
        
        #Realized gains including dividends
        if date in self.stock.dividends.index:
            self.realized_gain = self.realized_gain + (self.shares * subset_by_dateindex(self.stock.dividends, date, date).iloc[0][0]) + ((quantity*self.market_price)-(invested_prior_to_sell-self.invested))
        else:
            self.realized_gain = self.realized_gain + ((quantity*self.market_price)-(invested_prior_to_sell-self.invested))
        #Update ownership history - a dataframe with all relevant stats for every day
        self.ownership_history = self.ownership_history.append({"Date":date,"Ticker":self.stock.ticker,"Market_Price":self.market_price,
                                                               "Shares":self.shares,"Equity":self.equity,"Invested":self.invested,
                                                               "Unrealized_Gains":self.unrealized_gain, "Realized_Gains":self.realized_gain,
                                                               "Gains":(self.unrealized_gain + self.realized_gain),"Max_Invested":self._max_invested_to_date}, ignore_index=True)
        
        
    def update(self, date):
        self.market_price = subset_by_dateindex(self.stock.historical_data, date, date).iloc[0][0]
        self.equity = self.shares * self.market_price
        self.unrealized_gain = self.equity - self.invested  
        
        #Realized gains including dividends
        if date in self.stock.dividends.index:
            self.realized_gain = self.realized_gain + (self.shares * subset_by_dateindex(self.stock.dividends, date, date).iloc[0][0])
        
        #Update ownership history - a dataframe with all relevant stats for every day
        self.ownership_history = self.ownership_history.append({"Date":date,"Ticker":self.stock.ticker,"Market_Price":self.market_price,
                                                               "Shares":self.shares,"Equity":self.equity,"Invested":self.invested,
                                                               "Unrealized_Gains":self.unrealized_gain, "Realized_Gains":self.realized_gain,
                                                               "Gains":(self.unrealized_gain + self.realized_gain),"Max_Invested":self._max_invested_to_date}, ignore_index=True)
        
        
def load_portfolio(transactions):
    """
    Loads portfolio object from a dataframe of transactions.
    Requires the Stock and Position objects.
    Returns a dictionary with the position of every stock in the portfolio.
    """
    stocks = transactions.Symbol.unique()
    stock_dict = dict(zip(stocks, [None]*len(stocks)))
    for stock_name in stocks:
        stock_transactions = transactions[transactions['Symbol'] == stock_name]
        earliest_date = min(stock_transactions['Open date'])
        position = Position(Stock(stock_name))
        stock_dates = position.stock.historical_data.index
        for day_active_market in list(stock_dates[stock_dates >= earliest_date]):
            day_transaction = stock_transactions[stock_transactions['Open date'] == str(day_active_market)]
            if len(day_transaction) == 1:
                purchase_type = day_transaction.Type.values
                quantity = day_transaction.Qty.values[0]
                if purchase_type == "Buy":
                    position.buy(quantity, day_active_market)
                    print(f"You bought {quantity} shares of {stock_name} on {str(day_active_market)[:10]} for ${round(position.market_price,2)}")
                else:
                    position.sell(quantity, day_active_market)
                    print(f"You sold {quantity} shares of {stock_name} on {str(day_active_market)[:10]} for ${round(position.market_price,2)}")
            else:
                position.update(day_active_market)
        stock_dict[stock_name] = position
    return stock_dict

def get_portfolio_feature_data(portfolio, feature, return_type = 'perStock'):
    """
    Returns dataframe with the specified feature of each stock or a combination of all stocks
    Input:
    1. portfolio: a Portfolio object
    2. feature: the feature in the ownership_history dataframe within a portfolio object to return
    3. return_type: either 'perStock' or 'combined' - returns a dataframes with the features with a column for every stock or a summation of the values of all stocks
    """
    stocks = list(portfolio.stock_positions.keys())
    df_to_return = pd.DataFrame()
    for s in stocks:
        stock_column = portfolio.stock_positions[s].ownership_history.set_index('Date')[[feature]].rename(columns={feature:s})
        stock_column.replace(0,np.nan,inplace=True)
        df_to_return = pd.concat([df_to_return, stock_column], sort=False, axis=1)
    if return_type == 'perStock':
        return df_to_return
    elif return_type == 'combined':
        string = "Total_"+str(feature)
        return pd.DataFrame(df_to_return.sum(axis=1), columns=[string])
    return df_to_return

class Portfolio:
    def __init__(self, transactions):
        self.stock_positions = load_portfolio(transactions)
        
    def plot_portfolio_feature(self, feature):
        return get_portfolio_feature_data(self, feature, return_type='combined').plot(figsize=(15,8))
    
    def plot_feature_perStock(self, feature):
        return get_portfolio_feature_data(self, feature, return_type='perStock').plot(figsize=(15,8))
    
    def plot_portfolio(self):
        invested = get_portfolio_feature_data(self, 'Invested', 'combined')
        maxinvested = max(invested['Total_Invested'])
        combined_return = get_portfolio_feature_data(self, 'Gains', 'combined')/maxinvested
        return combined_return.plot(figsize=(15,8))
        
    def portfolio_from_to(self, start_date=None, end_date=None, market_ticker = "^GSPC", plotType = "performance"):
        if plotType == "performance":
            invested = get_portfolio_feature_data(self, 'Max_Invested', 'combined')
            invested = subset_by_dateindex(invested, start_date, end_date)

            unr_gain_at_dates = subset_by_dateindex(get_portfolio_feature_data(self, 'Gains', 'combined'), start_date, end_date)
            unr_gain_at_dates = unr_gain_at_dates.merge(invested, left_index=True, right_index=True)
            rets = unr_gain_at_dates['Total_Gains']/unr_gain_at_dates['Total_Max_Invested']

            adj_gains = rets-rets.iloc[0]

            if start_date == None:
                start_date = min(unr_gain_at_dates.index)
            market = import_stock_data(market_ticker, start_date, end_date)
            adj_market = market-market.iloc[0]

            with_market = pd.DataFrame(adj_gains, columns=['Total_Return'])
            with_market[market_ticker] = adj_market/market.iloc[0]

            ret_dataframe = pd.DataFrame({"Portfolio":[f"{round(with_market['Total_Return'].iloc[-1]*100,2)}%"],
                                         "Market":[f"{round(with_market[market_ticker].iloc[-1]*100,2)}%"]}, 
                                         index=['Return'])

            #Show Plot
            with_market.plot(figsize=(15,8))
            plt.show()

            return ret_dataframe
                                                       
        else:
            invested = get_portfolio_feature_data(self, 'Invested', 'combined')
            invested = subset_by_dateindex(invested, start_date, end_date)
            maxinvested = max(invested['Total_Invested'])
            unr_gain_at_dates = subset_by_dateindex(get_portfolio_feature_data(self, 'Gains', 'combined'), start_date, end_date)
            print(unr_gain_at_dates)
            adj_gains = unr_gain_at_dates-unr_gain_at_dates.iloc[0]
            combined_return = adj_gains/maxinvested

            if start_date == None:
                start_date = min(unr_gain_at_dates.index)
            market = import_stock_data(market_ticker, start_date, end_date)
            adj_market = market-market.iloc[0]

            with_market = combined_return
            with_market[market_ticker] = adj_market/market.iloc[0]

            ret_dataframe = pd.DataFrame({"Portfolio":[f"${round(adj_gains.iloc[-1][0],2)}",
                                                       f"{round(combined_return.iloc[-1][0]*100,2)}%"],
                                         "Market":[' ',
                                                   f"{round((100*adj_market/market.iloc[0]).iloc[-1][0],2)}%"]}, 
                                         index=['Gains','Return'])

            #Show Plot
            with_market.plot(figsize=(15,8))
            plt.show()

            return ret_dataframe
    
