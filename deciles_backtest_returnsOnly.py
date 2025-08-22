import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import time
import os

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

'''
Ideas: regime filter on underlying index. I.e. if nasdaq is in bull market, go long. If bear, go short. If flat, both?
'''

def get_returns(stock_list, start, end, start_value=10000.0):
    """ Performs a backtest on stocks in given date range with yfinance

    Keyword arguments:
    stock_list - list of tickers
    start - datetime start backtest
    end: datetime end backtest

    return - DataFrame of returns
    """
    start_value = float(start_value)

    # Initialize df with stocks as columns
    df = pd.DataFrame(columns=stock_list)

    for ticker in stock_list:

        # Fetch data
        df[ticker] = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']

        # Initial value for holding stock
        df[f'{ticker}_val'] = start_value
        
        # Calculate 1 day returns
        df[f'{ticker}_ret'] = df[ticker].pct_change()

    # Fill NaN values with 0.0 for returns
    df.fillna(float(0.0), inplace=True)

    # Make date a column and index an int
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # Avoid modifying the original DataFrame
    df = df.copy()

    # Get values of holdings
    for i in range(1, len(df)):

        for ticker in stock_list:

            df.loc[i, f'{ticker}_val'] = df[f'{ticker}_val'].iloc[i-1] * (1.0 + df[f'{ticker}_ret'].iloc[i])

    return df

# Get tickers list
file_name = 'nasdaq100_tickers.txt'

try:
    tickers_path = fr"C:\Users\derek\Coding Projects\tickers\{file_name}"
    with open(tickers_path, 'r') as f:
        tickers = f.read().splitlines()
except:
    tickers_path = fr"C:\Users\eklundmd\OneDrive - Thermo Fisher Scientific\Desktop\Python Projects\Learning\backtest\{file_name}"
    with open(tickers_path, 'r') as f:
        tickers = f.read().splitlines()

print(f"Tickers ({len(tickers)}):\n{tickers}\n")

# Timeframes for training data
# end = datetime.today() #.date()

top_returns = {} # Store returns for each month

start_backtest = 365
lookback = 150
while start_backtest >= 0:
    end = datetime.today() - timedelta(days=start_backtest * 1)  # 1 year
    start = end - timedelta(days=365 * 1)
    print(f"Prev Year Returns -- Start: {start} | End: {end}")

    # Download history
    df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, group_by='ticker')
    df_spy = yf.download('SPY', start=start, end=end, auto_adjust=True, multi_level_index=False)

    # Only select closing prices from inner level
    df = df[[col for col in df.columns if col[1] == 'Close']]
    df_spy['SPY'] = df_spy['Close']

    # Rename columns to tickers
    df.columns = [col[0] for col in df.columns]

    # Get returns/pe ratio/earnings growth
    returns = {}

    for col in df.columns:

        # Percent returns
        one_year_return = (df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]
        returns[col] = one_year_return

    n_decile = round(len(tickers) / 10)
    print(f"# stocks: {len(tickers)} | # top in top decile: {n_decile}")

    # Sort returns
    top_decile = dict(sorted(returns.items(), key = lambda x: x[1], reverse = True)[:n_decile])
    bottom_decile = dict(sorted(returns.items(), key = lambda x: x[1], reverse = True)[len(tickers) - n_decile:])
    print(f"Best performing stocks:\n{top_decile}")
    list_top = list(top_decile.keys())
    print(list_top)

    # Get returns from training set
    dt_start = end + timedelta(days=1) # Next day = start
    dt_end = dt_start + timedelta(days=30) # 1 month
    df_backtest = get_returns(stock_list=list_top, start=dt_start, end=dt_end)
    columns = df_backtest.columns.tolist()
    columns = [c for c in columns if '_val' in c]
    df_backtest = df_backtest[columns] # Reduce to stock value columns
    start_port_val = 0
    end_port_val = 0

    # Sum start and end backtest values
    for col in columns:
        start_port_val += df_backtest[col].iloc[0]
        end_port_val += df_backtest[col].iloc[-1]

    port_return = (end_port_val - start_port_val) / start_port_val * 100
    port_return = round(port_return, 2)

    # Get SPY to compare strategy returns
    df_spy_backtest = get_returns(stock_list=['SPY'], start=dt_start, end=dt_end, start_value=100000)
    # print(df_spy_backtest)

    # Returns 
    start_spy_val = df_spy_backtest['SPY_val'].iloc[0]
    end_spy_val = df_spy_backtest['SPY_val'].iloc[-1]
    spy_return = (end_spy_val - start_spy_val) / start_spy_val * 100
    spy_return = round(spy_return, 2)
    print(f"{"*"*25}")
    print(f"Prev Year Returns -- Start: {start} | End: {end}")
    print(f"Backtest -- start: {dt_start} | End: {dt_end}")
    print(f"Portfolio value: ${end_port_val:,.2f} ({port_return}%) | SPY value: ${end_spy_val:,.2f} ({spy_return}%)")
    print(f"{"*"*25}")

    # Determine regime
    print(df_spy)
    start_spy_val_yr = df_spy['SPY'].iloc[0]
    end_spy_val_yr = df_spy['SPY'].iloc[-1]
    midway_val_yr = df_spy['SPY'].iloc[lookback]
    print(f"Start: {start_spy_val_yr} | Mid: {midway_val_yr} | End: {end_spy_val_yr}")
    if end_spy_val_yr > midway_val_yr:
        regime = "Bull"
    elif end_spy_val_yr < midway_val_yr:
        regime = "Bear"
    else:
        regime = "Flat"

    # Add to returns dict
    dates_str = dt_start.strftime("%Y-%m-%d") + " to " + dt_end.strftime("%Y-%m-%d")
    return_str = str(f"Port ${end_port_val:,.2f} | SPY ${end_spy_val:,.2f} [{regime}]")
    top_returns[dates_str] = return_str

    # Move to next month
    start_backtest -= 30

print(top_returns)
