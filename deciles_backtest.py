import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os

'''
Earnings Growth Deciles (with Regime filter):
1. download 1 year data (all at once)
2. apply regime filter on price history and 

'''


def get_returns(stock_list, start, end):

    # Initialize df with stocks as columns
    df = pd.DataFrame(columns=stock_list)

    for ticker in stock_list:

        # Fetch data
        df[ticker] = yf.download(ticker, start=start, end=end, auto_adjust=True)['Close']

        # Initial value for holding stock
        df[f'{ticker}_val'] = 10000.0
        
        # Calculate 1 day returns
        df[f'{ticker}_ret'] = df[ticker].pct_change()

    # Fill NaN values with 0.0 for returns
    df.fillna(0.0, inplace=True)

    # Make date a column and index an int
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)

    # Avoid modifying the original DataFrame
    df = df.copy()

    # Get values of holdings
    for i in range(1, len(df)):

        for ticker in stock_list:

            df.loc[i, f'{ticker}_val'] = df[f'{ticker}_val'].iloc[i-1] * (1 + df[f'{ticker}_ret'].iloc[i])

    return df

# Get tickers list
file_name = 'nasdaq100_tickers.txt'
tickers_path = fr"C:\Users\derek\Coding Projects\tickers\{file_name}"
with open(tickers_path, 'r') as f:
    tickers = f.read().splitlines()


print(len(tickers))
# tickers = ['AAPL', 'AMZN']

# Timeframes for training data
end = datetime.today().date()
end = end - timedelta(days=365 * 1)  # 1 year
start = end - timedelta(days=365 * 1)

print(f"Start: {start} | End: {end}")

# Params
factor = 'market_cap'  # 'returns', 'eps_growth', 'revenue_growth', 'profit_growth'

# Download history
df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, group_by='ticker')

# Only select closing prices from inner level
df = df[[col for col in df.columns if col[1] == 'Close']]
# Rename columns to tickers
df.columns = [col[0] for col in df.columns]

print(df)

''' 
Factors to try/combine:
- returns
- eps growth


- principal component analysis (statistical)

* Only use US companies (no Chinese stocks)
'''

# Get returns/pe ratio/earnings growth
returns = {}

for col in df.columns:

    # Percent returns
    one_year_return = (df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]
    returns[col] = one_year_return

n_decile = round(len(tickers) / 10)
print(f"# stocks: {len(tickers)} | # top decile: {n_decile}")

# Get eps growth
eps_growth = {}
revenue_growth = {}
profit_growth = {}

df_eps = pd.DataFrame(columns=tickers)
for ticker in tickers:

    # Stock object
    stock = yf.Ticker(ticker)

    # Don't consider stocks in other countries
    if stock.info.get('country') == 'United States':
    # if 1 == 1:

        try:

            # Earnings YoY
            earnings = stock.get_income_stmt(freq='yearly')
            earnings = earnings.transpose()
            # Remove anything before end timestamp
            # print(earnings)
            # print(type(earnings.index))
            
            # Make start date same type as earnings index
            start_dt = pd.Timestamp(start)

            print("LOOK HERE", start, type(start_dt))
            earnings = earnings[earnings.index >= start_dt]
            # earnings = earnings.loc['2025-01-03':'2025-08-07']
            print(f"Reduced earnings... Check this:\n", earnings)
            
            # Calc eps growth
            eps_change = (earnings['DilutedEPS'].iloc[0] - earnings['DilutedEPS'].iloc[1]) / earnings['DilutedEPS'].iloc[1]
            eps_change = round(eps_change, 3)

            # Calc revenue growth
            revenue_change = (earnings['TotalRevenue'].iloc[0] - earnings['TotalRevenue'].iloc[1]) / earnings['TotalRevenue'].iloc[1]
            revenue_change = round(revenue_change, 3)

            # Calc profit growth
            profit_change = (earnings['NetIncome'].iloc[0] - earnings['NetIncome'].iloc[1]) / earnings['NetIncome'].iloc[1]
            profit_change = round(profit_change, 3)

            # Market cap
            market_cap = stock.info.get('marketCap', 0) 

            # Add to dict
            eps_growth[ticker] = eps_change
            revenue_growth[ticker] = revenue_change
            profit_growth[ticker] = profit_change
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue


if factor == 'returns':
    # Sort returns
    top_decile = dict(sorted(returns.items(), key = lambda x: x[1], reverse = True)[:n_decile])
    bottom_decile = dict(sorted(returns.items(), key = lambda x: x[1], reverse = True)[len(tickers) - n_decile:])
elif factor == 'eps_growth':
    # Sort eps
    top_decile = dict(sorted(eps_growth.items(), key = lambda x: x[1], reverse = True)[:n_decile])
    bottom_decile = dict(sorted(eps_growth.items(), key = lambda x: x[1], reverse = True)[len(tickers) - n_decile:])
elif factor == 'revenue_growth':
    # Sort revenue growth
    top_decile = dict(sorted(revenue_growth.items(), key = lambda x: x[1], reverse = True)[:n_decile])
    bottom_decile = dict(sorted(revenue_growth.items(), key = lambda x: x[1], reverse = True)[len(tickers) - n_decile:])
elif factor == 'profit_growth':
    # Sort profit growth
    top_decile = dict(sorted(profit_growth.items(), key = lambda x: x[1], reverse = True)[:n_decile])
    bottom_decile = dict(sorted(profit_growth.items(), key = lambda x: x[1], reverse = True)[len(tickers) - n_decile:])
elif factor == 'market_cap':
    # Sort by market cap
    top_decile = dict(sorted({k: v for k, v in returns.items() if k in eps_growth}.items(), key=lambda x: x[1], reverse=True)[:n_decile])
    bottom_decile = dict(sorted({k: v for k, v in returns.items() if k in eps_growth}.items(), key=lambda x: x[1], reverse=True)[len(tickers) - n_decile:])

print(f"Best performing stocks:\n{top_decile}")
print(f"Worst performing stocks:\n{bottom_decile}")

list_top = list(top_decile.keys())
list_top.append('SPY')
list_bottom = list(bottom_decile.keys())
list_bottom.append('SPY')

print(f"Top stocks (and SPY): {list_top}")
print(f"Bottom list (and SPY): {list_bottom}")

# Get returns from training set
df_long = get_returns(stock_list=list_top, start=start, end=end)
df_short = get_returns(stock_list=list_bottom, start=start, end=end)

# Time frames for backtest
start_trade = end + timedelta(days=1)
end_trade = start_trade + timedelta(days=30 * 1)  # 1 month

# Get returns for following month
df_long_trade = get_returns(stock_list=list_top, start=start_trade, end=end_trade)
df_short_trade = get_returns(stock_list=list_bottom, start=start_trade, end=end_trade)

if 1 == 1:
    spy_width = 2
    
    # Plot 1
    ax1 = plt.subplot(2, 2, 1)
    for ticker in list_top:
        if ticker == 'SPY':
            ax1.plot(df_long['date'], df_long[f'{ticker}_val'], label=ticker, color='black', linewidth=spy_width)
        else:
            ax1.plot(df_long['date'], df_long[f'{ticker}_val'], label=ticker)
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Date')
    plt.title(f'Long ({start} - {end})')
    plt.legend()
    plt.grid()

    # Plot 2
    ax2 = plt.subplot(2, 2, 2)
    for ticker in list_bottom:
        if ticker == 'SPY':
            ax2.plot(df_short['date'], df_short[f'{ticker}_val'], label=ticker, color='black', linewidth=spy_width)
        else:
            ax2.plot(df_short['date'], df_short[f'{ticker}_val'], label=ticker)
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Date')
    plt.title(f'Short ({start} - {end})')
    plt.tight_layout()
    plt.legend()
    plt.grid()

    # Plot 3
    ax3 = plt.subplot(2, 2, 3)
    for ticker in list_top:
        if ticker == 'SPY':
            ax3.plot(df_long_trade['date'], df_long_trade[f'{ticker}_val'], label=ticker, color='black', linewidth=spy_width)
        else:
            ax3.plot(df_long_trade['date'], df_long_trade[f'{ticker}_val'], label=ticker)
    ax3.set_ylabel('Value')
    ax3.set_xlabel('Date')
    plt.title(f'Long ({start_trade} - {end_trade})')
    plt.tight_layout()
    plt.legend()
    plt.grid()

    # Plot 4
    ax4 = plt.subplot(2, 2, 4)
    for ticker in list_bottom:
        if ticker == 'SPY':
            ax4.plot(df_short_trade['date'], df_short_trade[f'{ticker}_val'], label=ticker, color='black', linewidth=spy_width)
        else:
            ax4.plot(df_short_trade['date'], df_short_trade[f'{ticker}_val'], label=ticker)
    ax4.set_ylabel('Value')
    ax4.set_xlabel('Date')
    plt.title(f'Short ({start_trade} - {end_trade})')
    plt.tight_layout()
    plt.legend()
    plt.grid()

    plt.show()
