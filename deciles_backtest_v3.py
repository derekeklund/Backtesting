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
Earnings Growth Deciles (with Regime filter):
1. download 1 year data (all at once)
2. apply regime filter on price history
3. buy top decile, short bottom decile

decile_combos.py
- choose from following factors: market_cap, returns, eps_growth, revenue_growth, profit_growth (5 to start)
- add filters (regime + volvol + kalman, share dilution)
- generic build to modify to alpaca bot
- select from nasdaq100 + s&p500 list
'''

def get_returns(stock_list, start, end, start_value=10000.0):
    """ Performs a backtest on stocks in given date range with yfinance

    Keyword arguments:
    stock_list - list of tickers
    start - datetime start backtest
    end: datetime end backtest

    return - DataFrame of returns
    """

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
index = "russel2000" # "nasdaq100", "nasdaq100_test", "nyse", "sp500", "russel2000"
file_name = f"{index}_tickers.txt"

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
end = datetime.today() #.date()
end = end - timedelta(days=365 * 1)  # 1 year
start = end - timedelta(days=365 * 1)
start = "2023-01-01" # 2022 - 2024 (only goes back to 2022 w/ yfinance limitations)
end = "2023-12-31" # 2022 - 2024
print(f"Start: {start} | End: {end}")

# Input params
# tickers = ['NVDA', 'GOOGL', 'MSFT'] # above
# tickers = tickers[:10]  # for testing
# tickers = ['NVDA']
# tickers = ['ASML', 'AMD']
# start = '2024-01-01' # above
# end = '2024-12-31' # above

# Download history
df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, group_by='ticker')

# Only select closing prices from inner level
df = df[[col for col in df.columns if col[1] == 'Close']]

# Rename columns to tickers
df.columns = [col[0] for col in df.columns]

# print(df)
# time.sleep(999)

# Empty df
columns = ['Ticker', 'Date', 'MarketCap', 'Returns1year', 'EpsGrowth', 'RevenueGrowth', 'ProfitGrowth']
df_deciles = pd.DataFrame(columns=columns)

errors = {}

for ticker in tqdm(tickers):

    # Stock object
    stock = yf.Ticker(ticker)

    # For paper trading
    if 1 == 0:
        '''
        stock.info gives *current* info and can't be used for backtests
        but can be used to get trailing info for live bots

        Info from this dictionary:
        - country - filter out non US companies
        - industry - filter for/out cerain industries (semis, etc)
        - sector - filter for/out certain sectors (tech, etc)
        - longBusinessSummary - this could be good for the language processing ML scripts for later. Parse out what kind of business they do. Train a model to understand "scam/hype" companies ()
        - beta - the stock's price volatility relative to overall market (S&P500 = 1.0). Higher beta means higher swings up/down, not necessarily returns (therefore a better indicator of short-term risk than long-term)
        - trailingPE, forwardPE
        - averageVolume, averageVolume10days, averageDailyVolume10Day
        - marketCap
        - profitMargins
        - sharesOutstanding + sharesShort (get %)
        - priceToBook
        - trailingEps, forwardEps
        - 52WeekChange (1 year returns?)
        - totalRevenue
        - debtToEquity
        - revenuePerShare
        - returnOnEquity
        - grossProfits
        - freeCashFlow, operatingCashFlow
        - earningsGrowth
        - revenueGrowth
        - grossMargins
        - epsTrailingTwelveMonths, epsForward, epsCurrentYear
        - averageAnalystRating
        - trailingPegRatio
        '''
        for k,v in stock.info.items():
            print(k,v)

    # print(stock.history()) # HLOC, volume
    # print(stock.get_history_metadata()) # not helpful
    # print(stock.actions)  # dividend/split history df
    # print(stock.fast_info) # not helpful
    # print(stock.news) # helpful for language processing
    # print(stock.get_sec_filings()) # maybe parse these
    # balance_sheet = stock.balance_sheet.transpose()
    # print(balance_sheet.columns.to_list())
    # print(stock.financials)

    '''
    Helpful historical income statement columns:
    - 
    - 'DilutedAverageShares' [multiply by price at that date to get marketCap]
    - 'DilutedEPS', 'BasicEPS'
    - 'GrossProfit', 'NetIncome', ' NormalizedIncome', 'Net Income Continuous Operations', 'Operating Income'
    - 'TotalRevenue', 'Operating Revenue', 'Cost Of Revenue'
    '''
    try:
        # Get income statement (last couple years)
        df = stock.get_income_stmt(freq='yearly').transpose()

        # Remove rows beyond backtest end date
        df['datetime'] = pd.to_datetime(df.index)
        df = df[df['datetime'] < end]

        df.reset_index()

        # Get index dates for price history download
        dates = df.index.tolist()
        start_date = dates[1]
        start_historical = start_date - timedelta(days=3)
        end_date = dates[0]
        end_historical = end_date + timedelta(days=3)

        # Download prices
        df_prices = yf.download(ticker, start=start_historical, end=end_historical, auto_adjust=True, progress=False)

        # Reduce to just closes
        df_prices = df_prices[[col for col in df_prices.columns if col[0] == 'Close']]

        # Make single index
        df_prices.columns = [col[0] for col in df_prices.columns]

        # Create Close column
        df['Close'] = pd.Series(np.nan, index=df.index, dtype='float64') # Set float type

        # Get index of closest prices
        start_price_index = df_prices.index.get_indexer([start_date], method='nearest')[0]
        end_price_index = df_prices.index.get_indexer([end_date], method='nearest')[0]

        # Get prices from prices df
        start_val = df_prices.iloc[start_price_index]['Close']
        end_val = df_prices.iloc[end_price_index]['Close']

        # Add prices to income stmt df
        df.loc[start_date, 'Close'] = start_val
        df.loc[end_date, 'Close'] = end_val

        # new df?
        # df = df

        # Add market cap
        df['MarketCap'] = df['DilutedAverageShares'] * df['Close']

        # Reverse index dates
        df = df.iloc[::-1]

        # Calculate YoY (growth) values
        df['Returns1year'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['EpsGrowth'] = (df['DilutedEPS'] - df['DilutedEPS'].shift(1)) / df['DilutedEPS'].shift(1)
        df['ProfitGrowth'] = (df['GrossProfit'] - df['GrossProfit'].shift(1)) / df['GrossProfit'].shift(1)
        df['RevenueGrowth'] = (df['TotalRevenue'] - df['TotalRevenue'].shift(1)) / df['TotalRevenue'].shift(1)

        # Add columns, reset index
        df['Date'] = df.index
        df['Ticker'] = ticker
        df = df.reset_index()
        
        # Turn last row into dict
        latest_info = df.iloc[-1]
        company_factors = latest_info.to_dict()

        # Add dict values as new df row
        df_deciles.loc[len(df_deciles)] = company_factors
    except Exception as e:
        errors[ticker] = e


# Rate/rank each factor
df_deciles['mc_rank'] = df_deciles['MarketCap'].rank(method='average')
df_deciles['ret_rank'] = df_deciles['Returns1year'].rank(method='average')
df_deciles['eps_rank'] = df_deciles['EpsGrowth'].rank(method='average')
df_deciles['rev_rank'] = df_deciles['RevenueGrowth'].rank(method='average')
df_deciles['prof_rank'] = df_deciles['ProfitGrowth'].rank(method='average')

# Combinations
df_deciles['rev_prof_rank'] = df_deciles[['rev_rank', 'prof_rank']].sum(axis=1)
df_deciles['ret_prof_rank'] = df_deciles[['ret_rank', 'prof_rank']].sum(axis=1)
df_deciles['ret_rev_rank'] = df_deciles[['ret_rank', 'rev_rank']].sum(axis=1)

# Composite score
df_deciles['Composite'] = df_deciles[['mc_rank', 'ret_rank', 'eps_rank', 'rev_rank', 'prof_rank']].sum(axis=1)

print(df_deciles)

# Top and bottom deciles
factor = 'rev_rank' # Input param - mc_rank, ret_rank, eps_rank, rev_rank, prof_rank, Composite
factors = ['mc_rank', 'ret_rank', 'eps_rank', 'rev_rank', 'prof_rank', 'rev_prof_rank', 'ret_prof_rank', 'ret_rev_rank', 'Composite']
factors = ['ret_rank', 'eps_rank', 'rev_rank', 'prof_rank', 'Composite']
# factors = ['rev_rank']
n_decile = 10 # 10 percent
n_decile = round(len(tickers) / 10)
print(f"# stocks: {len(tickers)} | # top decile: {n_decile}")


# Download history
# start = "2025-01-01"
# end = "2025-08-22"
print(f"OG start date: {start} | OG end date: {end}")
start = datetime.strptime(start, "%Y-%m-%d").date() + timedelta(days=366)
end = datetime.strptime(end, "%Y-%m-%d").date() + timedelta(days=366)
if end > datetime.today().date():
    end = datetime.today().date()
print(f"Backtest start date: {start} | Backtest end date: {end}")
time.sleep(10)

# Test all factors
for factor in factors:
    # Sort by factor
    df_deciles = df_deciles.sort_values(by=factor, ascending=False)
    # Get top decile
    list_top_decile = df_deciles.iloc[:n_decile, df_deciles.columns.get_loc('Ticker')].tolist()
    list_bottom_decile = df_deciles.iloc[-n_decile:, df_deciles.columns.get_loc('Ticker')].tolist()
    list_middle = df_deciles.iloc[n_decile:-n_decile, df_deciles.columns.get_loc('Ticker')].tolist()
    # print(list_bottom_decile)

    all_returns_dict = {} # all tickers 

    # Get returns for middle decile (for comparison)
    middle_returns_dict = {}
    errors = {}
    for ticker in list_middle:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, interval="1mo", progress=False)['Close']
            returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
            middle_returns_dict[ticker] = returns.iloc[0]

            all_returns_dict[ticker] = returns.iloc[0]
        except Exception as e:
            errors[ticker] = str(e)
            # print(f"Error fetching {ticker}: {e}")

    # print(f"Errors: {errors}")

    # Sum returns
    total_middle_return = 0.0
    for k,v in middle_returns_dict.items():
        v = round(v, 4)
        # print(f"{k}: {v}")
        total_middle_return += v

    total_middle_return = round(total_middle_return, 4)
    middle_average = total_middle_return / len(middle_returns_dict)
    middle_average = round(middle_average, 4)

    # Get returns for top decile
    top_returns_dict = {}
    errors = {}
    for ticker in list_top_decile:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, interval="1mo", progress=False)['Close']
            returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
            top_returns_dict[ticker] = returns.iloc[0]

            all_returns_dict[ticker] = returns.iloc[0]
        except Exception as e:
            errors[ticker] = str(e)
            # print(f"Error fetching {ticker}: {e}")

    # print(f"Errors: {errors}")

    # Sum returns
    total_top_return = 0.0
    for k,v in top_returns_dict.items():
        v = round(v, 4)
        # print(f"{k}: {v}")
        total_top_return += v

    total_top_return = round(total_top_return, 4)
    top_average = total_top_return / len(top_returns_dict)
    top_average = round(top_average, 4)

    # Get returns for bottom decile
    bottom_returns_dict = {}
    errors = {}
    for ticker in list_bottom_decile:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, interval="1mo", progress=False)['Close']
            returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
            bottom_returns_dict[ticker] = returns.iloc[0]

            all_returns_dict[ticker] = returns.iloc[0]
        except Exception as e:
            errors[ticker] = str(e)
    #         print(f"Error fetching {ticker}: {e}")

    # print(f"Errors: {errors}")

    # Sum returns
    total_bottom_return = 0.0
    for k,v in bottom_returns_dict.items():
        v = round(v, 4)
        # print(f"{k}: {v}")
        total_bottom_return += v

    total_bottom_return = round(total_bottom_return, 4)
    bottom_average = total_bottom_return / len(bottom_returns_dict)
    bottom_average = round(bottom_average, 4)

    # Sum all returns
    total_all_return = 0.0
    for k,v in all_returns_dict.items():
        v = round(v, 4)
        # print(f"{k}: {v}")
        total_all_return += v

    total_all_return = round(total_all_return, 4)
    all_average = total_all_return / len(all_returns_dict)
    all_average = round(all_average, 4)

    print(f"{"~"*40}")
    print(f"Backtest for ***{factor}*** from {start} to {end}")
    print(f"All stocks total return: {total_all_return} | avg: {all_average} ({len(all_returns_dict)} tickers)")
    print(f"{"~"*20}")
    print(f"Top decile total return: {total_top_return} | avg: {top_average} ({len(top_returns_dict)} tickers) {top_returns_dict.keys()}")
    print(f"Middle decile total return: {total_middle_return} | avg: {middle_average} ({len(middle_returns_dict)} tickers)")
    print(f"Bottom decile total return: {total_bottom_return} | avg: {bottom_average} ({len(bottom_returns_dict)} tickers) {bottom_returns_dict.keys()}")

    # Bar chart of averages
    labels = ['Top Decile', 'Middle 80%', 'Bottom Decile']
    averages = [top_average, middle_average, bottom_average]
    plt.clf() # Clear previous plot
    plt.bar(labels, averages, color=['green', 'blue', 'red'])
    plt.ylabel(f'Average Return')
    plt.suptitle(f'{start} to {end}')
    plt.title(f'{index} - {factor} {all_average*100}% total return ({len(all_returns_dict)} stocks)')
    # plt.ylim(min(averages) * 1.1, max(averages) * 1.1)
    for i, v in enumerate(averages):
        plt.text(i, v + 0.001, str(round(v*100, 4)) + "%", ha='center', fontweight='bold')
    # plt.show()

    # Save plot
    output_dir = r"C:\Users\derek\Coding Projects\Cream\statistics"
    plt.savefig(f"{output_dir}\\{index}_{factor}_deciles_{start}_to_{end}.png")

# Save df to excel
df_deciles.to_excel(f"{index}_ranking_df.xlsx", index=True)
