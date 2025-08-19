import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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
end = datetime.today().date()
end = end - timedelta(days=365 * 1)  # 1 year
start = end - timedelta(days=365 * 1)
print(f"Start: {start} | End: {end}")

# Input params
# tickers = ['NVDA', 'GOOGL', 'MSFT'] # above
# tickers = ['NVDA']
# start = '2024-01-01' # above
# end = '2024-12-31' # above
factor = 'market_cap'  # 'returns', 'eps_growth', 'revenue_growth', 'profit_growth'

# Download history
df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, group_by='ticker')

# Only select closing prices from inner level
df = df[[col for col in df.columns if col[1] == 'Close']]

# Rename columns to tickers
df.columns = [col[0] for col in df.columns]

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
        df_income_stmts = stock.get_income_stmt(freq='yearly').transpose()

        # Get index dates for price history download
        dates = df_income_stmts.index.tolist()
        start = dates[-1]
        end = dates[0] + timedelta(days=1)

        # Download prices
        df_prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        # Reduce to just closes
        df_prices = df_prices[[col for col in df_prices.columns if col[0] == 'Close']]

        # Make single index
        df_prices.columns = [col[0] for col in df_prices.columns]

        # Merge prices and income statement df
        df = pd.merge(df_income_stmts, df_prices, left_index=True, right_index=True, how='inner')

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


print(df_deciles)

print(errors)

# Save to excel:
# Save dfs to excel
try:
    os.chdir(r'C:\Users\eklundmd\OneDrive - Thermo Fisher Scientific\Desktop\Python Projects\Learning\backtest')
    with pd.ExcelWriter('df_deciles.xlsx') as writer:  
        df_deciles.to_excel(writer, sheet_name='deciles')
except:
    with pd.ExcelWriter('df_deciles.xlsx') as writer:  
        df_deciles.to_excel(writer, sheet_name='deciles')

# Plot
if 1 == 0:
    plt.figure(figsize=(10, 6))
    # plt.plot(df['DilutedEPS'], label='EPS')
    # plt.plot(df['TotalRevenue'], label="Revenue")
    # plt.plot(df['NetIncome'], label='Income')
    plt.plot('MarketCap')
    plt.legend()
    plt.grid()
    # plt.show()
