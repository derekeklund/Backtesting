import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from polygon import RESTClient
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
import time
import yfinance as yf

''' 

1. Get data from 2020-2024 in one df. DONE
2. Start with 2020-01-01 to 2021-01-01. DONE
3. Get the beta and spread of the stock and market for the past year
4. Trade based on the spread daily
5. Increment one day at a time and calculate the new beta and spread
6. Repeat until end (2025-12-31)
'''
# Polygon API
polygon_client = RESTClient(api_key="5hI5fdbeORm_DVWZ1PFxoWQtP5iL5EJx")

start_time = time.time()

# Input parameters
tickers = ['NVDL', 'USD']
# tickers = ['MA', 'V'] # depends on beta, should switch to whichever is closer to beta of 1
tickers = tickers[::-1] # reverse the list

'''Consider a 14+ day SMA/RSI trailing and beta calculation function
Make sure beta has no lookahead bias'''
granularity = "1min" # "1min", "1hour", "1day"

spread_width = 0.25
start_date = '2025-01-01'
end_date = '2025-03-15'
starting_cap = 50000
data_pull = 'polygon' # 'polygon' or 'yahoo' or 'csv'

''' Look at cross-class asset start arb (e.g. Coffee Futures and Folger's Coffee stock)
or BTC and Gold 

Best so far NVDL/USD

'''
# SP500, ETFs, Nasdaq, Dow, Russell, Gold, Silver, Bitcoin, Ethereum, Oil, Gas, Corn, Wheat, Soybeans, Coffee, Sugar, Cocoa
root_dir = r'C:\Users\derek\Coding Projects\Backtesting\companyCSVs'
os.chdir(root_dir)

def check_for_stationarity(X, cutoff=0.01):
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely non-stationary.')
        return False
    
def get_yf_data(ticker, start_date, end_date, granularity):

    print(f">>>>> Getting data for {ticker} from Yahoo Finance...")

    # Get the data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date, interval=granularity)

    # Reset the index
    df = df.reset_index()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['Datetime'], unit='ms')

    # Get rid of multi-index
    df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close'})

    # Calculate the returns
    df['returns'] = df['close'].pct_change().dropna()
    df['additive_returns'] = df['close'].diff()[1:]

    # Keep timestamp, close (as ticker name), and returns
    df = df[['timestamp', 'close', 'returns', 'additive_returns']]

    # Rename columns
    df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'additive_returns': f'{ticker}_additive_returns'})

    # Reset the index
    df = df.reset_index(drop=True)

    return df
    
def get_polygon_data(ticker, multiplier):
    print(f">>>>> Getting data for {ticker} from polygon...")

    aggs = []
    for a in polygon_client.list_aggs(ticker=ticker, multiplier=multiplier, timespan="minute", from_=start_date, to=end_date, adjusted=True, sort="asc", limit=50000):
        aggs.append(a)

    try:
        # Turn aggs into a dataframe
        df = pd.DataFrame(aggs)
        # print(df)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Remove anything 17:00:00 and later
        df = df[df['timestamp'].dt.time < datetime.datetime.strptime('20:00:00', '%H:%M:%S').time()]

        # Remove anything from 00:00:00 to 01:00:00
        df = df[df['timestamp'].dt.time > datetime.datetime.strptime('01:00:00', '%H:%M:%S').time()]

        if granularity == "1min":
            df = df
        elif granularity == "1hour":
            df = df[df['timestamp'].dt.minute == 0]
        elif granularity == "1day":
            # Only keep the 4pm time only
            df = df[df['timestamp'].dt.hour == 16]
            df = df[df['timestamp'].dt.minute == 0]

        # Calculate the returns
        df['returns'] = df['close'].pct_change().dropna()
        df['additive_returns'] = df['close'].diff()[1:]

        df = df[['timestamp', 'close', 'returns', 'additive_returns']]

        # Rename columns
        df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'additive_returns': f'{ticker}_additive_returns'})

        # Reset the index
        df = df.reset_index(drop=True)

        return df
    
    except Exception as e:
        print(e)
    
def read_company_ohlc(ticker, granularity):
    # Read the data from the CSV file
    file = f'{ticker}_ohlc.csv'

    # Scan folders for the file
    for root, dirs, files in os.walk(root_dir):
        if file in files:
            file = os.path.join(root, file)
            break

    # Read the data from the CSV file
    df = pd.read_csv(file)

    print("1", df)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if granularity == "1min":
        df = df
    elif granularity == "1hour":
        df = df[df['timestamp'].dt.minute == 0]
    elif granularity == "1day":
        # Only keep the 4pm time only
        df = df[df['timestamp'].dt.hour == 16]
        df = df[df['timestamp'].dt.minute == 0]

    # Calculate the returns
    df['returns'] = df['close'].pct_change().dropna()

    df['additive_returns'] = df['close'].diff()[1:]

    # Keep timestamp, close (as ticker name), and returns
    df = df[['timestamp', 'close', 'returns', 'additive_returns']]

    # Rename columns
    df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'additive_returns': f'{ticker}_additive_returns'})

    # Reset the index
    df = df.reset_index(drop=True)

    print("2", df)

    return df

# Create the plot
style.use('ggplot')
# fig, ax = plt.subplots(figsize=(10,6))

main_df = pd.DataFrame()

for ticker in tickers:
    if data_pull == 'csv':
        df = read_company_ohlc(ticker, granularity) # From saved csv files
    elif data_pull == 'polygon':
        df = get_polygon_data(ticker, 1) # Streaming from polygon (some are incorrect)
    elif data_pull == 'yahoo':
        df = get_yf_data(ticker, start_date, end_date, '15m')

    # print(f"df ({ticker}): {df}")

    if main_df.empty:
        main_df = df
    else:
        main_df = pd.merge(main_df, df, on='timestamp', how='inner')

    # Clean up the data
    main_df = main_df.dropna()

    # # Plot the data
    # ax.plot(df['timestamp'], df[ticker], label=ticker)

print(f"main_df: {main_df}")

# Get constant for regression
X = sm.add_constant(main_df[f'{tickers[0]}_returns'])
# print(X)
model = sm.OLS(main_df[f'{tickers[1]}_returns'], X).fit()
# print(model.summary())

# Calculate beta
beta = round(model.params.iloc[1], 4)
print(f"Beta of {tickers[1]}: {beta}")

# Get additive returns & check for stationarity
S1 = main_df[f'{tickers[0]}_additive_returns']
S1.name = tickers[0]
check_for_stationarity(S1)

S2 = main_df[f'{tickers[1]}_additive_returns']
S2.name = tickers[1]
check_for_stationarity(S2)

# Get the spread
Z = S1 - beta * S2
Z.name = 'Spread'
print(Z)
check_for_stationarity(Z)

# plt.plot(Z)
# plt.xlabel('Time')
# plt.ylabel('Spread')
# plt.legend([Z.name])
# plt.show()

main_df['Spread'] = Z

# Only keep the dataframe between 2023-01-01 and 2023-06-30
main_df = main_df[(main_df['timestamp'] >= start_date) & (main_df['timestamp'] <= end_date)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Initial share counts
main_df[f'{tickers[0]}_shares'] = 0.0
main_df[f'{tickers[1]}_shares'] = 0.0

# Assign portfolio value column
main_df = main_df.assign(portfolio_value=0.0)

# Trim dataframe to be only where flags are not 0
main_df = main_df[(main_df['Spread'] >= spread_width) | (main_df['Spread'] <= -spread_width)]
# main_df = main_df[(main_df[f'{tickers[0]}_flag'] != 0) | (main_df[f'{tickers[1]}_flag'] != 0)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Set the initial portfolio value
main_df.loc[0, 'portfolio_value'] = starting_cap

# Iterate through the data and trade based on the spread (long/short pairs trading)
for i in tqdm(range(0, len(main_df))):

    # Bring forward the shares
    if i > 0:
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i-1, f'{tickers[0]}_shares']
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i-1, f'{tickers[1]}_shares']

        # Update portfolio value
        main_df.loc[i, 'portfolio_value'] = main_df.loc[i, f'{tickers[0]}_shares'] * main_df.loc[i, tickers[0]] + main_df.loc[i, f'{tickers[1]}_shares'] * main_df.loc[i, tickers[1]]

    if main_df.loc[i, 'Spread'] > spread_width:
        # Sell
        main_df.loc[i, f'{tickers[0]}_shares'] = 0

        # Buy
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[1]]
    elif main_df.loc[i, 'Spread'] < -spread_width:
        # Sell 
        main_df.loc[i, f'{tickers[1]}_shares'] = 0

        # Buy
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[0]]


print(main_df)

# Buy and hold values
ticker1_hold = main_df[tickers[0]].iloc[-1] / main_df[tickers[0]].iloc[0] * starting_cap
ticker2_hold = main_df[tickers[1]].iloc[-1] / main_df[tickers[1]].iloc[0] * starting_cap

portfolio_value = main_df['portfolio_value'].iloc[-1]

# Round em up
ticker1_hold = round(ticker1_hold, 2)
ticker2_hold = round(ticker2_hold, 2)
portfolio_value = round(portfolio_value, 2)

print(f"Buy and hold {tickers[0]}: ${ticker1_hold}")
print(f"Buy and hold {tickers[1]}: ${ticker2_hold}")
print(f"Pairs trading portfolio value: ${portfolio_value}\n")
print(f"Beta of {tickers[1]}/{tickers[0]}: {beta}")


if 1 == 1:
    # Save the data to an Excel file
    os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\pairs_tests')
    main_df.to_excel(f'{tickers[0]}_{tickers[1]}_pairs_test_new.xlsx', index=False)

if 1 == 1:
    style.use('ggplot')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    # Plot the two tickers
    ax1.plot(main_df['timestamp'], main_df[tickers[0]], label=tickers[0])
    ax1.plot(main_df['timestamp'], main_df[tickers[1]], label=tickers[1])
    ax1.set_title(f'{tickers[0]} and {tickers[1]} {granularity} OHLC')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()

    ax3.plot(main_df['timestamp'], main_df['portfolio_value'], label='Port', color='gray')

    # Give space between the two plots
    plt.subplots_adjust(hspace=0.5)

    # Plot the spread
    ax2.plot(main_df['timestamp'], main_df['Spread'], label='Spread', color='gray')
    ax2.axhline(y=spread_width, color='red', linestyle='--')
    ax2.axhline(y=-spread_width, color='blue', linestyle='--')
    ax2.set_title(f'Spread (> {spread_width} or < {-spread_width})')
    ax2.set_xlabel('Date')

    # Plot where the spread is above the threshold
    ax2.fill_between(main_df['timestamp'], main_df['Spread'], spread_width, where=(main_df['Spread'] >= spread_width), color='red', alpha=0.5)
    ax2.fill_between(main_df['timestamp'], main_df['Spread'], -spread_width, where=(main_df['Spread'] <= -spread_width), color='blue', alpha=0.5)

    # Format x-axis as YYYY-MM
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.title(f'{tickers} {granularity} OHLC')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=25)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) # Max number of xticks

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    plt.show()
