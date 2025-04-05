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
Next steps:
- get rid of lookahead bias by getting the mean for the last 180 days (or whatever), calculate the z-score for day 181 onwards, and THEN start trading (while updating the mean and the z-score each day BEFORE making the trade). I believe this rolling window is applied in the later versions of the semis_bots

- Look at cross-class asset start arb (e.g. Coffee Futures and Folger's Coffee stock)
or BTC and Gold (other ideas: SP500, ETFs, Nasdaq, Dow, Russell, Gold, Silver, Bitcoin, Ethereum, Oil, Gas, Corn, Wheat, Soybeans, Coffee, Sugar, Cocoa)
'''
# Polygon API
polygon_client = RESTClient(api_key="5hI5fdbeORm_DVWZ1PFxoWQtP5iL5EJx")

start_time = time.time()

# Input parameters
tickers = ['FBTC', 'BTC']
tickers = ['NVDL', 'USD']
# tickers = ['OKLO', 'PLTR']
# tickers = ['GOOG', 'WFC']
# tickers = ['MA', 'V'] # depends on beta, should switch to whichever is closer to beta of 1
tickers = tickers[::-1] # reverse the list

'''Consider a 14+ day SMA/RSI trailing and beta calculation function
Make sure beta has no lookahead bias'''
granularity = "1min" # "1min", "1hour", "1day"
reduce_df = False # COULD throw off calculations
z_score_threshold = 1.0 # Z-score threshold for trading
start_date = '2024-01-01'
end_date = '2025-04-04'
starting_cap = 50000
data_pull = 'polygon' # 'polygon' or 'yahoo' or 'csv'

# Change to csv directory
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

    # Ignore timezones
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Get rid of multi-index
    df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close'})

    # Calculate the returns
    df['returns'] = df['close'].pct_change().dropna()
    
    # Calculate the log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Keep timestamp, close (as ticker name), and returns
    df = df[['timestamp', 'close', 'returns', 'log_returns']]

    # Rename columns
    df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'log': f'{ticker}log_returns'})

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

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert to NY timezone from UTC
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

        # Remove timezone (so it can be saved to excel)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # Remove anything 17:00:00 and later
        df = df[df['timestamp'].dt.time < datetime.datetime.strptime('17:00:00', '%H:%M:%S').time()]

        # Remove anything from 00:00:00 to 01:00:00
        df = df[df['timestamp'].dt.time > datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()]

        if granularity == "1min":
            df = df
        elif granularity == "1hour":
            df = df[df['timestamp'].dt.minute == 0]
        elif granularity == "1day":
            # Only keep the 4pm time only
            df = df[df['timestamp'].dt.hour == 16]
            df = df[df['timestamp'].dt.minute == 0]

        # Calculate the returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Only keep some columns
        df = df[['timestamp', 'close', 'returns', 'log_returns']]

        # Rename columns
        df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'log_returns': f'{ticker}_log_returns'})

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

    # Calculate the log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Keep timestamp, close (as ticker name), and returns
    df = df[['timestamp', 'close', 'returns', 'log_returns']]

    # Rename columns
    df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'log_returns': f'{ticker}_log_returns'})

    # Reset the index
    df = df.reset_index(drop=True)

    return df

main_df = pd.DataFrame()

for ticker in tickers:
    if data_pull == 'csv':
        df = read_company_ohlc(ticker, granularity) # From saved csv files
    elif data_pull == 'polygon':
        df = get_polygon_data(ticker, 15) # Streaming from polygon (some are incorrect)
    elif data_pull == 'yahoo':
        df = get_yf_data(ticker, start_date, end_date, '15m')

    # Combine the dataframes
    if main_df.empty:
        main_df = df
    else:
        main_df = pd.merge(main_df, df, on='timestamp', how='inner')

    # Clean up the data
    main_df = main_df.dropna()

print(f"main_df: {main_df}")

# Get constant for regression
X = sm.add_constant(main_df[f'{tickers[0]}_log_returns'])
model = sm.OLS(main_df[f'{tickers[1]}_log_returns'], X).fit()
# print(model.summary())

# Calculate beta
beta = round(model.params.iloc[1], 4)
print(f"Beta of {tickers[1]}: {beta}")

# Get log returns & check for stationarity
S1 = main_df[f'{tickers[0]}_log_returns']
S1.name = f'{tickers[0]}_log_returns'
check_for_stationarity(S1)

S2 = main_df[f'{tickers[1]}_log_returns']
S2.name = f'{tickers[1]}_log_returns'
check_for_stationarity(S2)

# Get the spread
Z = S1 - beta * S2
Z.name = 'Spread'
check_for_stationarity(Z)

main_df['Spread'] = Z

# Only keep the dataframe between 2023-01-01 and 2023-06-30
main_df = main_df[(main_df['timestamp'] >= start_date) & (main_df['timestamp'] <= end_date)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Initial share counts
main_df[f'{tickers[0]}_shares'] = 0.0
main_df[f'{tickers[1]}_shares'] = 0.0
main_df[f'{tickers[0]}_BnH'] = 0.0
main_df[f'{tickers[1]}_BnH'] = 0.0

# Assign portfolio value column
main_df = main_df.assign(portfolio_value=0.0)

# Calculate mean, std, and z-score of the spread
mean = main_df['Spread'].mean()
std = main_df['Spread'].std()
variance = main_df['Spread'].var()

# Add z-score to the dataframe
main_df['Z-score'] = (main_df['Spread'] - mean) / std

# Trim dataframe to be only where flags are not 0
if reduce_df == True:
    main_df = main_df[(main_df['Z-score'] >= 1) | (main_df['Z-score'] <= -1)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Set the initial portfolio + BnH values
main_df.loc[0, 'portfolio_value'] = starting_cap
main_df.loc[0, f'{tickers[0]}_BnH'] = starting_cap
main_df.loc[0, f'{tickers[1]}_BnH'] = starting_cap

# Iterate through the data and trade based on the spread (long/short pairs trading)
for i in tqdm(range(0, len(main_df))):

    # Bring forward the shares
    if i > 0:
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i-1, f'{tickers[0]}_shares']
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i-1, f'{tickers[1]}_shares']

        # Update port value if shares are not 0
        if main_df.loc[i, f'{tickers[0]}_shares'] + main_df.loc[i, f'{tickers[1]}_shares'] > 0.0:
            main_df.loc[i, 'portfolio_value'] = main_df.loc[i, f'{tickers[0]}_shares'] * main_df.loc[i, tickers[0]] + main_df.loc[i, f'{tickers[1]}_shares'] * main_df.loc[i, tickers[1]]
        else:
            # No change to port value
            main_df.loc[i, 'portfolio_value'] = main_df.loc[i-1, 'portfolio_value']
            
        # Update buy and hold values
        main_df.loc[i, f'{tickers[0]}_BnH'] = main_df.loc[i-1, f'{tickers[0]}_BnH'] * (1 + main_df.loc[i, f'{tickers[0]}_returns'])
        main_df.loc[i, f'{tickers[1]}_BnH'] = main_df.loc[i-1, f'{tickers[1]}_BnH'] * (1 + main_df.loc[i, f'{tickers[1]}_returns'])

    if main_df.loc[i, 'Z-score'] > 1:
        # Sell (update shares)
        main_df.loc[i, f'{tickers[0]}_shares'] = 0.0

        # Buy (update shares)
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[1]]

    elif main_df.loc[i, 'Z-score'] < -1:
        # Sell (update shares)
        main_df.loc[i, f'{tickers[1]}_shares'] = 0.0

        # Buy (update shares)
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[0]]

# Show all columns
pd.set_option('display.max_columns', None)
print(main_df)

# Buy and hold + port values
ticker1_hold = main_df[tickers[0]].iloc[-1] / main_df[tickers[0]].iloc[0] * starting_cap
ticker2_hold = main_df[tickers[1]].iloc[-1] / main_df[tickers[1]].iloc[0] * starting_cap
portfolio_value = main_df['portfolio_value'].iloc[-1]

# Round em up
ticker1_hold = round(ticker1_hold, 2)
ticker2_hold = round(ticker2_hold, 2)
portfolio_value = round(portfolio_value, 2)

# Performance stats
ticker1_return = (ticker1_hold - starting_cap) / starting_cap * 100
ticker2_return = (ticker2_hold - starting_cap) / starting_cap * 100
portfolio_return = (portfolio_value - starting_cap) / starting_cap * 100

# Round em up
ticker1_return = round(ticker1_return, 2)
ticker2_return = round(ticker2_return, 2)
portfolio_return = round(portfolio_return, 2)

print(f"Buy and hold {tickers[0]}: ${ticker1_hold} ({ticker1_return}%)")
print(f"Buy and hold {tickers[1]}: ${ticker2_hold} ({ticker2_return}%)")
print(f"Pairs trading portfolio value: ${portfolio_value} ({portfolio_return}%)\n")
print(f"Beta of {tickers[1]}/{tickers[0]}: {beta}")

# Save the data to an Excel file
os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\pairs_tests')
main_df.to_excel(f'{tickers[0]}_{tickers[1]}_pairs_test.xlsx', index=False)

# Plot the data
style.use('ggplot')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# 1st plot: share prices
ax1.plot(main_df['timestamp'], main_df[tickers[0]], label=tickers[0], color='darkblue')
ax1.plot(main_df['timestamp'], main_df[tickers[1]], label=tickers[1], color='red')
ax1.set_title(f'Returns: {tickers[0]} {ticker1_return}% | {tickers[1]} {ticker2_return}%')
ax1.set_ylabel('Share Price ($)')
ax1.legend()


# 2nd plot: z-score
ax2.plot(main_df['timestamp'], main_df['Z-score'], label='Z-score', color='gray')
ax2.axhline(y=z_score_threshold, color='red', linestyle='--')
ax2.axhline(y=-z_score_threshold, color='blue', linestyle='--')
ax2.set_title(f'Z-Score (> 1.0 = buy {tickers[1]} | < -1.0 = buy {tickers[0]})')
ax2.set_ylabel('Z-Score')

# Plot where the spread is above the z-score threshold
ax2.fill_between(main_df['timestamp'], main_df['Z-score'], z_score_threshold, where=(main_df['Z-score'] >= z_score_threshold), color='red', alpha=0.5)
ax2.fill_between(main_df['timestamp'], main_df['Z-score'], -z_score_threshold, where=(main_df['Z-score'] <= -z_score_threshold), color='blue', alpha=0.5)

# 3rd plot: Port value vs BnH values
ax3.plot(main_df['timestamp'], main_df['portfolio_value'], label='Port', color='green')
ax3.plot(main_df['timestamp'], main_df[f'{tickers[0]}_BnH'], label=f'{tickers[0]} BnH', color='darkblue')
ax3.plot(main_df['timestamp'], main_df[f'{tickers[1]}_BnH'], label=f'{tickers[1]} BnH', color='red')
ax3.set_title(f'Pairs {tickers[0]} / {tickers[1]} Port Value: ${portfolio_value:,.2f} ({portfolio_return:.2f}%)')
ax3.set_ylabel('Value ($)')

# Give space between the plots
plt.subplots_adjust(hspace=0.5)

# Caluclate runtime
end_time = time.time()
print(f"Time taken: {round((end_time - start_time), 2)} seconds")

# Show the charts
plt.show()
