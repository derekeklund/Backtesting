import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
import time


''' 
Next look at the spread and if it's negative, buy the stock and sell the comparison stock (M1/market). If it's positive, do the opposite. This is a pairs trading strategy.

Might have to switch it up and backtest both.

1. Get data from 2020-2024 in one df
2. Start with 2020-01-01 to 2021-01-01
3. Get the beta and spread of the stock and market for the past year
4. Trade based on the spread daily
5. Increment one day at a time and calculate the new beta and spread
6. Repeat until end (2025-12-31)
'''

def check_for_stationarity(X, cutoff=0.01):
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely non-stationary.')
        return False
    
start_time = time.time()

tickers = ['MA', 'V']
granularity = "1hour" # "1min", "1hour", "1day"

os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\companyCSVs\Pairs')

def read_company_ohlc(ticker, granularity):
    # Read the data from the CSV file
    file = f'{ticker}_ohlc.csv'

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

    df['additive_returns'] = df['close'].diff()[1:]

    # Keep timestamp, close (as ticker name), and returns
    df = df[['timestamp', 'close', 'returns', 'additive_returns']]

    # Rename columns
    df = df.rename(columns={'close': ticker, 'returns': f'{ticker}_returns', 'additive_returns': f'{ticker}_additive_returns'})

    # Reset the index
    df = df.reset_index(drop=True)

    return df

# Create the plot
style.use('ggplot')
fig, ax = plt.subplots(figsize=(10,6))

main_df = pd.DataFrame()

for ticker in tickers:
    df = read_company_ohlc(ticker, granularity)

    # print(f"df ({ticker}): {df}")

    if main_df.empty:
        main_df = df
    else:
        main_df = pd.merge(main_df, df, on='timestamp', how='inner')

    # Clean up the data
    main_df = main_df.dropna()

    # # Plot the data
    # ax.plot(df['timestamp'], df[ticker], label=ticker)

# print(f"main_df: {main_df}")

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

plt.plot(Z)
plt.xlabel('Time')
plt.ylabel('Spread')
plt.legend([Z.name])
# plt.show()

main_df['Spread'] = Z

# Only keep the dataframe between 2023-01-01 and 2023-06-30
start_date = '2024-01-01'
end_date = '2024-06-30'
main_df = main_df[(main_df['timestamp'] >= start_date) & (main_df['timestamp'] <= end_date)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Initial share counts
main_df[f'{tickers[0]}_shares'] = 0.0
main_df[f'{tickers[1]}_shares'] = 0.0

# Assign portfolio value column
main_df = main_df.assign(portfolio_value=0.0)

# Trim dataframe to be only where flags are not 0
main_df = main_df[(main_df['Spread'] >= 5) | (main_df['Spread'] <= -5)]
# main_df = main_df[(main_df[f'{tickers[0]}_flag'] != 0) | (main_df[f'{tickers[1]}_flag'] != 0)]

# Reset the index
main_df = main_df.reset_index(drop=True)

# Set the initial portfolio value
main_df.loc[0, 'portfolio_value'] = 100000.0

# Iterate through the data and trade based on the spread (long/short pairs trading)
for i in tqdm(range(0, len(main_df))):

    print(f"i: {i}, Spread: {main_df.loc[i, 'Spread']}")

    # Bring forward the shares
    if i > 0:
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i-1, f'{tickers[0]}_shares']
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i-1, f'{tickers[1]}_shares']

        # Update portfolio value
        main_df.loc[i, 'portfolio_value'] = main_df.loc[i, f'{tickers[0]}_shares'] * main_df.loc[i, tickers[0]] + main_df.loc[i, f'{tickers[1]}_shares'] * main_df.loc[i, tickers[1]]

    if main_df.loc[i, 'Spread'] > 5:
        # Sell
        main_df.loc[i, f'{tickers[0]}_shares'] = 0

        # Buy
        main_df.loc[i, f'{tickers[1]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[1]]
    elif main_df.loc[i, 'Spread'] < -5:
        # Sell 
        main_df.loc[i, f'{tickers[1]}_shares'] = 0

        # Buy
        main_df.loc[i, f'{tickers[0]}_shares'] = main_df.loc[i, 'portfolio_value'] / main_df.loc[i, tickers[0]]


print(main_df)


# Save the data to an Excel file
os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\pairs_tests')
main_df.to_excel(f'{tickers[0]}_{tickers[1]}_pairs_test.xlsx', index=False)


if 1 == 0:
    # Save the data to an Excel file
    os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\pairs_tests')
    main_df.to_excel(f'{tickers[0]}_{tickers[1]}_pairs_test.xlsx', index=False)

    for ticker in tickers:
        # Plot the data
        ax.plot(df['timestamp'], df[ticker], label=ticker)

    # Format x-axis as YYYY-MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.title(f'{tickers} {granularity} OHLC')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=25)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) # Max number of xticks

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    plt.show()
