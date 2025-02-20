import alpaca_trade_api as alpacaapi
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TrailingStopOrderRequest
from alpaca.data.live import StockDataStream
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from polygon import RESTClient
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
import pandas as pd
import numpy as np
import datetime
import logging
import os
import time
import pytz

'''
Semi Pairs Trading Bot
1. Get 6 months of prices for two tickers (USD and NVDL)
2. Get integration level of 0 for both. Test for stationarity
3. Get beta and spread
4. Trade on spread
'''

def date_and_time():
    # Define the Eastern Time zone
    eastern = pytz.timezone('US/Eastern')
    
    # Get the current date
    current_date = datetime.datetime.now(eastern).date().strftime('%m-%d-%Y')
    current_time = datetime.datetime.now(eastern).time().strftime('%H:%M:%S')
    
    return current_date, current_time

def timetz(*args):
    return datetime.datetime.now(eastern).timetuple()

def log_message(message):
    logging.info(message)
    print(message)

try:
    os.chdir(r'C:\Users\derek\Coding Projects\logs')
except:
    pass

current_date, current_time = date_and_time()

# Replace colons with dashes for file naming
file_time = current_time.replace(':', '-')

# Define the Eastern Time zone
eastern = pytz.timezone('US/Eastern')

# Adjust logging to be eastern time
logging.Formatter.converter = timetz

# Configure the logger
logging.basicConfig(filename=f'semi_pairs_bot_{current_date}_{file_time}.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

log_message(f"Good morning!")

# Alpaca API
api_data_url = 'https://paper-api.alpaca.markets/v2'
api_key = 'PK7B9FFOJ8TKTQUQKDPI'
secret_key = '84YWIwZnQyBghACuLjdgXLhqEgrHGWbLfaLzt5Tk'

# Create trading client with API key and secret
alpaca_client = TradingClient(api_key, secret_key, paper=True)

# Polygon API
polygon_client = RESTClient(api_key="5hI5fdbeORm_DVWZ1PFxoWQtP5iL5EJx")

# Get account details
account = dict(alpaca_client.get_account())
log_message("*****************************************************")
for k, v in account.items():
    log_message(f"{k:30} {v}")

    if k == 'cash':
        initial_cash = v
log_message("*****************************************************")

# Input parameters
tickers = ['NVDL', 'USD']
tickers = ['USD', 'NVDL']
granularity = '1hour' # 1min, 1hour, 1day
spread_width = 0.5
days = 180

# Get yesterday's date in format YYYY-MM-DD
start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

log_message(start_date)
log_message(end_date)

pd.set_option('display.max_columns', None)

def get_company_data(ticker):
    log_message(f"Getting data for {ticker}")

    aggs = []
    for a in polygon_client.list_aggs(ticker=ticker, multiplier=1, timespan="hour", from_=start_date, to=end_date, adjusted=True, sort="asc", limit=50000):
        aggs.append(a)

    try:
        # Turn aggs into a dataframe
        df = pd.DataFrame(aggs)
        # log_message(df)

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

        # log_message(df)
        # log_message(len(df))

        return df
    
    except Exception as e:
        log_message(e)


df_pairs = pd.DataFrame()

# Get data for tickers into one dataframe
for ticker in tickers:

    df = get_company_data(ticker)

    if df_pairs.empty:
        df_pairs = df
    else:
        df_pairs = pd.merge(df_pairs, df, on='timestamp', how='inner')

    # Clean up the data
    df_pairs = df_pairs.dropna()


def check_for_stationarity(X, cutoff=0.01):
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        log_message('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely stationary.')
        return True
    else:
        log_message('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely non-stationary.')
        return False

# Get constant for regression
X = sm.add_constant(df_pairs[f'{tickers[0]}_returns'])
# log_message(X)
model = sm.OLS(df_pairs[f'{tickers[1]}_returns'], X).fit()

# Calculate beta
beta1 = round(model.params.iloc[1], 4)
log_message(f"Beta1 of {tickers[1]}: {beta1}")

X = sm.add_constant(df_pairs[f'{tickers[1]}_returns'])
# log_message(X)
model = sm.OLS(df_pairs[f'{tickers[0]}_returns'], X).fit()

# Calculate beta
beta2 = round(model.params.iloc[1], 4)
log_message(f"Beta2 of {tickers[0]}: {beta2}")

if 1 - abs(beta1) < 1 - abs(beta2):
    beta = beta1
else:
    beta = beta2

log_message(f"True Beta: {beta}")

# Get additive returns & check for stationarity
P1 = df_pairs[f'{tickers[0]}_additive_returns']
P1.name = tickers[0]
check_for_stationarity(P1)

P2 = df_pairs[f'{tickers[1]}_additive_returns']
P2.name = tickers[1]
check_for_stationarity(P2)

# Get the spread
Z = P1 - beta * P2
Z.name = 'Spread'
check_for_stationarity(Z)

plt.plot(Z)
plt.xlabel('Time')
plt.ylabel('Spread')
plt.legend([Z.name])
# plt.show()

df_pairs['Spread'] = Z

df_pairs = df_pairs[(df_pairs['Spread'] >= spread_width) | (df_pairs['Spread'] <= -spread_width)]

# Reset the index
df_pairs = df_pairs.reset_index(drop=True)

'''
Trading time
'''

# Connect to Alpaca API
api = alpacaapi.REST(api_key, secret_key, api_data_url, api_version='v2')

today = datetime.datetime.now(eastern).date()
day_of_week = today.weekday()

log_message(f"Today: {today} | Day of week: {day_of_week}")

if day_of_week == 5 or day_of_week == 6:
    log_message("Weekend. Exiting program.")
    exit()

# Wait until 9:35am to submit the order
# while datetime.datetime.now(eastern).time() < datetime.time(9, 30):

#     log_message(f"Waiting for market open.")
#     time.sleep(60)

log_message(f"\n{df_pairs.head(5)}")

# Get bars from 9:30am to 9:35am. Buy at 9:35am
while datetime.datetime.now(eastern).time() < datetime.time(16, 00):

    # Get datetime of last bar
    last_bar_datetime = datetime.datetime.now(eastern).strftime('%Y-%m-%d %H:%M:%S')

    last_index = len(df_pairs)
    df_pairs.loc[last_index, 'timestamp'] = last_bar_datetime

    # Get current bar at market open
    for ticker in tickers:
        last_bar = api.get_latest_bar(ticker)
        last_bar = float(last_bar.o) # Open price
        log_message(f"Market opening bar ({ticker}): {last_bar}")

        # Append to df_pairs
        df_pairs.loc[last_index, ticker] = last_bar

        # Calculate returns
        df_pairs.loc[last_index, f'{ticker}_returns'] = df_pairs.loc[last_index, ticker] / df_pairs.loc[last_index - 1, ticker] - 1

        # Calculate additive returns
        df_pairs.loc[last_index, f'{ticker}_additive_returns'] = df_pairs.loc[last_index, ticker] - df_pairs.loc[last_index - 1, ticker]

        

    # Update beta and spread

    # Get additive returns & check for stationarity
    P1 = df_pairs[f'{tickers[0]}_additive_returns']
    P1.name = tickers[0]
    check_for_stationarity(P1)

    P2 = df_pairs[f'{tickers[1]}_additive_returns']
    P2.name = tickers[1]
    check_for_stationarity(P2)

    # Get the spread
    Z = P1 - beta * P2
    Z.name = 'Spread'
    check_for_stationarity(Z)

    df_pairs['Spread'] = Z

    # Get updated buying power + number of shares to buy
    buying_power = float(account['portfolio_value']) # All in
    
    log_message(f"Buying power: {buying_power}")
    log_message(f"\n{df_pairs.tail(5)}")

    # Get positions
    positions = alpaca_client.get_all_positions()
    for pos in positions:
        log_message(f">>>>> Symbol: {pos.symbol} | Qty: {pos.qty} | Market value: {pos.market_value} | Cost basis: {pos.cost_basis}")
    

    pos_symbol = None

    for pos in positions:
        pos_symbol = pos.symbol
        pos_qty = pos.qty

    if datetime.datetime.now(eastern).time() > datetime.time(9, 35):
        if df_pairs.loc[last_index, 'Spread'] > spread_width and pos_symbol != tickers[1]:

            if pos_symbol != None:

                log_message(f"Selling {tickers[0]}")

                last_bar = api.get_latest_bar(tickers[0])
                last_bar = float(last_bar.o) # Open price
                log_message(f">>>> Current bar ({tickers[0]}): {last_bar}")
                log_message(f">>>> Portfolio BP: {buying_power}")
                log_message(f">>>> Position Quantity: {pos_qty}")

                # Create market details
                sell_order = MarketOrderRequest(
                    symbol=tickers[0],
                    qty=pos_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY # FOK = fill or kill
                )

                # Submit initial market open buy order
                try:
                    order = alpaca_client.submit_order(order_data=sell_order)
                except Exception as e:
                    log_message("Issue with order--", e)

                log_message(f"Sell order data: {sell_order}\n")
                time.sleep(5)

            log_message(f"Buying {tickers[1]}")

            # Calculate quantity to buy
            last_bar = api.get_latest_bar(tickers[1])
            last_bar = float(last_bar.o) # Open price
            log_message(f">>>> Current bar ({tickers[1]}): {last_bar}")
            quantity = (buying_power / last_bar) * 0.99 # 1% less than all in to account for fees/fill price
            quantity = np.floor(quantity)
            log_message(f">>>> Portfolio BP: {buying_power}")
            log_message(f">>>> Position Quantity: {quantity}")

            # Create market details
            buy_order = MarketOrderRequest(
                symbol=tickers[1],
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY # FOK = fill or kill
            )

            # Submit initial market open buy order
            try:
                order = alpaca_client.submit_order(order_data=buy_order)
            except Exception as e:
                log_message("Issue with order--", e)
            
            log_message(f"Buy order data: {buy_order}\n")
            time.sleep(5)

        elif df_pairs.loc[last_index, 'Spread'] < -spread_width and pos_symbol != tickers[0]:

            if pos_symbol != None:

                log_message(f"Selling {tickers[1]}")

                last_bar = api.get_latest_bar(tickers[1])
                last_bar = float(last_bar.o) # Open price
                log_message(f">>>> Current bar ({tickers[1]}): {last_bar}")
                log_message(f">>>> Portfolio BP: {buying_power}")
                log_message(f">>>> Position Quantity: {pos_qty}")

                # Create market details
                sell_order = MarketOrderRequest(
                    symbol=tickers[1],
                    qty=pos_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY # FOK = fill or kill
                )

                # Submit initial market open buy order
                try:
                    order = alpaca_client.submit_order(order_data=sell_order)
                except Exception as e:
                    log_message("Issue with order--", e)

                log_message(f"Sell order data: {sell_order}\n")
                time.sleep(5)

            log_message(f"Buying {tickers[0]}")

            # Calculate quantity to buy
            last_bar = api.get_latest_bar(tickers[0])
            last_bar = float(last_bar.o) # Open price
            log_message(f">>>> Current bar ({tickers[0]}): {last_bar}")
            quantity = (buying_power / last_bar) * 0.99 # 1% less than all in to account for fees/fill price
            quantity = np.floor(quantity)
            log_message(f">>>> Portfolio BP: {buying_power}")
            log_message(f">>>> Position Quantity: {quantity}")

            # Create market details
            buy_order = MarketOrderRequest(
                symbol=tickers[0],
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY # FOK = fill or kill
            )

            # Submit initial market open buy order
            try:
                order = alpaca_client.submit_order(order_data=buy_order)
            except Exception as e:
                log_message("Issue with order--", e)

            log_message(f"Buy order data: {buy_order}\n")
            time.sleep(5)

            
    else:
        log_message("Waiting for market open.")

    # Save the data to an Excel file
    os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\pairs_tests')
    df_pairs.to_excel(f'{tickers[0]}_{tickers[1]}_pairs_test.xlsx', index=False)

    log_message(f"Sleeping for 5 minutes.")
    time.sleep(300)

# Loop/async function to check for spread and trade every hour
# Add new row to df_pairs every hour to update beta and spread
