import yfinance as yf
from pykalman import KalmanFilter
import pandas as pd
import numpy as np
import datetime
from datetime import time as dt_time
import matplotlib.pyplot as plt
# from polygon import RESTClient
import time
import pytz
import logging
import os

import alpaca_trade_api as alpacaapi
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import MarketOrderRequest

'''
Combo Indicators Bot (3 indicators that can be turned on/off)

Exit Strat = Stop & Reverse Exit (Always in)
- Sell signal --> exit long and go short
- Buy signal --> exit short and go long
* Essentially, "Buy" means two things:
1. exit short
2. enter long

1. Bollinger Stretch Bot
- Uses the 20-day SMA
- Upper and lower bands set by 2-SD from the SMA

2. Moving Average Stretch Bot
- Uses the 20-day EMA
- Buy if price is 1-SD below the EMA

3. Connors RSI Alpaca Bot
- Uses Connors RSI (RSI + Streak + Percent Rank)
- Buy if CRSI < 10 (oversold) & sell if CRSI > 90 (overbought)

General steps:
1. Get at least 200 bars (1m, 1hr, 1day, etc.) of closing prices from yf
2. Add new bars to the dataframe as they come in from Alpaca
3. Calculate Connor's RSI and buy if under 10 (oversold)
and sell if over 90 (overbought)

Things to adjust:
1. bar size (interval) - 1m, 5m, 15m, 30m, 1h, 1d, etc. [default = 1h]
2. # of bars - 100, 150, 200, etc. [currently 200 bars]
3. Bollinger band period [default = 20]
4. Bollinger band standard deviation [default = 2]
5. Moving Average Stretch period [default = 20]
6. Moving Average Stretch standard deviation [default = 1]
7. Connors RSI period [default = 3]
8. Connors RSI streak period [default = 2]
9. Connors RSI percent rank period [default = 100]
'''

def log_message(message):
    # general_logger.info(message)
    print(message)

def log_positions():
    # positions = trading_client.get_all_positions()

    # if not positions:
    #     log_message("No positions held")
    #     return

    # for i, position in enumerate(positions):
    #     # Format the position details
    #     position_value = round(float(position.market_value), 2)
    #     log_message(f"Position #{i+1}: {position.symbol} | Qty: {position.qty} | Avg Price: {position.avg_entry_price} | Market Value: ${position_value:,.2f} | Unrealized P/L: {position.unrealized_pl}\n")

    # return positions
    return

def log_trade(order):

    trade_logger.info(f"*"*50)
    timestamp = datetime.datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %p')
    
    # Record trade details
    trade_logger.info(f"{order.symbol} | {order.side} | qty: {order.qty} | notional: {order.notional} ")
    trade_logger.info(f"{timestamp} | id: {order.id}")

    log_message(f"*"*50)

def log_error(message):
    general_logger.error(f"ERROR: {message}\n{'~'*50}", stack_info=False, exc_info=True)

def date_and_time():
    # Define the Eastern Time zone
    eastern = pytz.timezone('America/New_York')

    # Get the current date
    current_date = datetime.datetime.now(eastern).date().strftime('%m-%d-%Y')
    current_time = datetime.datetime.now(eastern).time().strftime('%H:%M:%S')

    return current_date, current_time

def timetz(*args):
    eastern = pytz.timezone('America/New_York')
    return datetime.datetime.now(eastern).timetuple()

def connors_rsi(df, rsi_period, streak_period, percent_rank_period):
    
    # 1. Calculate RSI w/ pandas series
    def calculate_rsi(series, rsi_period):
        delta = series.diff() # Diff b/w prices

        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()

        rs = gain / loss # Relative Strength
        rsi = 100 - (100 / (1 + rs))

        return rsi # Returns pandas series

    rsi = calculate_rsi(df['Close'], rsi_period)
    df['rsi'] = rsi

    # 2. Calculate Up/Down Lengths
    df['streak'] = 0

    for i in range(1, len(df)):

        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            # Add 1 if > 0
            df.at[df.index[i], 'streak'] = df['streak'].iloc[i-1] + 1 if df['streak'].iloc[i-1] > 0 else 1
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            # Minus 1 if < 0
            df.at[df.index[i], 'streak'] = df['streak'].iloc[i-1] - 1 if df['streak'].iloc[i-1] < 0 else -1
        else:
            df.at[df.index[i], 'streak'] = 0

    rsi_streak = calculate_rsi(df['streak'], streak_period)
    df['rsi_streak'] = rsi_streak

    # 3. Percent rank of 1-period return
    df['1_day_return'] = df['Close'].pct_change()
    if ticker == 'NVDA':
        df['nvdl_1_day_return'] = df['NVDL'].pct_change() # NVDL 1-day return
    df['percent_rank'] = df['1_day_return'].rolling(window=percent_rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

    # Final Connors RSI
    df['crsi'] = (df['rsi'] + df['rsi_streak'] + df['percent_rank']) / 3

    # Tracing calculations
    pd.set_option('display.max_columns', None) # Show all columns
    # log_message(f"Intermediate DataFrame:\n{df}")
    # log_message(f"CRSI Calculation = [ RSI({rsi_period}) + Streak({streak_period}) + Percent Rank({percent_rank_period}) ] / 3")

    # return df[['Close', 'crsi']]
    return df

def ATR(df, window):

    # Avoid modifying the original DataFrame
    df = df.copy() 

    # Add previous close column
    df['Prev_Close'] = df['Close'].shift(1)

    # Calculate True Range for each row
    df['True_Range'] = df.apply(
        lambda row: max(abs(row['High'] - row['Low']), abs(row['High'] - row['Prev_Close']), abs(row['Low'] - row['Prev_Close'])), axis=1)
    
    # # ATR = rolling mean of True Range
    df['ATR'] = df['True_Range'].rolling(window=window).mean()

    return df

def relative_volume(df, window):

    # Avoid modifying the original DataFrame
    df = df.copy()

    # Calculate the rolling average volume
    df['Volume_Average'] = df['Volume'].rolling(window=window).mean()

    # Calculate relative volume
    df['Relative_Volume'] = df['Volume'] / df['Volume_Average']

    return df

def kalman(df):
    """
    Apply Kalman filter to the closing prices in the DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame containing 'Close' price series.
    
    Returns:
    DataFrame: DataFrame with Kalman filter estimates added.
    """
    # Initialize Kalman Filter
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    
    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(df['Close'].values)
    df['Kalman_Mean'] = state_means

    # Compute 1-sd deviation from the Kalman mean
    df['Kalman_Std'] = df['Kalman_Mean'].rolling(window=30).std()
    df['Kalman_Upper'] = df['Kalman_Mean'] + df['Kalman_Std']
    df['Kalman_Lower'] = df['Kalman_Mean'] - df['Kalman_Std']
    
    return df
   
# Define the Eastern Time zone
eastern = pytz.timezone('America/New_York')

# Adjust logging to be eastern time
logging.Formatter.converter = timetz

try:
    os.chdir(r'C:\Users\derek\Coding Projects\logs')
except:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)

current_dir = os.getcwd()

# Create the log directory if it doesn't exist
if not os.path.exists(current_dir):
    os.makedirs(current_dir)

# Get date + time for file naming
file_date, current_time = date_and_time()
file_time = current_time.replace(':', '-')

# Get name of script and join with current dir
script_name = os.path.splitext(os.path.basename(__file__))[0] + f"_{file_date}_{file_time}.log"
log_file = os.path.join(current_dir, script_name)
trade_file = os.path.join(current_dir, 'combos_trades.log')

# General log
general_logger = logging.getLogger("general_logger")
general_logger.setLevel(logging.DEBUG)
general_logger_handler = logging.FileHandler(log_file)
general_format = logging.Formatter('%(asctime)s - %(message)s')
general_logger_handler.setFormatter(general_format)
general_logger.addHandler(general_logger_handler)

# Set up trades log
trade_logger = logging.getLogger("trade_logger")
trade_logger.setLevel(logging.DEBUG)
trade_logger_handler = logging.FileHandler(trade_file)
trade_format = logging.Formatter('%(message)s')
trade_logger_handler.setFormatter(trade_format)
trade_logger.addHandler(trade_logger_handler)

# Configure logging with the specified directory
logging.basicConfig(
    # filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt= "%m/%d/%Y %I:%M:%S %p"
)

log_message(f"~~~~~ Combo Indicators Bot ~~~~~")
log_message(f"Log file path: {log_file}")
log_message(f"Current Directory: {current_dir}")

# # API credentials
# api_key = 'PKTM13DDQTVDYWGBP2TF'
# secret_key = 'DqVsNposXekCdO7yZhteZbZeQ7ppMYjDShklO4iv'
# api_data_url = 'https://paper-api.alpaca.markets/'

# # Create trading client with API key and secret
# trading_client = TradingClient(api_key, secret_key, paper=True)

# # Get account details
# account = dict(trading_client.get_account())
# log_message("*"*50)
# for k, v in account.items():
#     log_message(f"{k:30} {v}")
# log_message("*"*50)

# Input params
ticker = "NVDA"
interval = "1d" # Bars! 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk
days_ago = 365 # How many days back to get data from
lookback_length = 150 # Number of days (25 * 7 = 175 bars for 1h interval)
# Get lookback + midway
lookback = -1 * lookback_length # -25 days for 1h bars
midway = int(lookback / 2) # 12.5 days for 1h bars
window = 20 # days for SMA + EMA
start = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d') # Tomorrow's date

# Start & End for backtesting
# start = '2015-01-01' # 1
# start = '2015-07-01' # 2
# start = '2016-01-01' # 3
# start = '2016-07-01' # 4
# start = '2017-01-01' # 5
# start = '2017-07-01' # 6
# start = '2018-01-01' # 7
# start = '2018-07-01' # 8
# start = '2019-01-01' # 9
# start = '2019-07-01' # 10
# start = '2020-01-01' # 11

start = '2023-01-01' # 12
# start = '2023-05-01' # 13
# start = '2023-09-01' # 14
# start = '2024-01-01' # 15
# start = '2024-05-01' # 16
# start = '2024-09-01' # 17
# start = '2025-01-01' # 18

df_results = pd.DataFrame(columns=['Start', 'End', 'Low Belief', 'High Belief', 'Bollinger', 'MA Stretch', 'CRSI', 'Buy Heavy', 'Regime', 'VolVol', 'Total Trades', 'Port Value', 'Returns', 'BnH', 'Diff', 'MaxDraw', 'Combo'])

starts = ['2023-01-01', '2023-05-01', '2023-09-01', '2024-01-01', '2024-05-01', '2024-09-01', '2023-01-01']
# starts = ['2023-02-01', '2023-06-01', '2023-10-01', '2024-02-01', '2024-06-01', '2024-10-01', '2023-02-01']
# starts = ['2023-03-01', '2023-07-01', '2023-11-01', '2024-03-01', '2024-07-01', '2024-11-01', '2023-03-01']
# starts = ['2023-04-01', '2023-08-01', '2023-12-01', '2024-04-01', '2024-08-01', '2024-12-01', '2023-04-01']

# starts = ['2023-01-01', '2023-05-01', '2023-09-01', '2024-01-01', '2024-05-01', '2024-09-01', '2023-01-01','2023-02-01', '2023-06-01', '2023-10-01', '2024-02-01', '2024-06-01', '2024-10-01', '2023-02-01', '2023-03-01', '2023-07-01', '2023-11-01', '2024-03-01', '2024-07-01', '2024-11-01', '2023-03-01', '2023-04-01', '2023-08-01', '2023-12-01', '2024-04-01', '2024-08-01', '2023-04-01']
len_starts = len(starts)
print(f"Number of starts: {len_starts}")

switches = ['bollinger', 'ma_stretch', 'crsi', 'regime', 'volvol', 'kalman', 'buy_heavy', 'strong_belief']
# switches = ['regime', 'volvol', 'kalman']

from itertools import product
# from tqdm import tqdm

combo = 1

# Loop over all combinations of switches
for p in product([False, True], repeat=7):
    params = dict(zip(switches,p))

    # Set switches
    if params['bollinger'] == True:
        bollinger = True
    else:
        bollinger = False
    if params['ma_stretch'] == True:
        ma_stretch = True
    else:
        ma_stretch = False
    if params['crsi'] == True:
        crsi = True
    else:
        crsi = False
    if params['regime'] == True:
        regime_filter = True
    else:
        regime_filter = False
    if params['volvol'] == True:
        volvol_filter = True
    else:
        volvol_filter = False
    if params['kalman'] == True:
        kalman_filter = True
    else:
        kalman_filter = False
    if params['buy_heavy'] == True:
        buy_heavy = True
    else:
        buy_heavy = False


if 1 == 1:
    one_sim = True # Set to True to run one simulation (and disable 1 == 1)
    starts = ['2023-01-01'] # Set to one start date for one simulation
    len_starts = len(starts) # Length of starts list

    # Test out all periods [EPOCHS]
    for i, start in enumerate(starts):

        end = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days=365)).strftime('%Y-%m-%d')

        if i == len_starts - 1:
            end = '2025-06-12'

        print(f"sim {i+1} - {start} - {end}")


        # end = 1.5 years later
        # end = (datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        # end = '2025-06-12'

        # Beliefs
        low_belief = 1 # 0.8 = 80% belief (20% cash)
        high_belief = 2 # 1.25 pretty good (1.5 --> 94% drawdown)

        if one_sim:
            # Indicator switches
            bollinger = False # Use Bollinger Bands
            ma_stretch = False # Use Moving Average Stretch
            crsi = False # Use Connors RSI
            # Filter switches
            regime_filter = False # Bull/Bear/Flat Regime Filter
            volvol_filter = False # Volatility/Volume Filter
            kalman_filter = False # Kalman Filter
            # Lean towards buy over sell
            buy_heavy = False

        # CRSI parameters
        if buy_heavy:
            crsi_buy_level = 20 # Buy if CRSI is under this
            crsi_sell_level = 95 # Sell if CRSI is over this
        else:
            crsi_buy_level = 10 # Buy if CRSI is under this
            crsi_sell_level = 90 # Sell if CRSI is over this
        rsi_period = 3 # Connors RSI period
        streak_period = 2 # Connors RSI streak period
        percent_rank_period = 100 # Connors RSI percent rank period

        # Bollinger parameters
        bollinger_sd = 2 # Bollinger Bands standard deviation
        sma_window = window # Bollinger Bands SMA window

        # Moving Average Stretch parameters
        stretch_sd = 2 # Moving Average Stretch standard deviation
        ema_window = window # Moving Average Stretch EMA window

        log_message(f"Entering loop!")

        # Get prices
        stock = yf.Ticker(ticker)
        price_history = stock.history(start=start, end=end, interval=interval)

        # Select columns onlys
        # df = price_history[['Close']]
        df = price_history[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.copy()  # Avoid SettingWithCopyWarning


        if ticker == 'NVDA':
            # Add column for NVDL
            stock = yf.Ticker('NVDL')
            nvdla_history = stock.history(start=start, end=end, interval=interval)
            nvdla_history = nvdla_history[['Close']]
            nvdla_history.rename(columns={'Close': 'NVDL'}, inplace=True) # Rename column
            df = df.join(nvdla_history, how='inner') # Join NVDL prices to main df

        # Get CRSI
        df = connors_rsi(df, rsi_period=3, streak_period=2, percent_rank_period=100)

        # Get Bollinger Bands / cutoffs
        df['sma_20'] = df['Close'].rolling(window=window).mean() # 20-day SMA
        df['1_sd_sma'] = df['Close'].rolling(window=window).std() # 1-SD from SMA
        if buy_heavy:
            df['lower_band'] = df['sma_20'] - ((bollinger_sd * 0.5) * df['1_sd_sma']) # Lower band (-1 SD)
            df['upper_band'] = df['sma_20'] + ((bollinger_sd * 1.5) * df['1_sd_sma']) # Upper band (+1 SD)
        else:
            df['lower_band'] = df['sma_20'] - (bollinger_sd * df['1_sd_sma']) # Lower band (-2 SD)
            df['upper_band'] = df['sma_20'] + (bollinger_sd * df['1_sd_sma']) # Upper band (+2 SD)
        bollinger_buy_level = round(df['lower_band'].iloc[-1], 2)
        bollinger_sell_level = round(df['upper_band'].iloc[-1], 2)

        # Get Moving Average Stretch / cutoffs
        df['ema_20'] = df['Close'].ewm(span=window, adjust=False).mean() # 20-day EMA
        df['ema_diff'] = df['Close'] - df['ema_20'] # Difference from EMA
        df['1_sd_ema'] = df['ema_diff'].rolling(window=window).std() # 1-SD from EMA
        df['ma_buy_level'] = df['ema_20'] - (stretch_sd * df['1_sd_ema']) # Buy if EMA diff is below this
        df['ma_sell_level'] = df['ema_20'] + (stretch_sd * df['1_sd_ema']) # Sell if EMA diff is above this
        std_dev_ema = df['1_sd_ema'].iloc[-1] # Current std_dev
        std_dev_ema = round(std_dev_ema, 2)
        ema_diff = df['ema_diff'].iloc[-1]  # Latest EMA difference
        ema_diff = round(ema_diff, 2)
        ma_stretch_buy_level = std_dev_ema * -1 # Buy if EMA diff is below this
        ma_stretch_sell_level = std_dev_ema # Sell if EMA diff is above this

        # Get ATR and Relative Volume
        df = ATR(df, window=window) # Average True Range
        df['ATR_lookback'] = df['ATR'].shift(14) # Shift lookback price
        df = relative_volume(df, window=window) # Relative Volume

        # Get Kalman Filter
        df = kalman(df) # Kalman Filter

        # Latest values for order calculations
        current_price = round(df['Close'].iloc[-1], 2)
        current_crsi = round(df['crsi'].iloc[-1], 2)

        # Tracing
        log_message(f"Latest DataFrame:\n{df}")
        log_message(f"----- Quick Algo Details -----")
        log_message(f"Ticker: {ticker} | Bar Size: {interval} | SMA/EMA Window: {window}")
        log_message(f"Lookback: {lookback} | Midway: {midway}")
        log_message(f"Start Date: {start} | End Date: {end} | Days Ago: {days_ago}")

        # Get current NVDL position
        # try:
        #     position = trading_client.get_open_position('NVDL')
        #     position_value = round(float(position.market_value), 2)
        #     log_message(f"Current Position: {position.symbol} | Market Value: {position_value:,.2f} | Qty: {position.qty} | Avg Price: {position.avg_entry_price}\n")
        # except Exception as e:
        #     position = None
        #     log_message(f">>>>> No position exists for NVDL <<<<<\n")

        log_message(f'----- Indicators & Filters Switches -----')
        log_message(f"Bollinger Bands: {bollinger} | Moving Average Stretch: {ma_stretch} | Connors RSI: {crsi}")
        log_message(f"Regime Filter (Bull/Bear/Bull): {regime_filter} | VolVol Filter: {volvol_filter}\n")

        # Overall buy/sell signal (add short signal later)
        combo_threshold = 0
        if bollinger:
            combo_threshold += 1 # Bollinger Bands
        if ma_stretch:
            combo_threshold += 1 # Moving Average Stretch
        if crsi:
            combo_threshold += 1 # Connors RSI

        # OVerall filter checks
        filter_threshold = 0
        if regime_filter:
            filter_threshold += 1 # Bull/Bear/Flat Regime Filter
        if volvol_filter:
            filter_threshold += 1 # Volatility/Volume Filter
        if kalman_filter:
            filter_threshold += 1 # Kalman Filter

        log_message(f"Combo Threshold: {combo_threshold} | Filter Threshold: {filter_threshold}\n")

        # combo_threshold = 3 # Max of 3
        buy_signals = 0
        sell_signals = 0

        # filter_threshold = 2 # Momentum (bull/bear/flat) and VolVol filters
        filter_checks = 0

        # Conditions for flags
        crsi_conditions = [
            (df['crsi'] <= crsi_buy_level),
            (df['crsi'] >= crsi_sell_level),
            ((df['crsi'] > crsi_buy_level) & (df['crsi'] < crsi_sell_level))
        ]
        bollinger_conditions = [
            (df['Low'] <= df['lower_band']),
            (df['High'] >= df['upper_band']),
            ((df['Low'] > df['lower_band']) & (df['High'] < df['upper_band']))
        ]
        ma_stretch_conditions = [
            (df['ema_diff'] <= ma_stretch_buy_level),
            (df['ema_diff'] >= ma_stretch_sell_level),
            ((df['ema_diff'] > ma_stretch_buy_level) & (df['ema_diff'] < ma_stretch_sell_level))
        ]
        kalman_conditions = [
            (df['Close'] < df['Kalman_Lower']),
            (df['Close'] >= df['Kalman_Lower']),
            (df['Close'] >= df['Kalman_Upper'])
        ]

        flag_values = [1, -1, 0]

        # Set flags
        if crsi:
            df['crsi_flag'] = np.select(crsi_conditions, flag_values)
        else:
            df['crsi_flag'] = 0
        if bollinger:
            df['bollinger_flag'] = np.select(bollinger_conditions, flag_values)
        else:
            df['bollinger_flag'] = 0
        if ma_stretch:
            df['ma_stretch_flag'] = np.select(ma_stretch_conditions, flag_values)
        else:
            df['ma_stretch_flag'] = 0
        if kalman_filter:
            df['kalman_flag'] = np.select(kalman_conditions, flag_values)
        else:
            df['kalman_flag'] = 0
        df = df.assign(flag_tally = lambda x: x['crsi_flag'] + x['bollinger_flag'] + x['ma_stretch_flag'] + x['kalman_flag'])

        # Lookback and midway prices (for regime filter)
        df['lookback'] = df['Close'].shift(lookback* -1) # Shift lookback price
        df['midway'] = df['Close'].shift(midway * -1) # Shift midway price


        df = df.assign(regime1_filter = lambda x: np.select(
            [(x['Close'] > x['lookback']), 
            (x['Close'] < x['lookback'])], 
            ['Bull', 'Bear'],
            default='Flat'))
        df = df.assign(regime2_filter = lambda x: np.select(
            [(x['Close'] > x['midway']), 
            (x['Close'] < x['midway'])], 
            ['Bull', 'Bear'],
            default='Flat'))
        # Bear/Bull/Flat regime filter
        if regime_filter:
            df = df.assign(full_regime = lambda x: np.select(
                [(x['regime1_filter'] == 'Bull') & (x['regime2_filter'] == 'Bull'),
                (x['regime1_filter'] == 'Bear') & (x['regime2_filter'] == 'Bear')],
                [1, -1],
                default=0))
        else:
            df['full_regime'] = 0

        # VolVol Filter
        df = df.assign(volume_filter = lambda x: np.select(
            [(x['Relative_Volume'] > 1.0), 
            (x['Relative_Volume'] < 1.0)], 
            ['High', 'Low'],
            default='Normal'))
        df = df.assign(volatility_filter = lambda x: np.select(
            [(x['ATR'] > x['ATR_lookback']),
            (x['ATR'] < x['ATR_lookback'])], 
            ['High', 'Low'],
            default='Normal'))
        if volvol_filter:
            df = df.assign(volvol_full = lambda x: np.select(
                [(x['volume_filter'] == 'High') & (x['volatility_filter'] == 'Low'),
                (x['volume_filter'] == 'Low') & (x['volatility_filter'] == 'High')],
                [1, 1],
                default=0))
        else:
            df['volvol_full'] = 0

        # Kalman filter
        if kalman_filter:
            df = df.assign(kalman_filter = lambda x: np.select(
                [(x['kalman_flag']) == 1,
                (x['kalman_flag']) < 1],
                [1, -1],
                default=0))
        else:
            df['kalman_filter'] = 0

        if regime_filter:
            df = df.assign(full_regime = lambda x: np.select(
                [(x['regime1_filter'] == 'Bull') & (x['regime2_filter'] == 'Bull'),
                (x['regime1_filter'] == 'Bear') & (x['regime2_filter'] == 'Bear')],
                [1, -1],
                default=0))
        else:
            df['full_regime'] = 0

        df = df.assign(buy_filter_tally = lambda x: x['full_regime'] + x['volvol_full'] + x['kalman_filter'])
        df = df.assign(sell_filter_tally = lambda x: x['full_regime'] - x['volvol_full'] + x['kalman_filter'])


        df = df.assign(decision = lambda x: np.select(
            [(x['flag_tally'] >= combo_threshold) & (x['buy_filter_tally'] >= filter_threshold),
            (x['flag_tally'] <= -combo_threshold) & (x['sell_filter_tally'] <= filter_threshold)],
            ['Buy', 'Sell'],
            default='Hold'))

        # START BACKTEST
        starting_cap = 50000.0 # Starting capital
        max_value = starting_cap # Max portfolio value
        bnh_max_value = starting_cap # Max buy and hold value
        max_drawdown = 0.0 # Max drawdown
        bnh_max_drawdown = 0.0
        max_drawdown_pct = 0.0 # Max drawdown percentage
        bnh_max_drawdown_pct = 0.0 # Max drawdown percentage for buy and hold
        # current_belief = 0.8 # Start with low belief

        # Make the index not a datetime index
        df['timestamp'] = df.index # Save index as a column
        # Make the index a range index
        df.reset_index(drop=True, inplace=True) # Reset index to range index

        # Intialize columns for backtest
        df.loc[0, '1_day_return'] = 0.0 # So no NaN
        df['port_value'] = 0.0
        df['regt'] = 0.0 
        df['total_shares'] = 0.0
        df['change_in_shares'] = 0.0
        df['share_value'] = 0.0
        df['cash'] = 0.0
        df['position'] = 'short'
        df['trade'] = 0.0 # Trade value
        df['bnh'] = 0.0 # Buy and hold value
        df['trade_pnl'] = 0.0 # Trade PnL

        trade_count = 0 # Count trades
        bankrupt = False # Flag for bankruptcy

        for i in range(0, len(df)):

            # Set initial cash pile
            if i == 0:
                df.loc[i, 'share_value'] = low_belief * starting_cap # 0.8 belief
                df.loc[i, 'cash'] =  starting_cap - df.loc[i, 'share_value'] # 0.2 cash
                if ticker == 'NVDA':
                    df.loc[i, 'total_shares'] = df['share_value'].iloc[i] / df['NVDL'].iloc[i] # Shares owned
                else:
                    df.loc[i, 'total_shares'] = df['share_value'].iloc[i] / df['Close'].iloc[i] # Shares owned
                df.loc[i, 'bnh'] = starting_cap # Buy and hold value
                
            else:

                # If port_value is <= 0, 'bankrupt' 
                # if df['port_value'].iloc[i-1] <= 0:
                #     bankrupt = True
                #     log_message(f"Bankruptcy at index {i} with portfolio value: {df['port_value'].iloc[i-1]}")
                #     break

                # Bring forward Cash
                df.loc[i, 'cash'] = df['cash'].iloc[i-1]

                # Bring forward total number of shares
                df.loc[i, 'total_shares'] = df['total_shares'].iloc[i-1]

                # Bring forward position
                df.loc[i, 'position'] = df['position'].iloc[i-1]

                # START TESTING

                if ticker == 'NVDA':
                    df.loc[i, 'share_value'] = df['total_shares'].iloc[i] * df['NVDL'].iloc[i]
                else:
                    # Calculate share value for non-NVDL tickers
                    df.loc[i, 'share_value'] = df['total_shares'].iloc[i] * df['Close'].iloc[i]

                # Calculate the portfolio value
                df.loc[i, 'port_value'] = df['cash'].iloc[i] + df['share_value'].iloc[i]

                trading_capital = (high_belief - low_belief) * df['port_value'].iloc[i]

                # END TESTING (below disabled)

                # Update amount to buy/sell
                # trading_capital = (high_belief - low_belief) * df['port_value'].iloc[i-1] # 40% of portfolio value

                if df.loc[i, 'decision'] == 'Buy' and df.loc[i, 'position'] == 'short': # Buy if in short position

                    # For buy
                    trading_capital = (high_belief - low_belief) * df['port_value'].iloc[i]

                    # Calculate the number of shares to buy
                    if ticker == 'NVDA':
                        df.loc[i, 'change_in_shares'] = trading_capital / df['NVDL'].iloc[i]
                    else:
                        # Calculate shares for non-NVDL tickers
                        df.loc[i, 'change_in_shares'] = trading_capital / df['Close'].iloc[i]

                    # Update the total number of shares
                    df.loc[i, 'total_shares'] = df['total_shares'].iloc[i-1] + df['change_in_shares'].iloc[i]

                    # Update the cash pile
                    if ticker == 'NVDA':
                        df.loc[i, 'cash'] = df['cash'].iloc[i-1] - (df['change_in_shares'].iloc[i] * df['NVDL'].iloc[i])
                    else:
                        df.loc[i, 'cash'] = df['cash'].iloc[i-1] - (df['change_in_shares'].iloc[i] * df['Close'].iloc[i])

                    # Update position
                    df.loc[i, 'position'] = 'long'

                    # Update trade value
                    df.loc[i, 'trade'] = 1

                    trade_count += 1 # Increment trade count

                elif df.loc[i, 'decision'] == 'Sell' and df.loc[i, 'position'] == 'long': # Sell if in long position

                    # For sell (needs work for less than 2.0 belief). [just cash out of margin for now]
                    trading_capital = -1 * df['cash'].iloc[i]

                    # Calculate the number of shares to sell
                    if ticker == 'NVDA':
                        df.loc[i, 'change_in_shares'] = trading_capital / df['NVDL'].iloc[i]
                    else:
                        df.loc[i, 'change_in_shares'] = trading_capital / df['Close'].iloc[i]

                    # Update the total number of shares
                    df.loc[i, 'total_shares'] = df['total_shares'].iloc[i-1] - df['change_in_shares'].iloc[i]

                    # Update the cash pile
                    if ticker == 'NVDA':
                        df.loc[i, 'cash'] = df['cash'].iloc[i-1] + (df['change_in_shares'].iloc[i] * df['NVDL'].iloc[i])
                    else:
                        # Update cash for non-NVDL tickers
                        df.loc[i, 'cash'] = df['cash'].iloc[i-1] + (df['change_in_shares'].iloc[i] * df['Close'].iloc[i])

                    # Update position
                    df.loc[i, 'position'] = 'short'

                    # Update trade value
                    df.loc[i, 'trade'] = -1

                    trade_count += 1 # Increment trade count

                # Get yesterday's bnh value
                df.loc[i, 'bnh'] = df['bnh'].iloc[i-1]

                # Update the bnh value
                if ticker == 'NVDA':
                    # If NVDL, use NVDL price
                    df.loc[i, 'bnh'] = df.loc[i, 'bnh'] * (1 + df['nvdl_1_day_return'].iloc[i])
                else:
                    df.loc[i, 'bnh'] = df.loc[i, 'bnh'] * (1 + df['1_day_return'].iloc[i])

            # Calculate the shares + portfolio value
            if ticker == 'NVDA':
                df.loc[i, 'share_value'] = df['total_shares'].iloc[i] * df['NVDL'].iloc[i]
            else:
                # Calculate share value for non-NVDL tickers
                df.loc[i, 'share_value'] = df['total_shares'].iloc[i] * df['Close'].iloc[i]

            # Calculate the portfolio value
            df.loc[i, 'port_value'] = df['cash'].iloc[i] + df['share_value'].iloc[i]

            # Calculae trade PnL
            if df.loc[i, 'trade'] == 1:
                open_trade_value = df['port_value'].iloc[i]
            elif df.loc[i, 'trade'] == -1:
                # Get close value for trade PnL
                close_trade_value = df['port_value'].iloc[i]
                df.loc[i, 'trade_pnl'] = close_trade_value - open_trade_value # Calculate trade PnL

            # Calculate max portfolio value and drawdown
            if df.loc[i, 'port_value'] > max_value:
                max_value = df.loc[i, 'port_value']
            if df.loc[i, 'port_value'] < max_value:
                # Calculate drawdown
                drawdown = max_value - df.loc[i, 'port_value']
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                # Calculate max drawdown percentage
                if max_value != 0:
                    drawdown_pct = (drawdown / max_value) * 100

                    if drawdown_pct > max_drawdown_pct:
                        max_drawdown_pct = drawdown_pct

            if df.loc[i, 'bnh'] > bnh_max_value:
                bnh_max_value = df.loc[i, 'bnh']
            if df.loc[i, 'bnh'] < bnh_max_value:
                # Calculate buy and hold drawdown
                bnh_drawdown = bnh_max_value - df.loc[i, 'bnh']
                if bnh_drawdown > bnh_max_drawdown:
                    bnh_max_drawdown = bnh_drawdown
  
                # Calculate BnH max drawdown percentage
                if bnh_max_value != 0:
                    bnh_drawdown_pct = (bnh_drawdown / bnh_max_value) * 100

                    if bnh_drawdown_pct > bnh_max_drawdown_pct:
                        bnh_max_drawdown_pct = bnh_drawdown_pct
            
        # Set index back to datetime
        df.set_index('timestamp', inplace=True)

        log_message(f"----- Backtest Summary -----")
        buy_count = len(df[df['trade'] == 1]) # Count buy trades
        sell_count = len(df[df['trade'] == -1]) # Count sell trades
        port_value = df['port_value'].iloc[-1] # Final portfolio value
        port_return = ((port_value - df['port_value'].iloc[0]) / df['port_value'].iloc[0]) * 100  # Calculate percentage return
        bnh_value = df['bnh'].iloc[-1]  # Buy and hold value
        bnh_return = ((bnh_value - df['bnh'].iloc[0]) / df['bnh'].iloc[0]) * 100  # Buy and hold percentage return
        log_message(f"Total Trades: {trade_count} | Buy Trades: {buy_count} | Sell Trades: {sell_count}")
        log_message(f"Final Portfolio Value: ${port_value:,.2f} ({round((port_return), 2)}%)")
        log_message(f"Buy and Hold Value: ${bnh_value:,.2f} ({round((bnh_return), 2)}%)")
        log_message(f"Buy and Hold Max Drawdown: ${bnh_max_drawdown:,.2f} ({round(bnh_max_drawdown_pct, 2)}%)")
        log_message(f"Max Portfolio Value: ${max_value:,.2f} | Max Drawdown: ${max_drawdown:,.2f} ({round(max_drawdown_pct, 2)}%)")
        algo_period = f"{start} to {end}"
        log_message(f"Backtest Period: {algo_period} | Ticker: {ticker} | Interval: {interval}")

        # END BACKTEST

        if 1 == 0:
            # Save summary to a text file
            summary_file = f"{ticker}_backtest_summary.txt"
            with open(summary_file, 'a') as f:
                if ticker == 'NVDA':
                    real_ticker = 'NVDL'
                f.write(f"----- Backtest Summary ({real_ticker} - {algo_period} [{interval}])-----\n")
                f.write(f"Bollinger Bands: {bollinger} | Moving Average Stretch: {ma_stretch} | Connors RSI: {crsi}\n")
                f.write(f"Regime Filter (Bull/Bear/Flat): {regime_filter} | VolVol Filter: {volvol_filter} | Buy Heavy: {buy_heavy}\n")
                f.write(f"Low Belief: {low_belief} | High Belief: {high_belief}\n")
                f.write(f"Total Trades: {trade_count} | Buy Trades: {buy_count} | Sell Trades: {sell_count}\n")
                f.write(f"Final Portfolio Value: ${port_value:,.2f} ({round((port_return), 2)}%)\n")
                f.write(f"Buy and Hold Value: ${bnh_value:,.2f} ({round((bnh_return), 2)}%)\n")
                f.write(f"BnH max drawdown: ${bnh_max_drawdown:,.2f} ({round(bnh_max_drawdown_pct, 2)}%)\n")
                f.write(f"Max Portfolio Value: ${max_value:,.2f} | Max Drawdown: ${max_drawdown:,.2f} ({round(max_drawdown_pct, 2)}%)\n")
                # f.write(f"Backtest Period: {algo_period} | Ticker: {ticker} | Interval: {interval}\n")

        if one_sim:
            # Make index timezone unaware
            df.index = df.index.tz_localize(None)  # Make index timezone unaware

            try:
                os.chdir(r'C:\Users\eklundmd\OneDrive - PPD\Desktop\Python Projects\backtest')
            except:
                log_message(f"all gucci")    

            # Save df to excel
            df.to_excel(f"{ticker}_backtest_df.xlsx", index=True)

        new_results_row = pd.DataFrame({
            'Start': [start],
            'End': [end],
            'Low Belief': [low_belief],
            'High Belief': [high_belief],
            'Bollinger': [bollinger],
            'MA Stretch': [ma_stretch],
            'CRSI': [crsi],
            'Buy Heavy': [buy_heavy],
            'Regime': [regime_filter],
            'VolVol': [volvol_filter],
            'Kalman': [kalman_filter],
            'Total Trades': [trade_count],
            'Port Value': [round(port_value, 2)],
            'Returns': [round(port_return, 2)],
            'BnH': [round(bnh_return, 2)],
            'Diff': [round(port_return - bnh_return, 2)],
            'MaxDraw': [round(max_drawdown_pct, 2)],
            'Combo': [combo]
        })
        df_results = pd.concat([df_results, new_results_row], ignore_index=True)

    print(f"************ Count {combo}/64 **************")
    time.sleep(2)
    combo += 1

# Read existing results from excel
try:
    df_results_og = pd.read_excel(f'backtest_summary_{ticker}.xlsx')

    df_results = pd.concat([df_results_og, df_results], ignore_index=True)
    df_results.to_excel(f'backtest_summary_{ticker}.xlsx', index=False)


except FileNotFoundError:
    # Save results to excel
    df_results.to_excel(f'backtest_summary_{ticker}.xlsx', index=False)

    

if one_sim:

    # Plot closing price and volume
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(3, 2, 1) # rows=2, cols=1, index=1
    ax1.plot(df['Close'], label='Closing Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.grid()
    ax2 = ax1.twinx()  # Create a second y-axis for volume
    ax2.bar(df.index, df['Volume'], label='Volume', color='orange', align='center', width=0.5)
    ax2.set_ylabel('Volume')
    plt.title(f'{ticker} Closing Price + Volume ({interval})')
    plt.xlabel('Date')
    plt.legend()
    # plt.grid()

    # Plot ATR and Relative Volume
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(df['ATR'], label='20-day ATR (Volatility)', color='purple', alpha=0.5)
    ax3.set_ylabel('ATR')
    ax3.grid()
    ax3_2 = ax3.twinx()  # Create a second y-axis for relative volume
    ax3_2.set_ylabel('Relative Volume')
    ax3_2.plot(df['Relative_Volume'], label='Rel. Volume', color='green', alpha=0.5)
    plt.title(f'ATR + Relative Volume ({interval})')
    plt.xlabel('Date')
    plt.legend()
    # plt.grid()

    # Plot port_value vs bnh from backtest
    ax7 = plt.subplot(3, 2, 5)
    ax7.plot(df['port_value'], label='Portfolio Value', color='blue')
    ax7.plot(df['bnh'], color='orange')
    # Plot arrows for buy/sell signals
    plt.scatter(df.index[df['trade'] == 1], df['port_value'][df['trade'] == 1], marker='^', color='green',  s=50)
    plt.scatter(df.index[df['trade'] == -1], df['port_value'][df['trade'] == -1], marker='v', color='red', s=50)
    ax7.sharex(ax1) # share x axis with ax1
    ax7.set_ylabel('Value')
    ax7.set_xlabel('Date')
    plt.title(f'Portfolio Value vs Buy and Hold ({interval})')
    plt.legend(loc='lower left')
    plt.grid()

    if 1 == 0:
        # Plot Price and Relative Volume
        ax3 = plt.subplot(3, 2, 2)
        ax3.plot(df['Close'], label='Closing Price', color='blue')
        ax3.set_ylabel('Price')
        ax3_2 = ax3.twinx()  # Create a second y-axis for relative volume
        ax3_2.plot(df.index, df['Relative_Volume'], label='Relative Volume', color='orange')
        ax3_2.set_ylabel('Relative Volume')
        plt.xlabel('Date')
        plt.title(f'Closing Price + Relative Volume ({interval})')
        plt.legend()
        plt.grid()

        # Plot Price and ATR
        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(df['Close'], label='Closing Price', color='blue')
        ax6.set_ylabel('Price')
        ax6_2 = ax6.twinx()  # Create a second y-axis for ATR
        ax6_2.plot(df.index, df['ATR'], label='20-day ATR', color='purple')
        ax6_2.set_ylabel('ATR')
        plt.xlabel('Date')
        plt.title(f'Closing Price + ATR ({interval})')
        plt.legend()
        plt.grid()

    # Plot CRSI
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(df['crsi'], label='CRSI (3,2,100)')
    ax4.sharex(ax1) # share x axis with ax1
    plt.axhline(10, linestyle='--', color='black', label='Oversold (Buy)', alpha=0.5)
    plt.axhline(90, linestyle='--', color='black', label='Overbought (Sell)', alpha=0.5)
    # Plot arrows for buy/sell signals
    plt.scatter(df.index[df['crsi'] < crsi_buy_level], df['crsi'][df['crsi'] < crsi_buy_level], marker='^', color='green', label='Buy Signal', s=50)
    plt.scatter(df.index[df['crsi'] > crsi_sell_level], df['crsi'][df['crsi'] > crsi_sell_level], marker='v', color='red', label='Sell Signal', s=50)
    plt.title(f'CRSI')
    plt.xlabel('Date')
    plt.ylabel('CRSI')
    # plt.legend()
    plt.grid()

    # Plot Bollinger Bands
    ax5 = plt.subplot(3, 2, 2)
    ax5.plot(df['Close'], label='Closing Price', color='blue')
    ax5.plot(df['sma_20'], label='20-day SMA', color='orange')
    plt.plot(df['lower_band'], label=f'Lower Band (-2 Sigma)', color='black', alpha=0.5)
    plt.plot(df['upper_band'], label=f'Upper Band (+2 Sigma)', color='black', alpha=0.5)
    # Plot arrows for buy/sell signals
    plt.scatter(df.index[df['Close'] < df['lower_band']], df['lower_band'][df['Close'] < df['lower_band']], marker='^', color='green', label='Buy Signal', s=50)
    plt.scatter(df.index[df['Close'] > df['upper_band']], df['upper_band'][df['Close'] > df['upper_band']], marker='v', color='red', label='Sell Signal', s=50)
    plt.title(f'Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    # plt.legend()
    plt.grid()

    # Plot Moving Average Stretch
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(df['Close'], label='Closing Price', color='blue')
    ax6.plot(df['ema_20'], label='20-day EMA', color='orange')
    ax6.plot(df['ma_buy_level'], label='Buy Level (1-SD Below EMA)', color='green', linestyle='--')
    ax6.plot(df['ma_sell_level'], label='Sell Level (1-SD Above EMA)', color='red', linestyle='--')
    # Plot arrows for buy/sell signals

    plt.title(f'Moving Average Stretch')
    plt.xlabel('Date')
    plt.ylabel('Price')
    # plt.legend()
    plt.grid()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
