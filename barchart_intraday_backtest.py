import os
import pandas as pd
from openpyxl import load_workbook
from tqdm import tqdm
import datetime

'''
Best so far (TQQQ):
-0.01 trail stop loss
-0.025 trail stop buy
9:35am entry
3:30pm exit

- 2.5% buy = 25% better w/ 1min, 237% w/ 30min. 
- 3% buy = 12% better w/ 1min, 46% w/ 30min

BTC:
- 0.02 trail stop loss
- 0.04 trail stop buy
- no entry/exits
'''

# Input parameters
ticker = 'tqqq'
starting_cap = 50000
start_date = 'all'
start_date = '2024-01-01'
end_date = '2024-02-28'
# end_date = '2023-12-29'
trail_stop_loss = 0.01 # 1% stop loss (TQQQ)
# limit_buy = 0.025 # 2.5% buy (TQQQ)
limit_buy = 0.015 # 0.015
# trail_stop_loss = 0.02 # BTC
# limit_buy = 0.04 # BTC
candle_size = '1min'
# morning_entry = '9:30' # No zero to left
morning_entry = '9:35' # No zero to left
afternoon_exit = '15:30' # No zero to left
if candle_size == '30min':
    morning_entry = '9:30'
    afternoon_exit = '15:30'

# fill_correction = 0.10 # 10% fill correction
fill_correction = 0.00 # 0.15 might be best for TQQQ to account for slippage

if ticker == 'btc':
    morning_entry = '0:02'
    afternoon_exit = '0:01'
    if candle_size == '30min':
        morning_entry = '9:00'
        afternoon_exit = '9:30'
extra = '_2024'
# extra = '_Jan2025'
# extra = ''

# Get datetimes for entries/exits
morning_entry_dt = datetime.datetime.strptime(morning_entry, '%H:%M').time()
afternoon_exit_dt = datetime.datetime.strptime(afternoon_exit, '%H:%M').time()

print(f"Morning entry: {morning_entry_dt} | Afternoon exit: {afternoon_exit_dt}")

# Read the data from the CSV file
os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\barchart_downloads')
data = pd.read_csv(f'{ticker}_intraday-{candle_size}{extra}.csv')
# data = pd.read_csv('test_data.csv')

# Reverse the data
data = data.iloc[::-1]

# Reset the index
data = data.reset_index(drop=True)

# Split time into date and time
data['Date'] = data['Time'].str.split(' ').str[0]

# Convert date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convert time to datetime
data['Time'] = pd.to_datetime(data['Time'].str.split(' ').str[1], format='%H:%M').dt.time

# Only keep between 01-01-2024 and 02-28-2024
if start_date != 'all':
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Reset the index
data = data.reset_index(drop=True)

# New dataframe with date, time, open columns
data = data[['Date', 'Time', 'Open']]

# Corrective fills
data['Buy_Fill_Correction'] = data['Open'] + fill_correction
data['Sell_Fill_Correction'] = data['Open'] - fill_correction

# Add a Daily High column and fill it with the Open column
# data['Daily High'] = 0
data = data.assign(Daily_High=0.0)
data = data.assign(Daily_Low=0.0)

# Iterate through the data and first 9:30 entry is the new Daily High
print(len(data))
for i in tqdm(range(0, len(data))):
    # print(data.loc[i, 'Time'])
    if ticker == 'btc':
        if data.loc[i, 'Time'] == '00:00': # Reset daily for BTC
            data.loc[i, 'Daily_High'] = data.loc[i, 'Open']
            data.loc[i, 'Daily_Low'] = data.loc[i, 'Open']
    # elif data.loc[i, 'Time'] == '9:30': # Reset daily for stocks
    elif data.loc[i, 'Time'] == morning_entry_dt: # Reset daily for stocks
        # print(i)
        data.loc[i, 'Daily_High'] = data.loc[i, 'Open']
        data.loc[i, 'Daily_Low'] = data.loc[i, 'Open']

# Iterate through the data and if the Open column is greater than the Daily High column, update the Daily High column
for i in tqdm(range(1, len(data))):
    if data.loc[i, 'Open'] > data.loc[i-1, 'Daily_High']:
        data.loc[i, 'Daily_High'] = data.loc[i, 'Open']
    else:
        if data.loc[i, 'Time'] != morning_entry_dt:
            data.loc[i, 'Daily_High'] = data.loc[i-1, 'Daily_High']

    if data.loc[i, 'Open'] < data.loc[i-1, 'Daily_Low']:
        data.loc[i, 'Daily_Low'] = data.loc[i, 'Open']
    else:
        if data.loc[i, 'Time'] != morning_entry_dt:
            data.loc[i, 'Daily_Low'] = data.loc[i-1, 'Daily_Low']

data = data.assign(Perc_Trail_Loss=data['Open']/data['Daily_High'] - 1)

# New column called 'Flag' set to 0 (no action)
# data['Flag'] = 0
data = data.assign(Flag=0)

# New column called 'Limit_Buy_Mark' set to 0 (no action)
data = data.assign(Limit_Buy_Mark=0.0)

# Where time == morning_entry input, set Flag to 1
data.loc[data['Time'] == morning_entry_dt, 'Flag'] = 1

# Where time is 15:30, set Flag to -1
data.loc[data['Time'] == afternoon_exit_dt, 'Flag'] = -1

# Set all times after 15:30 to -1
# data.loc[data['Time'] >= afternoon_exit, 'Flag'] = -1

# Where precent change is less than -0.01 and time is not the morning_entry, set Flag to -1
# data.loc[(data['Perc_Trail_Loss'] < -trail_stop_loss) & (data['Time'] != morning_entry_dt), 'Flag'] = -1

# Where precent change is greater than -(limit_buy input) and time is not morning_entry, set Flag to 1
# if limit_buy > 0:
#     data.loc[(data['Perc_Trail_Loss'] < -limit_buy) & (data['Time'] != morning_entry_dt) & (data['Time'] <= afternoon_exit_dt), 'Flag'] = 1

data = data.assign(Shares=0.0)
data = data.assign(Share_Value=0.0)
data = data.assign(Cash=0.0)
data = data.assign(Account_Value=0.0)

# Set initial cash pile
# data.loc[0, 'Cash'] = 0
# data.loc[0, 'Account_Value'] = 10000
# data.loc[0, 'Shares'] = data.loc[0, 'Account_Value'] / data.loc[0, 'Open']
# data.loc[0, 'Share_Value'] = data.loc[0, 'Shares'] * data.loc[0, 'Open']
data.loc[0, 'Cash'] = starting_cap
data.loc[0, 'Account_Value'] = starting_cap
data.loc[0, 'Shares'] = 0
data.loc[0, 'Share_Value'] = 0
initial_shares = starting_cap / data.loc[0, 'Open']

# Set initial buy and hold comparison
data.loc[0, 'Buy_and_Hold'] = starting_cap

# Iterate over data. If flag is -1, sell all shares
for i in tqdm(range(1, len(data))):

    # Sell flag if trail stop loss is hit and shares are greater than 0 and limit buy not set yet
    if data.loc[i, 'Perc_Trail_Loss'] < -trail_stop_loss and data.loc[i-1, 'Shares'] > 0 and data.loc[i-1, 'Limit_Buy_Mark'] == 0:
        data.loc[i, 'Flag'] = -1

    # -1% trail stop loss
    if data.loc[i, 'Flag'] == -1:

        # Sell all shares if shares are greater than 0
        if data.loc[i-1, 'Shares'] > 0:
            data.loc[i, 'Cash'] = data.loc[i-1, 'Shares'] * data.loc[i, 'Sell_Fill_Correction'] # data.loc[i, 'Open']
            data.loc[i, 'Shares'] = 0

            limit_buy_mark = data.loc[i, 'Open'] * (1 - limit_buy)
            if data.loc[i, 'Time'] != afternoon_exit_dt:
                data.loc[i, 'Limit_Buy_Mark'] = limit_buy_mark
        else:
            data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']

    # Buy at 9:30/whenever (morning_entry input)
    elif data.loc[i, 'Flag'] == 1:
        
        if data.loc[i-1, 'Flag'] != -1:
            data.loc[i, 'Shares'] = data.loc[i-1, 'Account_Value'] / data.loc[i, 'Buy_Fill_Correction'] # data.loc[i, 'Open']

        else:
            data.loc[i, 'Shares'] = data.loc[i-1, 'Shares']

        data.loc[i, 'Cash'] = 0

    # No action
    else:
        data.loc[i, 'Shares'] = data.loc[i-1, 'Shares']
        data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']


        # if data.loc[i-1, 'Flag'] != -1:
        #     data.loc[i, 'Shares'] = data.loc[i-1, 'Shares']
        # elif data.loc[i-1, 'Flag'] == -1:
        #     data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']

    if data.loc[i-1, 'Limit_Buy_Mark'] > 0 and data.loc[i, 'Time'] <= afternoon_exit_dt:
        data.loc[i, 'Limit_Buy_Mark'] = data.loc[i-1, 'Limit_Buy_Mark']

    # Limit buy order goes through (and less than 3:30pm)
    if data.loc[i, 'Open'] <= data.loc[i, 'Limit_Buy_Mark'] and data.loc[i, 'Time'] < afternoon_exit_dt and data.loc[i-1, 'Shares'] == 0:
        data.loc[i, 'Shares'] = data.loc[i-1, 'Account_Value'] / data.loc[i, 'Buy_Fill_Correction'] # data.loc[i, 'Open']
        data.loc[i, 'Cash'] = 0
        data.loc[i, 'Flag'] = 1

    data.loc[i, 'Share_Value'] = data.loc[i, 'Shares'] * data.loc[i, 'Open']
    data.loc[i, 'Account_Value'] = data.loc[i, 'Cash'] + data.loc[i, 'Share_Value']

    # Buy and hold
    # data.loc[i, 'Buy_and_Hold'] = data.loc[0, 'Shares'] * data.loc[i, 'Open']
    data.loc[i, 'Buy_and_Hold'] = initial_shares * data.loc[i, 'Open']

    final_value = data.loc[i, 'Account_Value']
    final_bah = data.loc[i, 'Buy_and_Hold']
    percent_diff = round(((final_value / final_bah) - 1) * 100, 2)

    # Format in dollars
    format_value = "${:,.2f}".format(final_value)
    format_bah = "${:,.2f}".format(final_bah)

print(data)

file = 'test_data_processed.xlsx'

# Save the data to a CSV file
# data.to_csv(file, index=False)

# Save the data to an Excel file
data.to_excel(file, index=False)

# Freeze the top pane
wb = load_workbook(file)
ws = wb.active
ws.freeze_panes = 'A2'
wb.save(file)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['Account_Value'], label=f'{ticker} -{trail_stop_loss}% Trail Stop Loss/ +{limit_buy}% Trail Stop Buy')
plt.plot(data['Buy_and_Hold'], label='Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.suptitle(f'{format_value} \nvs\n {format_bah} B&H ({percent_diff}%)')

# save the image
plt.savefig(f'{ticker}_processed.png')

plt.show()

