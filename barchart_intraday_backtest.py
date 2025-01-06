import os
import pandas as pd

os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\barchart_downloads')

ticker = 'tqqq'
trail_stop = 0.0165

# Read the data from the CSV file
data = pd.read_csv(f'{ticker}_intraday-30min.csv')
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
data['Time'] = data['Time'].str.split(' ').str[1]

# New dataframe with date, time, open columns
data = data[['Date', 'Time', 'Open']]

# Add a Daily High column and fill it with the Open column
# data['Daily High'] = 0
data = data.assign(Daily_High=0)

# Iterate through the data and if the time is 9:30, update the Daily High column with the Open column
print(len(data))
for i in range(0, len(data)):
    # print(data.loc[i, 'Time'])
    if data.loc[i, 'Time'] == '9:30':
        print(i)
        data.loc[i, 'Daily_High'] = data.loc[i, 'Open']

# Iterate through the data and if the Open column is greater than the Daily High column, update the Daily High column
for i in range(1, len(data)):
    if data.loc[i, 'Open'] > data.loc[i-1, 'Daily_High']:
        data.loc[i, 'Daily_High'] = data.loc[i, 'Open']
    else:
        if data.loc[i, 'Time'] != '9:30':
            data.loc[i, 'Daily_High'] = data.loc[i-1, 'Daily_High']

# Percent change
# data['Percent Change'] = (data['Open'] / data['Daily High']) - 1

data = data.assign(Percent_Change=data['Open']/data['Daily_High'] - 1)

# New column called 'Flag' set to 0 (no action)
# data['Flag'] = 0
data = data.assign(Flag=0)

# Where time is 9:30, set Flag to 1
data.loc[data['Time'] == '9:30', 'Flag'] = 1

# Where time is 15:30, set Flag to -1
data.loc[data['Time'] == '15:30', 'Flag'] = -1

# Where precent change is less than -0.01 and time is not 9:30, set Flag to -1
data.loc[(data['Percent_Change'] < -trail_stop) & (data['Time'] != '9:30'), 'Flag'] = -1

data = data.assign(Shares=0.0)
data = data.assign(Share_Value=0.0)
data = data.assign(Cash=0.0)
data = data.assign(Account_Value=0.0)

# Set initial cash pile
data.loc[0, 'Cash'] = 0
data.loc[0, 'Account_Value'] = 10000
data.loc[0, 'Shares'] = data.loc[0, 'Account_Value'] / data.loc[0, 'Open']
data.loc[0, 'Share_Value'] = data.loc[0, 'Shares'] * data.loc[0, 'Open']

# Set initial buy and hold comparison
data.loc[0, 'Buy_and_Hold'] = 10000

# Iterate over data. If flag is -1, sell all shares
for i in range(1, len(data)):

    # -1% trail stop loss
    if data.loc[i, 'Flag'] == -1:

        # Sell all shares if shares are greater than 0
        if data.loc[i-1, 'Shares'] > 0:
            data.loc[i, 'Cash'] = data.loc[i-1, 'Shares'] * data.loc[i, 'Open']
            data.loc[i, 'Shares'] = 0
        else:
            data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']

    # Buy at 9:30
    elif data.loc[i, 'Flag'] == 1:
        data.loc[i, 'Shares'] = data.loc[i-1, 'Account_Value'] / data.loc[i, 'Open']
        data.loc[i, 'Cash'] = 0

    # No action
    else:
        data.loc[i, 'Shares'] = data.loc[i-1, 'Shares']
        data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']


        # if data.loc[i-1, 'Flag'] != -1:
        #     data.loc[i, 'Shares'] = data.loc[i-1, 'Shares']
        # elif data.loc[i-1, 'Flag'] == -1:
        #     data.loc[i, 'Cash'] = data.loc[i-1, 'Cash']

    data.loc[i, 'Share_Value'] = data.loc[i, 'Shares'] * data.loc[i, 'Open']
    data.loc[i, 'Account_Value'] = data.loc[i, 'Cash'] + data.loc[i, 'Share_Value']

    # Buy and hold
    data.loc[i, 'Buy_and_Hold'] = data.loc[0, 'Shares'] * data.loc[i, 'Open']

    final_value = data.loc[i, 'Account_Value']
    final_bah = data.loc[i, 'Buy_and_Hold']

    # Format in dollars
    format_value = "${:,.2f}".format(final_value)
    format_bah = "${:,.2f}".format(final_bah)

print(data)

# Save the data to a CSV file
data.to_csv('test_data_processed.csv', index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['Account_Value'], label=f'{ticker} -{trail_stop}% Trail Stop Loss')
plt.plot(data['Buy_and_Hold'], label='Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.suptitle(f'{format_value} vs {format_bah} B&H')

# save the image
plt.savefig(f'{ticker}_processed.png')

plt.show()

