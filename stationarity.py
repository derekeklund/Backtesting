import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from tqdm import tqdm
from statsmodels.tsa.stattools import coint, adfuller
import datetime

# Get stationarity of the data
def check_for_stationarity(X, cutoff=0.01):
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name + ' is likely non-stationary.')
        return False

# Input parameters
ticker = 'tqqq'
starting_cap = 50000
start_date = '2024-01-01'
end_date = '2024-02-28'
year = start_date[:4]


# Read the data from the CSV file
os.chdir(r'C:\Users\derek\Coding Projects\Backtesting\barchart_downloads')
data = pd.read_csv(f'{ticker}_intraday-1min_{year}.csv')

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

# Get Open as series
open = pd.Series(data['Open'])

print(f"open (name: {open.name}): {open}")
# check_for_stationarity(open)


thirtyMin = []
negThirtyMin = []
posThirtyMin = []

ThreeThirty = []
neg330 = []
pos330 = []

df_daily = pd.DataFrame(columns=['Date', '9:30am', '10am', '3:30pm', '3:55pm'])

for i in tqdm(range(len(data))):
    if data.loc[i, 'Time'] == datetime.time(9, 30):
        df_daily.loc[i, 'Date'] = data.loc[i, 'Date'].date()
        df_daily.loc[i, '9:30am'] = data.loc[i, 'Open']

# Reset index
df_daily = df_daily.reset_index(drop=True)

for i in tqdm(range(len(data))):
    if data.loc[i, 'Time'] == datetime.time(10, 0):
        # Add to 10am column for the date
        if data.loc[i, 'Date'].date() in df_daily['Date'].values:
            idx = df_daily.index[df_daily['Date'] == data.loc[i, 'Date'].date()]
            df_daily.loc[idx, '10am'] = data.loc[i, 'Open']

for i in tqdm(range(len(data))):
    if data.loc[i, 'Time'] == datetime.time(15, 30):
        # Add to 3:30pm column for the date
        if data.loc[i, 'Date'].date() in df_daily['Date'].values:
            idx = df_daily.index[df_daily['Date'] == data.loc[i, 'Date'].date()]
            df_daily.loc[idx, '3:30pm'] = data.loc[i, 'Open']

for i in tqdm(range(len(data))):
    if data.loc[i, 'Time'] == datetime.time(15, 55):
        # Add to 3:55pm column for the date
        if data.loc[i, 'Date'].date() in df_daily['Date'].values:
            idx = df_daily.index[df_daily['Date'] == data.loc[i, 'Date'].date()]
            df_daily.loc[idx, '3:55pm'] = data.loc[i, 'Open']
              
df_daily = df_daily.dropna()

# Add 'NextOpen' column
df_daily['NextOpen'] = df_daily['9:30am'].shift(-1)


df_daily = df_daily.assign(First30 = 1 - (df_daily['9:30am'] / df_daily['10am']))
df_daily = df_daily.assign(FinalBit = 1 - (df_daily['9:30am'] / df_daily['3:30pm']))
df_daily = df_daily.assign(Overnight = 1 - (df_daily['3:55pm'] / df_daily['NextOpen']))

df_daily = df_daily.dropna()

check_for_stationarity(df_daily['First30'])
check_for_stationarity(df_daily['FinalBit'])
check_for_stationarity(df_daily['Overnight'])


# Get mean of First30
countPos = len(df_daily['First30'])
avg = round((sum(df_daily['First30']) / len(df_daily['First30'])) * 100, 2)
print(f"10:00 Average: {avg} (length {countPos})")

# Get mean of FinalBit
countPos = len(df_daily['FinalBit'])
avg = round((sum(df_daily['FinalBit']) / len(df_daily['FinalBit'])) * 100, 2)
print(f"3:30 Average: {avg} (length {countPos})")

# Get mean of Overnight
countPos = len(df_daily['Overnight'])
avg = round((sum(df_daily['Overnight']) / len(df_daily['Overnight'])) * 100, 2)
print(f"Overnight Average: {avg} (length {countPos})")

# Save the data to an Excel file
file = 'new_test_data_processed.xlsx'
df_daily.to_excel(file, index=False)


        
    



if 1 == 0:
    for i in tqdm(range(len(data))):
        if data.loc[i, 'Time'] == datetime.time(9, 30):
            data.loc[i, 'OpeningBell'] = data.loc[i, 'Open']
        else:
            data.loc[i, 'OpeningBell'] = data.loc[i-1, 'OpeningBell']

        if data.loc[i, 'Time'] == datetime.time(10, 0):
            data.loc[i, 'Ten'] = 1 - (data.loc[i-1, 'OpeningBell'] / data.loc[i, 'Open'])
            thirtyMin.append(data.loc[i, 'Ten'])
            if data.loc[i, 'Ten'] < 0:
                negThirtyMin.append(data.loc[i, 'Ten'])
            else:
                posThirtyMin.append(data.loc[i, 'Ten'])

        if data.loc[i, 'Time'] == datetime.time(15, 30):
            data.loc[i, 'ThreeThirty'] = 1 - (data.loc[i-1, 'OpeningBell'] / data.loc[i, 'Open'])
            ThreeThirty.append(data.loc[i, 'ThreeThirty'])
            if data.loc[i, 'ThreeThirty'] < 0:
                neg330.append(data.loc[i, 'ThreeThirty'])
            else:
                pos330.append(data.loc[i, 'ThreeThirty'])

if 1 == 0:
    # Average the thirty minute data
    countPos = len(thirtyMin)
    avg = round((sum(thirtyMin) / len(thirtyMin)) * 100, 2)
    print(f"10:00 Average: {avg} (length {countPos})")
    thirtyMin = pd.Series(thirtyMin)
    print("thirtyMin series?", thirtyMin)
    thirtyMin.name = '10:00 Data'
    check_for_stationarity(thirtyMin)

    # Average the negative thirty minute data
    countNeg = len(negThirtyMin)
    avgNeg = round((sum(negThirtyMin) / len(negThirtyMin)) * 100, 2)
    print(f"10:00 Average Negative: {avgNeg} (length {countNeg})")
    negThirtyMin = pd.Series(negThirtyMin)
    negThirtyMin.name = '10:00 Negative Data'
    check_for_stationarity(negThirtyMin)

    # Average the positive thirty minute data
    countPos = len(posThirtyMin)
    avgPos = round((sum(posThirtyMin) / len(posThirtyMin)) * 100, 2)
    print(f"10:00 Average Positive: {avgPos} (length {countPos})")
    posThirtyMin = pd.Series(posThirtyMin)
    posThirtyMin.name = '10:00 Positive Data'
    check_for_stationarity(posThirtyMin)

    # Average the 3:30 data
    countPos = len(ThreeThirty)
    avg = round((sum(ThreeThirty) / len(ThreeThirty)) * 100, 2)
    print(f"3:30 Average: {avg} (length {countPos})")
    ThreeThirty = pd.Series(ThreeThirty)
    ThreeThirty.name = '3:30 Data'
    check_for_stationarity(ThreeThirty)

    # Average the negative 3:30 data
    countNeg = len(neg330)
    avgNeg = round((sum(neg330) / len(neg330)) * 100, 2)
    print(f"3:30 Average Negative: {avgNeg} (length {countNeg})")
    neg330 = pd.Series(neg330)
    neg330.name = '3:30 Negative Data'
    check_for_stationarity(neg330)

    # Average the positive 3:30 data
    countPos = len(pos330)
    avgPos = round((sum(pos330) / len(pos330)) * 100, 2)
    print(f"3:30 Average Positive: {avgPos} (length {countPos})")
    pos330 = pd.Series(pos330)
    pos330.name = '3:30 Positive Data'
    check_for_stationarity(pos330)

    # Save the data to an Excel file
    file = 'new_test_data_processed.xlsx'
    data.to_excel(file, index=False)

    # Plot this shit
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(thirtyMin)
    plt.show()

