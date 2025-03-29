import numpy as np
import datetime as dt
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def yearfrac(start_dates, end_dates, flag):
  if isinstance(start_dates, str):
    start_dates = [start_dates]

  if isinstance(end_dates, str):
   end_dates = [end_dates]

  if len(start_dates) > 1 and len(start_dates) != len(end_dates):
       raise ValueError("Error: start_dates and end_dates must have the same size when start_dates has more than one element.")

  start_dates = np.array([dt.datetime.strptime(date, "%Y-%m-%d") for date in start_dates])
  end_dates = np.array([dt.datetime.strptime(date, "%Y-%m-%d") for date in end_dates])

  if len(start_dates) == 1:
    actual_days = np.array([(end - start_dates[0]).days for end in end_dates])
  else:
   actual_days = np.array([(end - start).days for start, end in zip(start_dates, end_dates)])

  if flag == 2:
    return actual_days / 360

  if flag == 3:
    return actual_days / 365

  if flag == 6:
    d1 = np.minimum(30, np.array([date.day for date in start_dates]))
    d2 = np.minimum(30, np.array([date.day for date in end_dates]))
    if len(start_dates) == 1:
        year_diff = np.array([end.year  - start_dates[0].year for end in end_dates])
        month_diff = np.array([end.month - start_dates[0].month for end in end_dates])
    else:
        year_diff = np.array([end.year - start.year for start, end in zip(start_dates, end_dates)])
        month_diff = np.array([end.month - start.month for start, end in zip(start_dates, end_dates)])

  return (360 * year_diff + 30 * month_diff + (d2 - d1)) / 360



def zeroRates(dates, discounts):

    yf = yearfrac(dates[0], dates, 3)
    discounts = np.array(discounts)
    zRates = np.zeros(len(discounts))
    zRates[1:] = -np.log(discounts[1:]) / yf[1:]


    return zRates




#Read the file datesSet.mat
def read_mat_file_dates(file_path):
    # Carica il file .mat
    data = sio.loadmat(file_path)
    variabile = data['datesSet']
    # Array per dates of the settlement
    settlement = variabile[0][0][0][0]
    #Array per dates of the depos
    depos = np.array([])
    for i in range(len(variabile[0][0][1])):
        depos = np.append(depos, np.array(variabile[0][0][1][i]))
    # Array per dates of the futures (settlement and expiry)
    futures_settlement = np.array([])
    for i in range(len(variabile[0][0][2])):
        futures_settlement = np.append(futures_settlement, np.array(variabile[0][0][2][i][0]))
    futures_scadenza = np.array([])
    for i in range(len(variabile[0][0][2])):
        futures_scadenza = np.append(futures_scadenza, np.array(variabile[0][0][2][i][1]))
    # Array per dates of the swaps
    swaps = np.array([])
    for i in range(len(variabile[0][0][3])):
        swaps = np.append(swaps, np.array(variabile[0][0][3][i]))
    #Create dict
    dates = {
        "settlement": settlement,
        "depos": depos,
        "futures_settlement": futures_settlement,
        "futures_scadenza": futures_scadenza,
        "swaps": swaps,
    }
    return dates


#Read the file ratesSet.mat
def read_mat_file_rates(file_path):
    # Carica il file .mat
    data = sio.loadmat(file_path)

    variabile = data['ratesSet']
    # Array per rates of the depos (bid and ask)
    depos_bid = np.array([])
    for i in range(len(variabile[0][0][0])):
        depos_bid = np.append(depos_bid, np.array(variabile[0][0][0][i][0]))
    depos_ask = np.array([])
    for i in range(len(variabile[0][0][0])):
        depos_ask = np.append(depos_ask, np.array(variabile[0][0][0][i][1]))
    depos = (depos_bid + depos_ask) / 2   #Average
    # Array per rates of the futures (bid and ask)
    futures_bid = np.array([])
    for i in range(len(variabile[0][0][1])):
        futures_bid = np.append(futures_bid, np.array(variabile[0][0][1][i][0]))
    futures_ask = np.array([])
    for i in range(len(variabile[0][0][1])):
        futures_ask = np.append(futures_ask, np.array(variabile[0][0][1][i][1]))
    futures = (futures_bid + futures_ask) / 2  # Average
    # Array per rates of the futures (bid and ask)
    swaps_bid = np.array([])
    for i in range(len(variabile[0][0][2])):
        swaps_bid = np.append(swaps_bid, np.array(variabile[0][0][2][i][0]))
    swaps_ask = np.array([])
    for i in range(len(variabile[0][0][2])):
        swaps_ask = np.append(swaps_ask, np.array(variabile[0][0][2][i][1]))
    swaps = (swaps_bid + swaps_ask) / 2  # Average
    rates = {
        "depos_bid": depos_bid,
        "depos_ask": depos_ask,
        "depos": depos,
        "futures_bid": futures_bid,
        "futures_ask": futures_ask,
        "futures": futures,
        "swaps_bid": swaps_bid,
        "swaps_ask": swaps_ask,
        "swaps": swaps,
    }
    return rates




def adjust_to_next_business_day(date):

    if isinstance(date, str):
        date = pd.to_datetime(date)

    if date.weekday() == 5:  # Saturday
        date += pd.Timedelta(days=2)

    elif date.weekday() == 6:  # Sunday
        date += pd.Timedelta(days=1)

    return date

def convert_date_to_string(dateArray: list[float]):
    inizio = 738919
    dateArrayNew = []
    for item in dateArray:
        # MATLAB datenum calculation (item should be a numeric value)
        matlab_datenum = int(item) - inizio

        # MATLAB base date (1st January 0000)
        base_date = dt.datetime(2023, 2, 2)

        # Convert MATLAB datenum to Python datetime
        converted_date = base_date + dt.timedelta(days=matlab_datenum)

        # Format the Python datetime to the desired string format
        formatted_date = converted_date.strftime("%Y-%m-%d")

        # Update the datesSet["swaps"] list with the formatted date
        dateArrayNew.append(formatted_date)
    return dateArrayNew

def plot_bootstrap_results(dates, discounts):
    # Convert date strings to datetime objects
    years = [dt.datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Compute zero rates from discount factors (assuming zeroRates is a defined function)
    zero_rates = zeroRates(dates, discounts) * 100  # Multiplying by 100 for percentage representation

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Formatting the x-axis to show dates properly on the primary axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))  # Show only the year for ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format to display only the year
    plt.gcf().autofmt_xdate()

    # Plotting Discount Factors on the primary axis (ax1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Discount Factor', color='tab:blue')
    ax1.plot(years, discounts, '-o', color='tab:blue', linewidth=1.5, markersize=4, label='Discount Factors')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(axis='y', linestyle='--')  # Grid on Y-axis only
    ax1.set_ylim(0.3, 1)  # Set zero rates axis range from 0 to 350

    # Plotting Zero Rates on the secondary axis (ax2)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Zero Rate (%)', color='tab:red')
    ax2.plot(years, zero_rates * 100, '*-', color='tab:red', linewidth=1.5, markersize=4, label='Zero Rates')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 350)  # Set zero rates axis range from 0 to 350



    # Title and layout
    plt.title('Reconstruction of Discount Factors and Zero Rates')
    fig.tight_layout()

    # Show the plot
    plt.show()
    # Show the plot
    plt.show()