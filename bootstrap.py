import numpy as np
import datetime as dt
import math as mt
import pandas as pd
from pandas.tseries.offsets import BDay
from scipy.interpolate import CubicSpline

from utilities import adjust_to_next_business_day
from utilities import zeroRates
from utilities import yearfrac


def bootstrap(datesSet: dict[str, list[float]], ratesSet: dict[str, list[float]]) -> tuple[list[float], list[float]]:
    ## building a complete set o swaps dates
    newSwapsDates = []
    for year in range(2025, 2074): #creates a set of swaps dates untill 2073 excluding weekends
        newSwapsDates.append(adjust_to_next_business_day(dt.date(year, 2, 2)).strftime("%Y-%m-%d"))
    ##building a complete set of swap rates
    newSwapsRates = array_zeros = np.zeros(len(newSwapsDates))
    noRateIndexes = []
    ratesIndexes = []

    #we create the array to host the new swaps rates with suitable nan values
    #and we get indexes for later use
    for i, newSwapsDate in enumerate(newSwapsDates):
        if newSwapsDate in datesSet["swaps"]:
            # Get the index of the date in dates1
            idx = datesSet["swaps"].index(newSwapsDate)
            newSwapsRates[i] = ratesSet["swaps"][idx]
            ratesIndexes.append(i)
        else: noRateIndexes.append(i)
    #we interpolate on nan values using cubic spline
    ratesBuoni = [newSwapsRates[i] for i in ratesIndexes]
    dateBuone = [newSwapsDates[i] for i in ratesIndexes]
    #print(dateBuone)
    swapsRatesSpline = CubicSpline(yearfrac("2023-02-03",dateBuone,3), ratesBuoni)

    for i in noRateIndexes:
     newSwapsRates[i] = swapsRatesSpline(yearfrac("2023-02-03",newSwapsDates[i],3))

    #defining much needed new vectors
    datesSet["swaps"] = newSwapsDates
    ratesSet["swaps"] = newSwapsRates
    discounts = []
    dates = []

    # Loop over the range 1 to 4 (using Python 0-based indexing)
    for j in range(4):
        # Extracting the date for the current deposit
        dates.append(datesSet['depos'][j])

        # Calculate the average rate
        Lrate = (ratesSet['depos'][j])

        # Compute the discount factor
        discount = 1 / (1 + yearfrac(datesSet['settlement'],datesSet['depos'][j],2) * Lrate)
        discounts.append(discount[0])

    gap = 4
    for j in range(0,6):
     Lrate = ratesSet["futures"][j]
     if yearfrac(datesSet["futures_settlement"][j],dates[j + gap - 1],3) > 0: # if the value date of the future is before the last known discount's date

      zeroRatesVector = zeroRates(datesSet["settlement"]+dates[gap+j-2:gap+j], [1]+discounts[gap+j-2:gap+j])
      zeroRateInterp =  np.interp(yearfrac(datesSet["settlement"],datesSet["futures_settlement"][j],3), yearfrac(datesSet["settlement"], datesSet["settlement"]+dates[gap + j - 2:gap + j],3), zeroRatesVector)
      discountInterp = mt.exp(-( yearfrac(datesSet["settlement"],datesSet["futures_settlement"][j],3)) * zeroRateInterp )
      discountAtValueDate = discountInterp


     else: #here we do flat extrapolation or nothing

      yKnownt1 = zeroRates([datesSet["settlement"][0], dates[j-1+gap]], [1,discounts[gap+j-1]])[1]
      discountAtValueDate = mt.exp(-(yearfrac(datesSet["settlement"][0],datesSet["futures_settlement"][j],3))* yKnownt1)

     #compute the Lrate
     time = yearfrac(datesSet["futures_settlement"][j], datesSet["futures_expiry"][j], 2)
     forwardDiscount = 1 / (1 + time * Lrate)

     discount = discountAtValueDate * forwardDiscount
     discounts.append(discount[0])
     dates.append(datesSet["futures_expiry"][j])

    gap = 7  # Adjust this according to the dataset
    zeroRateInterp = zeroRates(datesSet["settlement"]+[dates[gap - 1]]+[dates[gap]], [1, discounts[gap - 1], discounts[gap]])

    # Interpolating
    date_interp = adjust_to_next_business_day(dt.date(2024, 2, 2)).strftime("%Y-%m-%d")
    yInterp = np.interp(yearfrac(datesSet["settlement"][0], date_interp,3), yearfrac(datesSet["settlement"][0], [datesSet["settlement"][0],dates[gap - 1],dates[gap]],3), zeroRateInterp)

    # Going back to discount
    swapDiscountOne = np.exp(-yearfrac(datesSet["settlement"],date_interp,3) * yInterp)
    # discountInterp = mt.exp(-(datesSet["futures_settlement"] - datesSet["settlement"]) * zerorRateInterp / 365)

    # Defining new vectors for the loop
    swapsDatesForLoop = np.concatenate((datesSet["settlement"], [date_interp], datesSet["swaps"]))
    swapsDiscount =np.zeros(len(newSwapsDates)+1)
    swapsDiscount[0] = swapDiscountOne
    gap = 11

    for j in range(len(newSwapsRates)):
        discountSum = sum( swapsDiscount[i] * yearfrac(swapsDatesForLoop[i], swapsDatesForLoop[i + 1], 6) for i in range(j+1))
        swapsDiscount[j + 1] = (1 - ratesSet["swaps"][j] * discountSum) / (1 + ratesSet["swaps"][j] * yearfrac(swapsDatesForLoop[j + 1], swapsDatesForLoop[j + 2], 6))
        discounts.append(swapsDiscount[j + 1])
        dates.append(datesSet["swaps"][j])
    return dates, discounts

