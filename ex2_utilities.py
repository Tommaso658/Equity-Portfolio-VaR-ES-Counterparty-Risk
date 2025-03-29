"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 2: Corporate Bond Portfolio
"""

from typing import Union
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd
import datetime as dt
from ex1_utilities import (
    year_frac_act_x,
    business_date_offset,
    get_discount_factor_by_zero_rates_linear_interp,
    date_series,
)
from ex1_utilities import (get_discount_factor_by_zero_rates_linear_interp)
from scipy.optimize import fsolve


def defaultable_bond_dirty_price_from_intensity(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """

    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    

    # !!! COMPLETE THE FUNCTION !!!
    semestral_dates = date_series (ref_date, expiry, coupon_freq) # All coupon dates and the ref date
    annual_dates = date_series (ref_date, expiry, 1)
    deltas = np.array ([year_frac_act_x([ref_date], [annual_dates [i]], 6) for i in range (len (annual_dates) )])
    discount_factors = np.array ([get_discount_factor_by_zero_rates_linear_interp(ref_date, semestral_dates)])
    if isinstance(intensity, float) :
        survival_probs = np. exp(-intensity * deltas)
    else:
        survival_probs = np. exp(-intensity.values * deltas [1:])
        survival_probs = np. concatenate(([1.0], survival_probs) )

    defaultable_discounts = [discount_factors[i] * survival_probs [int(np.floor(i/2))+1] for i in range(len(deltas))]

    year_frac = np.array([year_frac_act_x([semestral_dates[i]], [semestral_dates [i+1]],6) for i in range (len(semestral_dates))])

    coupon_component = coupon_rate * sum(year_frac[i] * defaultable_discounts[i] for i in range(len(defaultable_discounts)))
    principal_component = defaultable_discounts [-1]

    recovery_component = recovery_rate * sum(discount_factors [2*i - 1] * (survival_probs[i-1] - survival_probs[i]) for i in range(1, len(survival_probs)))
    dirty_price = notional * (coupon_component + principal_component + recovery_component)
    
    return dirty_price
    """
     # 1) Schedulazione delle date di pagamento delle cedole
    semestral_dates = date_series(ref_date, expiry, coupon_freq)
    # Esempio: [t0=ref_date, t1, t2, ..., tN=expiry]

    # 2) Costruiamo la "griglia annuale" per calcolare i deltas 
    annual_dates = date_series(ref_date, expiry, 1)
    # Esempio: [t0=ref_date, t_anno1, t_anno2, ..., t_annoM=expiry]

    # 3) Calcoliamo i deltas = year fraction da ref_date a ciascuna annual_date
    deltas = np.array([
        year_frac_act_x([ref_date], [annual_dates[i]], 6)[0]
        for i in range(len(annual_dates))
    ])
    # Ad es. deltas = [0, 1.0, 2.0, ...] se l'expiry copre tot anni

    # 4) Interpoliamo i discount factor di mercato per ciascuna data di coupon
    #    (NOTA: qui assumiamo discount_factors.index contenga le date 
    #           e .values i discount factor corrispondenti)
    discount_factors_coupons = np.array([
        get_discount_factor_by_zero_rates_linear_interp(
            ref_date,
            d_coupon, 
            discount_factors.index,    # curve date
            discount_factors.values    # curve discount factors
        )
        for d_coupon in semestral_dates
    ])

    # 5) Calcoliamo le survival probabilities
    if isinstance(intensity, float):
        # Intensità costante: survival_prob(t) = exp(-λ * t)
        survival_probs = np.exp(-intensity * deltas)
        # Esempio: se deltas = [0, 1, 2], survival_probs = [1, e^-λ, e^-2λ]
    else:
        # Intensità piecewise: 
        # (Qui manteniamo la tua logica come bozza -> 
        #  se la Series ha un "valore" per i diversi deltas, 
        #  potresti dover gestire un loop, ma lasciamo l'idea base)
        survival_probs = np.exp(-intensity.values * deltas[1:])
        survival_probs = np.concatenate(([1.0], survival_probs))

    # 6) Creiamo i "defaultable discounts" moltiplicando 
    #    discount_factors_coupons[i] * survival_probs[...]
    #    L’indice per survival_probs parte da 1 ogni 2 passaggi (come da codice utente),
    #    ma occhio a non andare fuori indice!
    defaultable_discounts = []
    for i in range(len(deltas)):
        idx_surv = int(np.floor(i / 2)) + 1
        if idx_surv < len(survival_probs):
            val = discount_factors_coupons[i] * survival_probs[idx_surv]
        else:
            # Se l'indice di survival_probs va fuori range,
            # usiamo l'ultimo valore disponibile (oppure 0, da definire).
            val = discount_factors_coupons[i] * survival_probs[-1]
        defaultable_discounts.append(val)

    # 7) Calcoliamo la year fraction fra date di coupon consecutive
    #    Attenzione al range: se ho N date in semestral_dates, 
    #    ci sono N-1 intervalli.
    year_frac = np.array([
        year_frac_act_x([semestral_dates[i]], [semestral_dates[i+1]], 6)[0]
        for i in range(len(semestral_dates) - 1)
    ])

    # 8) Calcoliamo la parte di cedola (coupon_component)
    #    NB: defaultable_discounts ha la stessa lunghezza di 'deltas', 
    #    mentre year_frac ha lunghezza "numero di periodi". 
    #    Di solito l'ultimo discount serve per il rimborso, 
    #    quindi ci aspettiamo un allineamento simile al tuo codice.
    coupon_component = 0.0
    for i in range(len(year_frac)):
        coupon_component += coupon_rate * year_frac[i] * defaultable_discounts[i]

    # 9) Rimborso del notional (principal_component) 
    #    di solito all'ultima data:
    principal_component = defaultable_discounts[-1]

    # 10) Calcolo componente di recovery 
    #    (come nel tuo codice: 
    #     sum(...) su i in range(1, len(survival_probs)), 
    #     e usi discount_factors_coupons[2*i -1], 
    #     se esiste)
    recovery_component = 0.0
    for i in range(1, len(survival_probs)):
        idx_df = 2*i - 1
        if 0 <= idx_df < len(defaultable_discounts):
            # differenza di survival su i e i-1
            recovery_component += defaultable_discounts[idx_df] * (
                survival_probs[i-1] - survival_probs[i]
            )
    recovery_component *= recovery_rate

    # 11) Prezzo sporco finale
    dirty_price = notional * (coupon_component + principal_component + recovery_component)
    return dirty_price

def bond_cash_flows(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    notional: float = 1.0,
) -> pd.Series:
    """
    Calculate the cash flows of a bond.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments per years.
    notional (float): Notional amount.

    Returns:
        pd.Series: Bond cash flows.
    """

    # Payment dates
    cash_flows_dates = []
    payment_dt = expiry
    counter = 1
    while payment_dt > ref_date:
        cash_flows_dates.append(payment_dt)

        payment_dt = business_date_offset(
            expiry, month_offset=(-12 * counter) // coupon_freq
        )
        counter += 1

    cash_flows_dates.sort()
    # Coupon payments
    cash_flows = pd.Series(
        data=[notional * coupon_rate ] * year_frac_act_x([ref_date]+cash_flows_dates[:-1], cash_flows_dates,6),
        index=cash_flows_dates,
    )

    # Notional payment
    cash_flows[expiry] += notional

    return cash_flows



def defaultable_bond_dirty_price_from_z_spread2(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    z_spread: float,
    dates: pd.Series,
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # !!! COMPLETE THE FUNCTION !!!

    cf_schedule_yf = date_series(ref_date, expiry, coupon_freq)
    # print("cf_schedule_yf", cf_schedule_yf)
    cf_schedule = cf_schedule_yf[1:]
    # print("cf_schedule,", cf_schedule)

    discount_factors_cf = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, cf_schedule_yf[i], dates, discount_factors) for i in
        range(1, len(cf_schedule_yf))]
    # print("discounts", discount_factors_cf)
    coupon_rate_sem = np.sqrt(1 + coupon_rate) - 1
    coupons = (coupon_rate_sem * notional) * np.ones(len(cf_schedule))
    # print("coupons", coupons)

    coupons = np.array(coupons)

    yf = np.array([year_frac_act_x([ref_date], [cf_schedule_yf[i]], 6) for i in range(1, len(cf_schedule_yf))])

    yf_bond = np.array(
        [year_frac_act_x([cf_schedule_yf[i - 1]], [cf_schedule_yf[i]], 6) for i in range(1, len(cf_schedule_yf))])

    res = np.sum(coupons * yf_bond * discount_factors_cf * np.exp(-z_spread * yf)) + notional * discount_factors_cf[
        -1] * np.exp(-z_spread * yf[-1])

    return res


def defaultable_bond_dirty_price_from_z_spread(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    z_spread: float,
    dates: pd.Series,
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # !!! COMPLETE THE FUNCTION !!!
    cash_flows = bond_cash_flows(
        ref_date,
        expiry,
        coupon_rate,coupon_freq,notional)
    days_diff = [(cash_flows.index[i] - ref_date).days for i in range(len(cash_flows))]

    discounts_at_cash_flows = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, date, dates, discount_factors)[0]
        for date in cash_flows.index.tolist()]

    complete_dates = [ref_date] + list(cash_flows.index)



    res = np.sum(cash_flows.values * discounts_at_cash_flows * np.exp(-z_spread*np.array(days_diff)))

    return res


def constant_intensity_to_be_solved(cash_flows: pd.Series,
                                    ref_date:Union[dt.date, pd.Timestamp],
                                    discounts_at_cash_flows: list[float],
                                    notional:float,
                                    dirty_price:float,
                                    recovery_rate:float,
                                    intensity:float) -> float:



    # Convert ref_date to a pandas Timestamp if it's a date object
    ref_date = pd.to_datetime(ref_date)

    # Convert the cash flows index to a series of days from the reference date
    days_diff = [(cash_flows.index[i] - ref_date).days for i in range(len(cash_flows))]
    # First sum: Vectorized operation using element-wise multiplication
    first_sum = np.sum(np.array(cash_flows.values) * np.array(discounts_at_cash_flows) * np.exp(-intensity * days_diff))

    # Complete dates for second sum calculation
    complete_dates = [ref_date] + list(cash_flows.index)

    days_diff_complete = [(complete_dates[i] - ref_date).days for i in range(len(complete_dates))]
    # Second sum: Vectorized calculation using the difference of exponential terms
    second_sum = np.sum(np.array(discounts_at_cash_flows) * (
                np.exp(-intensity * np.array(days_diff_complete[:-1])) - np.exp(
            -intensity * np.array(days_diff_complete[1:]))))



    # Calculate and return the final result
    return dirty_price -first_sum -recovery_rate * second_sum


def constant_intensity_from_bond (
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    dates: list[dt.datetime],
    discount_factors: list[float],
    dirty_price:float,
    recovery_rate:float,
    end: float,
    notional: float = 1.0,

)->float:
    #we compute the cash flow and their dates for the input bond
    cash_flows = bond_cash_flows(ref_date,expiry,coupon_rate,coupon_freq)
    cash_flows = cash_flows[end:]

    #we compute the discounts at the relevant dates for cash flows
    discounts_at_cash_flows = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, date, dates, discount_factors)[0]
        for date in cash_flows.index.tolist()]

    #we use fsolve to find lambda
    def to_be_minimized(intensity):
        return constant_intensity_to_be_solved(cash_flows,ref_date, discounts_at_cash_flows,notional, dirty_price,recovery_rate,intensity)

    return fsolve(to_be_minimized, 0.000001)

def piece_wise_constant_intensity_from_bonds(ref_date: Union[dt.date, pd.Timestamp],
    expiry: list[Union[dt.date, pd.Timestamp]],
    coupon_rate: list[float],
    coupon_freq: list[int],
    dates: list[dt.datetime],
    discount_factors: list[float],
    dirty_price: list[float],
    recovery_rate: float,
    notional: float = 1.0,
) ->list[float]:

  num_bond = len(expiry)
  dirty_price_modifier = 0
  intensity_vector = np.zeros(num_bond)
  computed_intensities = 0
  start = 0
  end = 0
  for i in range(0,num_bond):
   dirty_price_modifier = 0
   cash_flows = bond_cash_flows(ref_date, expiry[i], coupon_rate[i], coupon_freq[i])
   discounts_at_cash_flows = [
       get_discount_factor_by_zero_rates_linear_interp(ref_date, date, dates, discount_factors)[0]
       for date in cash_flows.index.tolist()]
   days_diff = np.array([(cash_flows.index[i] - ref_date).days for i in range(len(cash_flows))])

   for j in range(1,computed_intensities+1):
     max_index = np.sum(np.array(cash_flows.index <expiry[j-1]))
     print("max index:",max_index)
     end = max_index
     dirty_price_modifier += sum(cash_flows.values[start:end]*discounts_at_cash_flows[start:end] \
     *np.exp(-intensity_vector[j-1]*days_diff[start:end]))
     days_diff_completo =[0]+days_diff
     dirty_price_modifier += recovery_rate*sum(discounts_at_cash_flows[start:end]*(np.exp(-intensity_vector[j-1]*days_diff_completo[start:end-1])
                                                                                   -np.exp(-intensity_vector[j-1]*days_diff_completo[start+1:end])))
     start = max_index

   intensity_vector[i] = constant_intensity_from_bond(ref_date,
   expiry[i],
   coupon_rate[i],
   coupon_freq[i],
   dates,
   discount_factors,
   dirty_price[i]-dirty_price_modifier,
   recovery_rate,end
   )
   print(dirty_price[i]-dirty_price_modifier)
   computed_intensities +=1

  return intensity_vector
