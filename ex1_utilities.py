"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

# Importing necessary standard libraries and modules
import datetime
from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from scipy.stats import norm

# Importing custom functions from other modules: bootstrap for discount curve generation,
# and yearfrac for year fraction calculations.
from bootstrap import bootstrap
from utilities import yearfrac

# Importing types for type annotations
from typing import Iterable, Union, List, Tuple


# Define an Enum for swaption types: Receiver and Payer.
class SwapType(Enum):
    """
    Types of swaptions.
    """
    RECEIVER = "receiver"
    PAYER = "payer"


def year_frac_act_x(start_dates: list[dt.datetime], end_dates: list[dt.datetime], flag: int):
    """
    Compute the year fraction between two dates using the ACT/x convention.

    Parameters:
        start_dates (list[dt.datetime]): List containing one or more start dates.
        end_dates (list[dt.datetime]): List containing one or more end dates.
        flag (int): Indicator for day count convention:
                    2 -> ACT/360,
                    3 -> ACT/365,
                    6 -> ACT/360 with 30/360 adjustment.
    
    Returns:
        float: Year fraction between the two dates.
    """
    # If a single start date is provided, compute the difference for each end date.
    if len(start_dates) == 1:
        actual_days = np.array([(end - start_dates[0]).days for end in end_dates])
    else:
        # Otherwise, compute the difference for each corresponding pair.
        actual_days = np.array([(end - start).days for start, end in zip(start_dates, end_dates)])

    # ACT/360 convention: use 360 days in a year.
    if flag == 2:
        return actual_days / 360

    # ACT/365 convention: use 365 days in a year.
    if flag == 3:
        return actual_days / 365

    # 30/360 convention (flag==6): adjust day counts to a maximum of 30 per month.
    if flag == 6:
        # Limit day values to 30 for both start and end dates.
        d1 = np.minimum(30, np.array([date.day for date in start_dates]))
        d2 = np.minimum(30, np.array([date.day for date in end_dates]))
        if len(start_dates) == 1:
            year_diff = np.array([end.year - start_dates[0].year for end in end_dates])
            month_diff = np.array([end.month - start_dates[0].month for end in end_dates])
        else:
            year_diff = np.array([end.year - start.year for start, end in zip(start_dates, end_dates)])
            month_diff = np.array([end.month - start.month for start, end in zip(start_dates, end_dates)])
        # Calculate the year fraction under the 30/360 method.
        return (360 * year_diff + 30 * month_diff + (d2 - d1)) / 360


def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex]): List of year fractions or dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of zero rates.
    """
    # 'effDates' and 'effDf' represent the effective dates and discount factors.
    effDates, effDf = dates, discount_factors
    if isinstance(effDates, pd.DatetimeIndex):
        # If dates are given as DatetimeIndex, compute year fractions between consecutive dates.
        effDates = [
            year_frac_act_x(effDates[i - 1], effDates[i], 3)
            for i in range(1, len(effDates))
        ]
        effDf = discount_factors[1:]
    
    # Compute zero rates using the continuous compounding formula.
    return -np.log(np.array(effDf)) / np.array(effDates)


def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear
    interpolation.

    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which the discount factor is interpolated.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """
    # Check if the dates list and discount factors have the same length.
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")

    # Compute the year fractions from the reference date to each date (excluding the reference date itself).
    year_fractions = year_frac_act_x([reference_date], dates[1:], 3)
    # Compute the zero rates from the discount factors.
    zero_rates = from_discount_factors_to_zero_rates(year_fractions, discount_factors[1:])
    # Compute the year fraction for the interpolation date.
    inter_year_frac = year_frac_act_x([reference_date], [interp_date], 3)
    # Linearly interpolate the zero rate.
    rate = np.interp(inter_year_frac, year_fractions, zero_rates)
    # Return the discount factor corresponding to the interpolated rate.
    return np.exp(-inter_year_frac * rate)


def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): Reference date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Closest following business date to base_date after applying the offset.
    """
    # Calculate the total number of months by adding the offset.
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Try to construct the adjusted date with the same day. If invalid, use the last valid day of the month.
    day = base_date.day
    try:
        adjusted_date = base_date.replace(year=year, month=month, day=day) + dt.timedelta(days=day_offset)
    except ValueError:
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(year=year, month=month, day=last_day_of_month) + dt.timedelta(days=day_offset)

    # If the adjusted date falls on a weekend, move to the next business day.
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date


def date_series(
    t0: Union[dt.date, pd.Timestamp], t1: Union[dt.date, pd.Timestamp], freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Return a list of dates from t0 to t1 inclusive with frequency 'freq', where freq is specified as
    the number of dates per year.
    """
    dates = [t0]
    # Continuously append dates using business_date_offset until reaching or surpassing t1.
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    
    # Ensure that the series does not exceed t1.
    if dates[-1] > t1:
        dates.pop()
    # Include t1 in the series if not already present.
    if dates[-1] != t1:
        dates.append(t1)

    return dates


def swaption_price_calculator(
    dates: list[dt.datetime],
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Valuation date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying swap expiry date.
        sigma_black (float): Swaption implied volatility.
        freq (int): Frequency (number of times per year) for fixed leg payments.
        discount_factors (pd.Series): Discount factors for computing present values.
        swaption_type (SwapType): Type of swaption (Receiver or Payer), default is Receiver.
        compute_delta (bool): Flag to indicate whether to compute and return delta along with price.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price, and if compute_delta is True, also returns delta.
    """
    # Calculate the fixed leg payment schedule from expiry to underlying expiry.
    fixed_leg_schedule = date_series(expiry, underlying_expiry, freq)
    # Compute the bond price value (BPV) as the sum of the discounted year fractions.
    bpv = sum([
        year_frac_act_x([fixed_leg_schedule[i - 1]], [fixed_leg_schedule[i]], 6) *
        get_discount_factor_by_zero_rates_linear_interp(ref_date, fixed_leg_schedule[i], dates, discount_factors)
        for i in range(1, len(fixed_leg_schedule))
    ])
    # Time to expiry in years (using ACT/365).
    T = (expiry - ref_date).days / 365.0

    # Calculate d1 and d2 for the Black formula.
    d1 = (np.log(S0 / strike) + 0.5 * sigma_black ** 2 * T) / (sigma_black * np.sqrt(T))
    d2 = d1 - sigma_black * np.sqrt(T)

    # Depending on the swaption type, calculate price and delta.
    if swaption_type == SwapType.RECEIVER:
        price = bpv * (-S0 * norm.cdf(-d1) + strike * norm.cdf(-d2))
        delta = -bpv * norm.cdf(-d1)
    else:
        price = bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        delta = bpv * norm.cdf(d1)
    
    # Return both price and delta if requested.
    if compute_delta:
        return [price, delta]
    else:
        return price


def swap_par_rate(
    dates: List[dt.datetime],
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
) -> float:
    """
    Given a fixed leg payment schedule and the discount factors, return the swap par rate. If a
    forward start date is provided, a forward swap rate is returned.

    Parameters:
        dates (List[dt.datetime]): All dates corresponding to the discount factors.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        fwd_start_date (dt.datetime | None): Forward start date; if provided, the par rate is computed from this date.

    Returns:
        float: Swap par rate.
    """
    # Calculate the discount factor at the forward start date or use 1 if not provided.
    discount_factor_t0 = 1 if fwd_start_date is None else get_discount_factor_by_zero_rates_linear_interp(
        dates[0],
        fwd_start_date,
        dates,
        discount_factors,
    )

    # Calculate the bond price value (BPV) for the fixed leg payments.
    bpv = sum([
        year_frac_act_x([fixed_leg_schedule[i-1]], [fixed_leg_schedule[i]], 6) *
        get_discount_factor_by_zero_rates_linear_interp(dates[0], fixed_leg_schedule[i], dates, discount_factors)
        for i in range(1, len(fixed_leg_schedule))
    ])

    # Get the discount factor corresponding to the final payment date.
    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        dates[0],
        fixed_leg_schedule[-1],
        dates,
        discount_factors,
    )

    # Return the swap par rate that sets the NPV to zero.
    return (discount_factor_t0 - discount_factor_tN) / bpv


def swap_mtm(
    dates,
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule, and the discount factors, return the swap mark-to-market (MtM).

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default is 'payer'.

    Returns:
        float: Swap mark-to-market.
    """
    # Convert discount factors into a pandas Series.
    discount_factors = pd.Series(discount_factors)
    # Compute the BPV of the fixed leg.
    bpv = sum([
        year_frac_act_x([fixed_leg_schedule[i - 1]], [fixed_leg_schedule[i]], 6) *
        get_discount_factor_by_zero_rates_linear_interp(dates[0], fixed_leg_schedule[i], dates, discount_factors)
        for i in range(1, len(fixed_leg_schedule))
    ])
    # Get the discount factor corresponding to the final payment date.
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        dates[0],
        fixed_leg_schedule[-1],
        dates,
        discount_factors.values,
    )
    # The floating leg value is 1 minus the discount factor at the final payment.
    float_leg = 1.0 - P_term
    # The fixed leg value is the swap rate multiplied by the BPV.
    fixed_leg = swap_rate * bpv

    # Determine the multiplier based on swap type.
    if swap_type == SwapType.RECEIVER:
        multiplier = -1
    elif swap_type == SwapType.PAYER:
        multiplier = 1
    else:
        raise ValueError("Unknown swap type.")

    # Return the MtM as the difference between the floating leg and fixed leg, adjusted by the multiplier.
    return multiplier * (float_leg - fixed_leg)


def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: List[float],
    dates: List[dt.date],
) -> float:
    """
    Given the specifics of an interest rate swap (IRS), return its rate sensitivity calculated as
    the duration of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (List[float]): Discount factors.
        dates (List[dt.date]): All relevant dates for discounting.

    Returns:
        float: Proxy duration of the swap (measure of rate sensitivity).
    """
    # Initialize an array to store interpolated discount factors for each payment date.
    discounts_interp = np.zeros(len(fixed_leg_payment_dates))
    for i in range(len(fixed_leg_payment_dates)):
        discounts_interp[i] = get_discount_factor_by_zero_rates_linear_interp(
            ref_date,
            fixed_leg_payment_dates[i],
            dates,
            discount_factors
        )
    # Exclude the first discount factor (typically 1 for the valuation date).
    discounts_interp = discounts_interp[1:]
    # Compute the year fraction for each period between consecutive payment dates using the 30/360 convention.
    yf = year_frac_act_x(fixed_leg_payment_dates[:-1], fixed_leg_payment_dates[1:], 6)
    # Calculate the coupon payments for each period.
    c_i = yf * swap_rate
    # Add the notional repayment to the final coupon.
    c_i[-1] += 1
    # Calculate the price of an equivalent bond (should be close to 1 at initiation).
    IB_bond_t0 = np.sum(c_i * discounts_interp)
    # Compute the weighted average time (Macaulay Duration) for the cash flows.
    times = year_frac_act_x([ref_date], fixed_leg_payment_dates, 6)
    times = times[1:]
    Sium = np.sum(c_i * discounts_interp * times)
    DU = Sium / IB_bond_t0

    return DU


def year_frac_act_xNEW(t1: dt.datetime, t2: dt.datetime, x: int) -> float:
    """
    Compute the year fraction between two dates using the ACT/x convention.

    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.
        x (int): Number of days in a year.

    Returns:
        float: Year fraction between the two dates.
    """

    return (t2-t1).days / x
