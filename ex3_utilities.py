"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 3: Equity Portfolio VaR/ES and Counterparty Risk
"""
import pdb
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Tuple, Union
from numpy.typing import NDArray


class OptionType(Enum):
    """
    Types of options.
    """

    CALL = "call"
    PUT = "put"


def black_scholes_option_pricer(
    S: float,
    K: float,
    ttm: float,
    r: float,
    sigma: float,
    d: float,
    option_type: OptionType = OptionType.PUT,
    return_delta_gamma: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """
    Return the price (and possibly delta and gamma) of an option according to the Black-Scholes
    formula.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        ttm (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Implied volatility.
        d (float): Dividend yield.
        option_type (OptionType, {'put', 'call'}): Option type, default to put.
        return_delta_gamma (bool): If True the option delta and gamma are returned.

    Returns:
        Union[float, Tuple[float, float, float]]: Option price (and possibly delta and gamma).
    """

    d1 = (np.log(S / K) + (r - d + sigma**2 / 2) * ttm) / (sigma * np.sqrt(ttm)) # probabile errore nel -
    d2 = d1 - sigma * np.sqrt(ttm)

    if option_type == OptionType.CALL:
        if return_delta_gamma:
            return (
                S * np.exp(-d * ttm) * norm.cdf(d1)
                - K * np.exp(-r * ttm) * norm.cdf(d2),
                np.exp(-d * ttm) * norm.cdf(d1),
                np.exp(-d * ttm) * norm.pdf(d1) / (S * sigma * ttm ** (0.5)),
            )
        else:
            return S * np.exp(-d * ttm) * norm.cdf(d1) - K * np.exp(
                -r * ttm
            ) * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        if return_delta_gamma:
            return (
                K * np.exp(-r * ttm) * norm.cdf(-d2)
                - S * np.exp(-d * ttm) * norm.cdf(-d1),
                -np.exp(-d * ttm) * norm.cdf(-d1),
                np.exp(-d * ttm) * norm.pdf(d1) / (S * sigma * ttm ** (0.5)),
            ) # gamma di put e call sono uguali
        else:
            return K * np.exp(-r * ttm) * norm.cdf(-d2) - S * np.exp(
                -d * ttm
            ) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")


def principal_component_analysis(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Given a matrix, returns the eigenvalues vector and the eigenvectors matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sorting from greatest to lowest the eigenvalues and the eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def gaussian_var_es(
    mu: pd.Series,
    sigma: pd.DataFrame,
    alpha: float,
    weights: pd.Series,
    ptf_notional: float = 1e6,
    delta: float = 1,
) -> Tuple[float, float]:
    """
    Return VaR and ES computed via Gaussian parametric approach according to the following formulas:
        VaR_{alpha} = delta * mu + sqrt(delta) * sigma * VaR^{std}_{alpha}, where
            VaR^{std}_{alpha} = N^{-1}(alpha) and N is the standard normal cumulative distribution
            function.
        ES_{alpha} = delta * mu + sqrt(delta) * sigma * ES^{std}_{alpha}, where
            ES^{std}_{alpha} = phi(N^{-1}(alpha)) / (1 - alpha) and phi is the standard normal
            probability density function.

    Parameters:
        mu (pd.Series): Series of mean returns.
        sigma (pd.DataFrame): Returns covariance matrix.
        alpha (float): Confidence level.
        weights (pd.Series): Portfolio weights (considered unchanged).
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to 1 i.e. no adjusment is performed.

    Returns:
        Tuple[float, float]: VaR and ES.
    """

    # !!! IMPLEMENT THE FUNCTION !!!
    #computation of mean and variance of portfolio loss(linearized portfolio assumption)
    expect_l=-np.dot(weights,mu)
    var_l=np.dot(weights,np.dot(sigma,weights))

    #var and es computation
    Var_alpha=norm.ppf(alpha) #VaR standardizzato
    VaR=delta*expect_l+np.sqrt(delta)*np.sqrt(var_l)*Var_alpha 

    es_alpha=norm.pdf(Var_alpha)/(1-alpha) #ES standardizzato
    ES=delta*expect_l+np.sqrt(delta)*np.sqrt(var_l)*es_alpha

    return ptf_notional*VaR,ES*ptf_notional

    
"""

"""


def hs_var_es(
    returns: pd.DataFrame,
    alpha: float,
    weights: pd.Series,
    ptf_notional: float = 1e6,
    delta: float = 1,
    lambda_: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return VaR and ES computed via possibly weighted historical simulation:

    Parameters:
        returns (pd.DataFrame): Returns.
        alpha (float): Confidence level.
        weights (pd.Series): Portfolio weights (considered unchanged).
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to 1 i.e. no adjusment is performed.
        lambda_ (Optional[float]): Decay factor for weighted historical simulation, default to None
            i.e. standard historical simulation is performed.

    Returns:
        Tuple[float, float]: VaR and ES.
    """

    # !!! IMPLEMENT THE FUNCTION !!!
    
    # Calcolo dei ritorni ponderati (moltiplicazione dei ritorni per i pesi)
    weighted_losses = -returns.dot(weights)*ptf_notional
    total_days=len(returns.index)
    days_diff=pd.Series(range(total_days-1,-1,-1)) 
    # Se lambda_ è specificato, calcoliamo i pesi come lambda elevato alla differenza in giorni
    if lambda_ is not None:
        normalization_factor = (1 - lambda_) / (1 - lambda_**total_days)
        decay_weights = np.power(lambda_, days_diff)*normalization_factor  # Pesi decrescenti esponenziali basati su lambda e differenza giorni
        decay_weights=pd.Series(decay_weights.values,index=returns.index)
        sorted_indices = np.argsort(weighted_losses)[::-1]  # Ottieni gli indici ordinati in ordine decrescente
        sorted_losses = weighted_losses[sorted_indices] # Ordina le perdite ponderate
        sorted_weights = decay_weights[sorted_indices] # Ordina i pesi
        cumulative_weights = np.cumsum(sorted_weights) 
        var_index = np.where(cumulative_weights >= 1-alpha)[0][0]
        var = sorted_losses[var_index]
        es = np.mean(sorted_losses[:var_index+1])
    # Calcolare VaR e ES per il portafoglio
    else:
        sorted_losses = np.sort(weighted_losses) # Ordinare le perdite ponderate
        var = np.percentile(sorted_losses, (alpha) * 100) # VaR al livello di confidenza alpha
        es = np.mean(sorted_losses[sorted_losses >= var]) # ES come la media dei ritorni inferiori al VaR

    # Calcolare VaR e ES su base nominale
    var_value = var * delta  # VaR moltiplicato per il notionale e delta
    es_value = es * delta  # ES moltiplicato per il notionale e delta

    return var_value, es_value


def plausility_check(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
    ptf_notional: float = 1e6,
    delta: float = 1,
) -> float:
    """
    Perform plausibility check on a portfolio VaR estimating its order of magnitude.

    Parameters:
        returns (pd.DataFrame): Returns.
        weights (pd.Series): Portfolio weights.
        alpha (float): Confidence level.
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to one, i.e. no scaling is performed.

    Returns:
        float: Portfolio VaR order of magnitude.
    """

    sVaR = (
        -ptf_notional
        * weights
        * returns.quantile(q=[1 - alpha, alpha], axis=0).T.abs().sum(axis=1)
        / 2
    )

    return np.sqrt(delta * np.dot(sVaR, np.dot(returns.corr(), sVaR)))
