import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import cvxpy as cp
import pandas as pd

tickers = ["ROKU","DOCU","SNAP","ETSY","TWLO","NET","PINS","UBER","XYZ","COIN"]   ## Careful! there was a typo / unadpated --> XYZ
start = "2018-01-01"
end   = "2026-01-01"


# check this https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html#yfinance.download
data = yf.download(
    tickers=tickers,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,   # use adjusted prices, this is some sort of standard adjustmenet accounting for "splits" and "distributions"
    progress=True ##to see how it goes downloading
)





def optimize_portfolio(mu, sigma, q):
    """
    this function optimizs a quadracticc convex program
    mu expected risk (vector length N = number of stocks)
    sigma covariance matrix (NxN)
    mu:: np.array
    sigma:: np.array

    q:: float is the risk tolerance

    returns optimal weights and objective value
    """
    # Convert to numpy if needed
    if hasattr(mu, 'to_numpy'):
        mu_np = mu.to_numpy()
    else:
        mu_np = np.array(mu)

    if hasattr(sigma, 'to_numpy'):
        sigma_np = sigma.to_numpy()
    else:
        sigma_np = np.array(sigma)

    # FIX: Replace NaN values with 0 (can occur with insufficient data)
    sigma_np = np.nan_to_num(sigma_np, nan=0.0)

    # FIX: Ensure covariance matrix is symmetric (numerical precision issue)
    sigma_np = (sigma_np + sigma_np.T) / 2

    w = cp.Variable(len(mu_np))
    # risk and return terms
    risk = cp.quad_form(w, sigma_np)          # w^T Sigma w
    ret  = mu_np @ w                          # mu^T w

    objective = cp.Minimize(risk - q*ret)
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    # Handle solver failure: return equal weights as fallback
    if w.value is None:
        n = len(mu_np)
        return np.ones(n) / n, np.nan

    return w.value, prob.value




def metrics(mu,sigma,w):
    """
    Given portfolio weights w, this estimate the expected return mu^T w and volatility sqrt{w^T sigma w}
    mu :: np.array
    sigma :: np.array
    w :: np.array
    """
    rtrn = mu.dot(w)
    volatility = np.sqrt(w.dot(sigma.dot(w)))
    return rtrn, volatility



def get_mu_sigma(db, time_window=1, start_date=None, end_date=None):
    """
    Estimate return and risk out of value of stocks at "Open"
    The return is defined as the ratio r_t = P_t / P_{t-1} (price at end over price at start).
    It's standard to take log(r_t).

    For the time-window, we jump over #time_window number of days.
    This computes log(p_{1}, p_{tw}, p_{2*tw} ... )
    then .diff() computes log(p_{1} - log p_{tw}, log p_{tw} - log p_{2*tw}, log p_{2*tw} - log p_{3*tw} ... )

    We then compute mean and covariance matrix of this

    Parameters:
    -----------
    db : pd.DataFrame
        DataFrame with MultiIndex columns (Open, Close, etc.) and DatetimeIndex
    time_window : int
        Number of days to skip (1 = daily, 7 = weekly, 30 = monthly, etc.)
    start_date : str or pd.Timestamp, optional
        Start date for filtering data (inclusive)
    end_date : str or pd.Timestamp, optional
        End date for filtering data (inclusive)
    Returns:
    --------
    mu : pd.Series
        Mean log returns for each ticker
    sigma : pd.DataFrame
        Covariance matrix of log returns


    Examples:
    mu, sigma = get_mu_sigma(data, time_window=1, start_date='2021-01-01', end_date='2023-12-31')

    """
    open_px = db['Open']

    # Filter by date range if provided
    if start_date is not None:
        open_px = open_px.loc[pd.to_datetime(start_date):]
    if end_date is not None:
        open_px = open_px.loc[:pd.to_datetime(end_date)]

    # Apply time window slicing and compute log returns
    log_rtns = np.log(open_px)[::time_window].diff().dropna()

    return log_rtns.mean(), log_rtns.cov()


def rolling_rebalance(
    data,
    lookback_months=3,
    invest_months=1,
    q_selected=1.0,
    start_date="2021-01-01",
    end_date="2025-01-01"
):
    """
    Rolls a training (lookback) window forward by invest_months at a time.
    For each step:
    - optimize with last 'lookback_months'
    - 'invest' (simulate returns with fixed weights) for next 'invest_months'
    - repeat, growing window by invest_months at a time
    """
    from pandas.tseries.offsets import DateOffset

    cur_train_start = pd.to_datetime(start_date)
    cur_train_end = cur_train_start + DateOffset(months=lookback_months)
    cur_invest_end = cur_train_end + DateOffset(months=invest_months)
    final_end = pd.to_datetime(end_date)

    rebalance_dates = []
    all_weights = []
    all_returns = []
    all_vols = []

    while cur_invest_end <= final_end:
        # Optimize/fit on [cur_train_start, cur_train_end)
        mu_reb, sigma_reb = get_mu_sigma(
            data,
            time_window=1,
            start_date=cur_train_start,
            end_date=cur_train_end
        )
        w_reb = optimize_portfolio(mu_reb, sigma_reb, q_selected)[0]
        # Invest on [cur_train_end, cur_invest_end)
        mu_test, sigma_test = get_mu_sigma(data,time_window=1, start_date=cur_train_end,end_date=cur_invest_end)
        ret_test, vol_test = metrics(mu_test.to_numpy(), sigma_test.to_numpy(), w_reb)

        rebalance_dates.append(cur_train_end)
        all_weights.append(w_reb)
        all_returns.append(ret_test)
        all_vols.append(vol_test)

        # Move window forward by invest_months
        cur_train_start += DateOffset(months=invest_months)
        cur_train_end += DateOffset(months=invest_months)
        cur_invest_end += DateOffset(months=invest_months)

    return (
        np.array(rebalance_dates),
        np.array(all_weights),
        np.array(all_returns),
        np.array(all_vols)
    )


tolerances = np.logspace(-3,2,num=50)


idx_tol = 10
rebalance_dates, all_weights, all_returns, all_vols = rolling_rebalance(
    data, lookback_months=3, invest_months=1, q_selected=tolerances[idx_tol],
    start_date="2021-01-01", end_date="2025-01-01"
)
rebalance_dates

 ### The idea here is to take a rolling basis in which we "observe" behaviour for 3 months, "deploy that", and then update the optimal weights with a time-moving window to estimate the mu's and sigma's. but there's a bug, so let's wrap up here! There's more to explore, but i'm tired now :)
