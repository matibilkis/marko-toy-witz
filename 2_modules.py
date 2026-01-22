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




def optimize_portfolio(mu,sigma, q):
    """
    this function optimizs a quadracticc convex program
    mu expected risk (vector length N = number of stocks)
    sigma covariance matrix (NxN)
    mu:: np.array
    sigma:: np.array

    q:: float is the risk tolerance

    returns optimal weights and objective value
    """
    w = cp.Variable(len(mu))
    # risk and return terms
    risk = cp.quad_form(w, sigma)          # w^T Sigma w
    ret  = mu.to_numpy() @ w                          # mu^T w

    objective = cp.Minimize(risk - q*ret)
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
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


mu, sigma = get_mu_sigma(data, time_window=1, start_date='2021-01-01', end_date='2024-01-01')
returns_volatility, weights =[], []

tolerances = np.logspace(-3,2,num=50)
for ind,q in enumerate(tqdm(tolerances)):
    w_opt = optimize_portfolio(mu, sigma, q)[0]
    weights.append(w_opt)
    returns_volatility.append(metrics(mu.to_numpy(),sigma.to_numpy(),w_opt))

returns, volatility = np.squeeze(returns_volatility).T
# Plot efficient frontier

plt.figure()
ax=plt.subplot(111)
ax.scatter(volatility**2, returns , color="blue",alpha=0.7)
ax.set_xlabel("RISK")
ax.set_ylabel("RETURNS")
plt.savefig("figs/frontier_1d_21-24.png")


#### Now we want to understand how this behaves in the future...


## Let's assume i choose a risk tolerance of tolerances[10] ---> optimal portfolio (trained w/ data from 21' ---> '24 ) is weights[10]

weights[10]

## Now, if i have this, but i TEST with the '24 --> '25
mu_test, sigma_test = get_mu_sigma(data, time_window=1, start_date='2024-01-01', end_date='2025-01-01')
#

ret_test, vol_test = metrics(mu_test.to_numpy(),sigma_test.to_numpy(),w_opt)


ret_test, vol_test
np.squeeze(returns_volatility)[10]



mu_test, sigma_test = get_mu_sigma(data, time_window=1, start_date='2024-01-01', end_date='2025-01-01')
returns_volatility_test, weights_test =[], []
rv_opt_test, w_opt_test = [], []
for ind,q in enumerate(tqdm(tolerances)):
    w_opt_test.append(optimize_portfolio(mu_test, sigma_test, q)[0])
    rv_opt_test.append(metrics(mu_test.to_numpy(),sigma_test.to_numpy(),w_opt_test[ind]))
    returns_volatility_test.append(metrics(mu_test.to_numpy(),sigma_test.to_numpy(),weights[ind]))

returns_test, volatility_test = np.squeeze(returns_volatility_test).T
returns_test_opt, volatility_test_opt = np.squeeze(rv_opt_test).T



plt.figure()
ax=plt.subplot(111)
ax.set_title("Out-of-sample validation \n We compute optimal weights for 21-24, \nthen test return-risk w/ those weights on 24-25")
ax.scatter(volatility**2, returns , color="blue",alpha=0.7, label="21-24 optimal")
ax.scatter(volatility_test**2, returns_test, color="red",alpha=0.7,label="24-25 test")
ax.scatter(volatility_test_opt**2, returns_test_opt, color="green",alpha=0.7,label="24-25 optimized")

ax.set_xlabel("RISK")
ax.set_ylabel("RETURNS")
plt.legend()
plt.savefig("figs/frontier_1d_21-24_comparison_24-25_optimizd.png")

###This plot tells that the optimal weights for the training period definitely changed on the testing.
### Blue ---> optimized portfolio 21-24
### Red ----> using the optimized portfolio 21-24 during 2025   (note this shouldn't be convex, we are not lying on the frontier)
### Green --> the optimal portfolio at 2025


# The Cumulative Wealth Curve: how much you gain in 2025 using this portfolio, against how much you could have gained...

# Daily log returns during test period
log_returns_25 = np.log(data['Open']['2024-01-01':'2025-01-01']).diff().dropna()   # \log p^k_t/p^k_{t-1}
log_returns_25_portfolio_old = np.einsum('qk,ik->qi', np.squeeze(weights), log_returns_25.to_numpy())    # \sum_k \log p^k_t/p^k_{t-1} * w^k_q
log_returns_25_portfolio_optimized = np.einsum('qk,ik->qi', np.squeeze(w_opt_test), log_returns_25.to_numpy())    # \sum_k \log p^k_t/p^k_{t-1} * w^k_q





# Cumulative ---> how much you have when departing from 2024 (w/ return = 1 from day =1)
cumulative_q = np.exp(log_returns_25_portfolio_old.cumsum(axis=1))        # \sum p^k*w^k
cumulative_q_optimized = np.exp(log_returns_25_portfolio_optimized.cumsum(axis=1))        # \sum p^k*w^k


ind=10
ax=plt.subplot(111)
ax.set_title("Cumulative Wealth Curve in 2025\nBehaviour of portfolio optimized only 23-24\nVersus a <<perfect>> 2025\n(!)Not much of a difference risk tolerance...\nInteresting: what's the figure of merit! Wealth vs. mean-variance")
ax.plot(range(cumulative_q.shape[1]), cumulative_q[ind], label="optimized on '23-24'")
ax.plot(range(cumulative_q.shape[1]), cumulative_q_optimized[ind], label="optimized on '25")
plt.ylabel("cumulative return")
plt.xlabel("day of 2025")
plt.legend()
plt.savefig("figs/cumulative_wealth_curve_comparison_train_test_25.png")

#### THE MESSAGE HERE IS... be careful because the figure of merit is not money straightfoward!
