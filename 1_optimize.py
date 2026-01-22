import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import cvxpy as cp

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

data.columns
### Now the question is ...  how do we allocate the weights of a portfolio...
## *** Long-only
     ### ---> portfolio re-balance after a couple of days or months!


#### What's a portfolio ??? We need to choose the weights of each stock    w_i between 0,1  \sum_i w_i = 1
### Markowitz mean-variance ----> we want to minimize w^T \sigma w - q \mu^T w for some risk tolerance q


# Workflow
## 1 Computer expected return (for each stock) ---> this is \mu
## 2 Compute the covariance matrix \Sigma
## 3 Then run the convex quadratic prorgam


## 1 Computer expected return (for each stock) ---> this is \mu
 #### The "daily" return is defined as the ratio between the stock price P_t / P_{t-1} -1, ussually people take the log

open_px = data['Open']
log_rtns = np.log(open_px).diff().dropna()   # same as log(P_t/P_{t-1}) for all tickers   #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html
ax = log_rtns.plot(figsize=(10, 5), legend=True)
ax.set_title("Daily log-returns (Open)")
ax.set_ylabel("Log-return")
plt.savefig("figs/daily_log_retunrs.png")

### We estimate the mean value by doing \frac{1}{T}\sum_t r_t, and \hat{\Sigma} = \frac{1}{T-1} \sum_t (r_t - \hat{u}) (r_t - \hat{u})^T
### Note that this is a strong assumption, since nobody tells you the time-average is the same than the \mu ... (!) there's ussually time correlations, for sure. This is some sort of iid assumption...
mu = log_rtns.mean()
mu

sigma = log_rtns.cov()

plt.figure(figsize=(6, 5))
sns.heatmap(sigma, annot=False, fmt=".2e", cmap="coolwarm")
plt.title("Covariance matrix of log-returns")
plt.tight_layout()
plt.show()


### Let's check random costs

random_cost=[]
tolerances = np.logspace(-2,2,num=50)
for ind,q in enumerate(tqdm(tolerances)):

    ### let's compute a few random portfolios
    M = int(1e4)
    w_random = np.random.random((M,len(tickers)))
    w_random = w_random/np.sum(w_random,axis=1, keepdims=True)
    i =  np.einsum('mi,ij,mj->m',w_random,sigma.to_numpy(),w_random) - q*np.einsum('mj,j->m',w_random,mu)
    random_cost.append(i)


### We write the convex program in cvxpy. There are some different solvers SCS, ECOS, OSQP... we use SCS check this for details https://github.com/cvxgrp/scs


q = tolerances[0] ## this would be "low risk"
w = cp.Variable(len(mu))

# risk and return terms
risk = cp.quad_form(w, sigma.to_numpy())          # w^T Sigma w
ret  = mu.to_numpy() @ w                          # mu^T w

# objective: minimize risk - q * return
objective = cp.Minimize(risk - q*ret)

# constraints: fully invested, long-only
constraints = [
    cp.sum(w) == 1,
    w >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)  # or ECOS, OSQP, etc.

w_opt = w.value            # optimal weight vector (numpy array)
obj_opt = prob.value




ax=plt.subplot(111)
ax.scatter(np.arange(M),np.squeeze(random_cost)[0])
ax.axhline(obj_opt, color="red")
plt.savefig("figs/random_opt_and_QP_6Y_daily_q1e-2.png")


### Okay! then we have our "first" optimal portfolio :)
## It's comprised by weights... w_opt

w_opt
### and the return is... --->
mu.to_numpy().dot(w_opt)

### and the volatility is
np.sqrt(w_opt.dot(sigma.to_numpy().dot(w_opt)))

####
