import yfinance as yf
import matplotlib.pyplot as plt
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

import os
os.makedirs("data",exist_ok=True)
data.to_csv("data/prices.csv")

data.index, data.index.size


type(data)

data.describe()


data.head()

tuple(data.columns)



data.index


set([k[0] for k in data.columns])
set([k[1] for k in data.columns])
### Data is a multiIndex dataFrame... data[i,j] i= "close", "high","low", "open", "Volume"
# and j = which ticker


## plot open
plt.figure()
ax=plt.subplot(111)
ax.set_title("Open")
for i in tickers:
    oc = data[("Open",i)].copy().dropna(how="all")
    x,y = oc.index.to_numpy(), oc.to_numpy()
    ax.plot(x,y, label=i)
ax.legend()


## plot everything
for f in set([k[0] for k in data.columns]):
    plt.figure()
    ax=plt.subplot(111)
    ax.set_title(f)
    for i in tickers:
        oc = data[(f,i)].copy().dropna(how="all")
        x,y = oc.index.to_numpy(), oc.to_numpy()
        ax.plot(x,y, label=i)
    ax.legend()
    os.makedirs("figs",exist_ok=True)
    plt.savefig("figs/inspect_{}.png".format(f))



### what is this volume about???

data["Volume"]

vol = data["Volume"]                      # shape: dates × tickers [web:95]
# sum volume per year
vol_yearly = vol.resample("Y").sum()      # one row per year, one column per ticker
vol_yearly.plot(kind="bar", subplots=True, layout=(5, 2), figsize=(10, 12), sharex=True)
plt.savefig("figs/inspect_volume_yearly_subplots.png")
