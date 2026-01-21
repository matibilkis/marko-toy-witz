import yfinance as yf

tickers = ["ROKU","DOCU","SNAP","ETSY","TWLO","NET","PINS","UBER","SQ","COIN"]
start = "2018-01-01"
end   = "2026-01-01"

data = yf.download(
    tickers=tickers,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,   # use adjusted prices
    progress=False
)

# daily adjusted close prices (typical input for Markowitz)
prices = data["Adj Close"]

