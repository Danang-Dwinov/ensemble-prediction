import pandas as pd

#for rolling window feature
roll = 10

files = {
    "QQQ": ["data/qqq.csv"],
    "SPY": ["data/spy.csv"],
    "WMT": ["data/wmt.csv"],
    "DXY": ["data/dxy.csv"],
    "GLD": ["data/gld.csv"],
    "NVDA": ["data/nvda.csv"]
}

def load_stock(files):
    df = pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if "Volume" in df.columns:
        df["Volume"] = (
            df["Volume"]
            .astype(str)
            .str.replace(",", "")
            .astype(int)
        )

    return df
    
def rsi(prices, period=10):
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))
    
def make_dataset(start, end):

   stocks = {
        name: load_stock(path).loc[start:end]
        for name, path in files.items()
   }

   df = stocks["NVDA"]

   df["Return"] = df["Close"].pct_change()
   
   df["Target"] = (
        df["Return"].shift(-1) > 0
   ).astype(int)
   
   df["Aktual"] = (
       df["Return"] > 0
   ).astype(int)
   
   df["RSI"] = rsi(df["Close"], roll)
   
   df["MA"] = df["Return"].rolling(roll).mean()
   df["Vol"] = df["Return"].rolling(roll).std()
   
   df["VA"] = df["Volume"].rolling(roll).mean()
   df["RV"] = df["Volume"] / df["VA"]
   
   df= df.fillna(0)
   df= df.loc[~df.index.duplicated(keep='first')]
   df= df.loc[~df.index.duplicated(keep='last')]
   
   symbols = ["QQQ", "SPY", "WMT", "DXY", "GLD"]

   for i, symbol in enumerate(symbols, 1):

       stocks[symbol][f"X{i}"] = (
           stocks[symbol]["Close"].pct_change()
       )
        
   for symbol in ["QQQ", "SPY", "WMT"]:
        stocks[symbol]["VA"] = (
        stocks[symbol]["Volume"]
        .rolling(roll)
        .mean()
        )
        
        stocks[symbol]["RV"] = (
           stocks[symbol]["Volume"] /
           stocks[symbol]["VA"])

   features = [
       stocks[s]["X"+str(i)]
       for i, s in enumerate(
           ["QQQ","SPY","WMT","DXY","GLD"],
           start=1
       )
   ]
   
   
   df_x = pd.concat(
       [
           df["Target"],
           df["Return"],
           *features,
           df["RSI"],
           df["VA"],
           df["RV"],
           df["Vol"],
           df["MA"],
       ],
       axis=1, join="inner"
   )
   
   return df_x.fillna(0)
   
df_x = make_dataset(
    "2024-06-24",
    "2026-07-07")
