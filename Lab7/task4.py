import datetime
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn import covariance, cluster

company_symbols_map = {
    "TOT": "Total", "XOM": "Exxon", "CVX": "Chevron", "COP": "ConocoPhillips",
    "VLO": "Valero Energy", "MSFT": "Microsoft", "IBM": "IBM", "TWX": "Time Warner",
    "CMCSA": "Comcast", "CVC": "Cablevision", "YHOO": "Yahoo", "DELL": "Dell",
    "HPQ": "HP", "AMZN": "Amazon", "TM": "Toyota", "CAJ": "Canon", 
    "MTU": "Mitsubishi", "SNE": "Sony", "F": "Ford", "HMC": "Honda",
    "NAV": "Navistar", "NOC": "Northrop Grumman", "BA": "Boeing", 
    "KO": "Coca Cola", "MMM": "3M", "MCD": "Mc Donalds", "PEP": "Pepsi",
    "MDLZ": "Kraft Foods", "K": "Kellogg", "UN": "Unilever", "MAR": "Marriott",
    "PG": "Procter Gamble", "CL": "Colgate-Palmolive", "GE": "General Electrics",
    "WFC": "Wells Fargo", "JPM": "JPMorgan Chase", "AIG": "AIG", 
    "AXP": "American express", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "AAPL": "Apple", "SAP": "SAP", "CSCO": "Cisco", "TXN": "Texas instruments",
    "XRX": "Xerox", "LMT": "Lookheed Martin", "WMT": "Wal-Mart", 
    "WBA": "Walgreen", "HD": "Home Depot", "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer", "SNY": "Sanofi-Aventis", "NVS": "Novartis", 
    "KMB": "Kimberly-Clark", "R": "Ryder", "GD": "General Dynamics", 
    "RTN": "Raytheon", "CVS": "CVS", "CAT": "Caterpillar", "DD": "DuPont de Nemours"
}

symbols, names = np.array(list(company_symbols_map.items())).T

start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

quotes = []
valid_symbols = []

for symbol in symbols:
    try:
        stock_data = yf.Ticker(symbol).history(start=start_date, end=end_date)
        if not stock_data.empty:
            quotes.append(stock_data)
            valid_symbols.append(symbol)
        else:
            print(f"{symbol}: No data found")
    except Exception as e:
        print(f"{symbol}: Error fetching data - {e}")

if not quotes:
    print("No valid data retrieved.")
    exit()

min_length = min(len(quote) for quote in quotes)
quotes = [quote.iloc[:min_length] for quote in quotes]

opening_quotes = np.array([quote['Open'].values for quote in quotes], dtype=np.float64)
closing_quotes = np.array([quote['Close'].values for quote in quotes], dtype=np.float64)
quotes_diff = closing_quotes - opening_quotes

X = quotes_diff.copy().T
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

print('\nClustering of stocks based on difference in opening and closing quotes:\n')
for i in range(num_labels + 1):
    cluster_symbols = np.array(valid_symbols)[labels == i]
    cluster_names = np.array(names)[np.isin(symbols, cluster_symbols)]
    print(f"Cluster {i + 1} ==> {', '.join(cluster_names)}")
