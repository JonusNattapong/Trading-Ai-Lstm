"""
Test script for FRED gold data
"""

import requests
import pandas as pd
from io import StringIO

def test_fred_gold():
    print("Testing FRED gold data...")

    try:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {
            'id': 'GOLDAMGBD228NLBM',
            'cosd': '2020-01-01',
            'coed': '2024-01-01'
        }

        response = requests.get(fred_url, params=params)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
            return True
        else:
            print(f"Failed to get data: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def test_coingecko_btc():
    print("\nTesting CoinGecko BTC data...")

    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        start_ts = int(pd.to_datetime('2020-01-01').timestamp())
        end_ts = int(pd.to_datetime('2024-01-01').timestamp())

        params = {
            'vs_currency': 'usd',
            'from': start_ts,
            'to': end_ts
        }

        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if 'prices' in data:
                print(f"Number of price points: {len(data['prices'])}")
                print(f"First price point: {data['prices'][0]}")
                return True
        else:
            print(f"Failed to get data: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_fred_gold()
    test_coingecko_btc()