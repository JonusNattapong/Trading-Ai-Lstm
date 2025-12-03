"""
Advanced Web Scraper for Financial Data
Scrapes real-time and historical data from multiple financial websites
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FinancialWebScraper:
    """Advanced web scraper for financial data from multiple sources"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Data sources
        self.sources = {
            'gold': [
                'https://www.investing.com/commodities/gold-historical-data',
                'https://www.gold.org/goldhub/data/gold-prices'
            ],
            'bitcoin': [
                'https://coinmarketcap.com/currencies/bitcoin/historical-data/',
                'https://www.investing.com/crypto/bitcoin/btc-usd-historical-data'
            ]
        }

    def scrape_investing_com(self, symbol, start_date, end_date, asset_type='commodity'):
        """
        Scrape data from Investing.com

        Args:
            symbol: Asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            asset_type: 'commodity' or 'crypto'

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from Investing.com...")

        try:
            # URL mapping
            urls = {
                'XAUUSD': 'https://www.investing.com/commodities/gold-historical-data',
                'BTCUSD': 'https://www.investing.com/crypto/bitcoin/btc-usd-historical-data'
            }

            if symbol not in urls:
                print(f"No URL mapping for {symbol}")
                return None

            url = urls[symbol]

            # Try direct API approach first
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

            api_url = f"https://api.investing.com/api/financialdata/historical/{symbol.lower()}"
            params = {
                'start-date': start_ts,
                'end-date': end_ts,
                'time-frame': 'Daily',
                'add-missing-rows': 'false'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': url
            }

            response = self.session.get(api_url, params=params, headers=headers)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'data' in data:
                        records = []
                        for item in data['data']:
                            try:
                                records.append({
                                    'date': pd.to_datetime(item['date'], unit='ms'),
                                    'open': float(item['open']),
                                    'high': float(item['high']),
                                    'low': float(item['low']),
                                    'close': float(item['close']),
                                    'volume': float(item.get('volume', 0))
                                })
                            except (KeyError, ValueError):
                                continue

                        if records:
                            df = pd.DataFrame(records)
                            df = df.sort_values('date').reset_index(drop=True)
                            print(f"Successfully scraped {len(df)} records for {symbol} from API")
                            return df
                except json.JSONDecodeError:
                    pass

            # Fallback: try scraping the HTML page
            response = self.session.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for historical data table
            table = soup.find('table', {'class': 'historicalTbl'})
            if not table:
                # Try different class names
                table = soup.find('table', {'id': 'curr_table'})

            if table:
                rows = table.find_all('tr')
                data = []

                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        try:
                            date_str = cols[0].text.strip()
                            price_open = float(cols[1].text.strip().replace(',', ''))
                            price_high = float(cols[2].text.strip().replace(',', ''))
                            price_low = float(cols[3].text.strip().replace(',', ''))
                            price_close = float(cols[4].text.strip().replace(',', ''))
                            volume_str = cols[5].text.strip()

                            volume = 0
                            if volume_str and volume_str != '-':
                                if 'K' in volume_str:
                                    volume = float(volume_str.replace('K', '')) * 1000
                                elif 'M' in volume_str:
                                    volume = float(volume_str.replace('M', '')) * 1000000
                                else:
                                    volume = float(volume_str.replace(',', ''))

                            data.append({
                                'date': pd.to_datetime(date_str),
                                'open': price_open,
                                'high': price_high,
                                'low': price_low,
                                'close': price_close,
                                'volume': volume
                            })
                        except (ValueError, IndexError):
                            continue

                if data:
                    df = pd.DataFrame(data)
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"Successfully scraped {len(df)} records for {symbol} from table")
                    return df

            # Try to find JSON data in scripts
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and ('historicalData' in script.string or 'data' in script.string):
                    try:
                        # Look for JSON-like data
                        start = script.string.find('{')
                        end = script.string.rfind('}') + 1
                        if start != -1 and end > start:
                            json_str = script.string[start:end]
                            data_obj = json.loads(json_str)

                            if 'historicalData' in data_obj:
                                records = []
                                for item in data_obj['historicalData']:
                                    try:
                                        records.append({
                                            'date': pd.to_datetime(item['date']),
                                            'open': float(item['open']),
                                            'high': float(item['high']),
                                            'low': float(item['low']),
                                            'close': float(item['close']),
                                            'volume': float(item.get('volume', 0))
                                        })
                                    except (KeyError, ValueError):
                                        continue

                                if records:
                                    df = pd.DataFrame(records)
                                    df = df.sort_values('date').reset_index(drop=True)
                                    print(f"Successfully scraped {len(df)} records for {symbol} from script")
                                    return df
                    except (json.JSONDecodeError, ValueError):
                        continue

            print(f"No data found for {symbol} on Investing.com")
            return None

        except Exception as e:
            print(f"Error scraping Investing.com for {symbol}: {e}")
            return None

    def scrape_coinmarketcap(self, symbol, start_date, end_date):
        """
        Scrape cryptocurrency data from CoinMarketCap

        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from CoinMarketCap...")

        try:
            # Convert dates to Unix timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())

            url = f"https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=1&convertId=2781&timeStart={start_ts}&timeEnd={end_ts}"

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and 'quotes' in data['data']:
                quotes = data['data']['quotes']
                records = []

                for quote in quotes:
                    try:
                        timestamp = quote['timeOpen']
                        date = pd.to_datetime(timestamp, unit='ms')

                        records.append({
                            'date': date,
                            'open': float(quote['quote']['open']),
                            'high': float(quote['quote']['high']),
                            'low': float(quote['quote']['low']),
                            'close': float(quote['quote']['close']),
                            'volume': float(quote['quote']['volume'])
                        })
                    except (KeyError, ValueError):
                        continue

                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"Successfully scraped {len(df)} records for {symbol}")
                    return df

            print(f"No data found for {symbol}")
            return None

        except Exception as e:
            print(f"Error scraping CoinMarketCap for {symbol}: {e}")
            return None

    def scrape_alpha_vantage(self, symbol, api_key=None, outputsize='full'):
        """
        Scrape data using Alpha Vantage API

        Args:
            symbol: Stock/crypto symbol
            api_key: Alpha Vantage API key (free tier available)
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from Alpha Vantage...")

        if not api_key:
            print("Alpha Vantage API key required")
            return None

        try:
            # Determine function based on symbol type
            if symbol in ['BTCUSD', 'ETHUSD']:
                function = 'DIGITAL_CURRENCY_DAILY'
                market = 'USD'
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol.replace('USD', '')}&market={market}&apikey={api_key}&outputsize={outputsize}"
            else:
                function = 'TIME_SERIES_DAILY'
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&outputsize={outputsize}"

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            if 'Error Message' in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return None

            # Parse the response
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
            elif 'Time Series (Digital Currency Daily)' in data:
                time_series = data['Time Series (Digital Currency Daily)']
            else:
                print("Unexpected Alpha Vantage response format")
                return None

            records = []
            for date_str, values in time_series.items():
                try:
                    records.append({
                        'date': pd.to_datetime(date_str),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': float(values['5. volume'])
                    })
                except (KeyError, ValueError):
                    continue

            if records:
                df = pd.DataFrame(records)
                df = df.sort_values('date').reset_index(drop=True)
                print(f"Successfully scraped {len(df)} records for {symbol}")
                return df

            print(f"No data extracted for {symbol}")
            return None

        except Exception as e:
            print(f"Error scraping Alpha Vantage for {symbol}: {e}")
            return None

    def scrape_financial_modeling_prep(self, symbol, start_date, end_date):
        """
        Scrape data using Financial Modeling Prep API (free tier)

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from Financial Modeling Prep...")

        try:
            # Map symbols
            symbol_mapping = {
                'XAUUSD': 'GC=F',  # Gold futures
                'BTCUSD': 'BTCUSD',  # Bitcoin
                'GLD': 'GLD'  # Gold ETF
            }

            api_symbol = symbol_mapping.get(symbol, symbol)

            # Free API endpoint (no key required for basic data)
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{api_symbol}"

            params = {
                'from': start_date,
                'to': end_date
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'historical' in data:
                records = []
                for item in data['historical']:
                    try:
                        records.append({
                            'date': pd.to_datetime(item['date']),
                            'open': float(item['open']),
                            'high': float(item['high']),
                            'low': float(item['low']),
                            'close': float(item['close']),
                            'volume': float(item['volume'])
                        })
                    except (KeyError, ValueError):
                        continue

                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"Successfully scraped {len(df)} records for {symbol}")
                    return df

            print(f"No historical data for {symbol}")
            return None

        except Exception as e:
            print(f"Error scraping Financial Modeling Prep for {symbol}: {e}")
            return None

    def scrape_twelve_data(self, symbol, start_date, end_date, api_key=None):
        """
        Scrape data using Twelve Data API (free tier available)

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            api_key: Twelve Data API key (optional for free tier)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from Twelve Data...")

        try:
            # Map symbols
            symbol_mapping = {
                'XAUUSD': 'GC=F',
                'BTCUSD': 'BTC/USD',
                'GLD': 'GLD'
            }

            api_symbol = symbol_mapping.get(symbol, symbol)

            url = f"https://api.twelvedata.com/time_series"

            params = {
                'symbol': api_symbol,
                'interval': '1day',
                'start_date': start_date,
                'end_date': end_date,
                'outputsize': 5000  # Max for free tier
            }

            if api_key:
                params['apikey'] = api_key
            else:
                # Free tier (limited)
                params['apikey'] = 'demo'  # Public demo key

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'values' in data:
                records = []
                for item in data['values']:
                    try:
                        records.append({
                            'date': pd.to_datetime(item['datetime']),
                            'open': float(item['open']),
                            'high': float(item['high']),
                            'low': float(item['low']),
                            'close': float(item['close']),
                            'volume': float(item['volume'])
                        })
                    except (KeyError, ValueError):
                        continue

                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"Successfully scraped {len(df)} records for {symbol}")
                    return df

            print(f"No data for {symbol}")
            return None

        except Exception as e:
            print(f"Error scraping Twelve Data for {symbol}: {e}")
            return None

    def scrape_yahoo_finance_alternative(self, symbol, start_date, end_date):
        """
        Alternative Yahoo Finance scraper using different approach

        Args:
            symbol: Yahoo Finance symbol (e.g., 'GC=F', 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {symbol} data from Yahoo Finance alternative...")

        try:
            # Try a different approach - use a free CSV source
            # For gold, try using a different free source
            if symbol == 'GC=F':
                # Try using stooq.com for free historical data
                url = f"https://stooq.com/q/l/?s={symbol}&i=d"
                response = self.session.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for download link
                download_link = soup.find('a', {'class': 'btn btn-default'})
                if download_link and 'href' in download_link.attrs:
                    csv_url = f"https://stooq.com{download_link['href']}"
                    csv_response = self.session.get(csv_url)
                    csv_response.raise_for_status()

                    from io import StringIO
                    df = pd.read_csv(StringIO(csv_response.text))

                    if not df.empty:
                        # Rename columns
                        df = df.rename(columns={
                            'DATE': 'date',
                            'OPEN': 'open',
                            'HIGH': 'high',
                            'LOW': 'low',
                            'CLOSE': 'close',
                            'VOL': 'volume'
                        })

                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)

                        # Filter date range
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                        print(f"Successfully scraped {len(df)} records for {symbol} from stooq")
                        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # For Bitcoin, try a different approach
            if symbol == 'BTC-USD':
                # Try using a free crypto API
                url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
                start_ts = int(pd.to_datetime(start_date).timestamp())
                end_ts = int(pd.to_datetime(end_date).timestamp())

                params = {
                    'vs_currency': 'usd',
                    'from': start_ts,
                    'to': end_ts
                }

                response = self.session.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                if 'prices' in data:
                    records = []
                    prices = data['prices']
                    volumes = data.get('total_volumes', [])

                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0

                        records.append({
                            'date': pd.to_datetime(timestamp, unit='ms'),
                            'open': price,  # Approximation
                            'high': price,  # Approximation
                            'low': price,   # Approximation
                            'close': price,
                            'volume': volume
                        })

                    if records:
                        df = pd.DataFrame(records)
                        df = df.sort_values('date').reset_index(drop=True)
                        print(f"Successfully scraped {len(df)} records for {symbol} from CoinGecko")
                        return df

            # Fallback to synthetic data generation if all else fails
            print(f"No free data source available for {symbol}, will use synthetic fallback")
            return None

        except Exception as e:
            print(f"Error scraping Yahoo Finance alternative for {symbol}: {e}")
            return None

    def scrape_financial_data(self, symbol, start_date, end_date, sources=None):
        """
        Main method to scrape financial data from multiple sources

        Args:
            symbol: Trading symbol (XAUUSD, BTCUSD, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            sources: List of sources to try

        Returns:
            DataFrame with OHLCV data
        """
        if sources is None:
            # Prioritize free sources first
            sources = ['investing', 'coinmarketcap', 'yahoo_alt', 'financial_modeling_prep', 'twelve_data']

        for source in sources:
            try:
                if source == 'investing':
                    if symbol in ['XAUUSD', 'BTCUSD']:
                        data = self.scrape_investing_com(symbol, start_date, end_date)

                elif source == 'coinmarketcap':
                    if symbol == 'BTCUSD':
                        data = self.scrape_coinmarketcap('BTC', start_date, end_date)

                elif source == 'yahoo_alt':
                    # Map symbols to Yahoo format
                    yahoo_symbol = {
                        'XAUUSD': 'GC=F',
                        'BTCUSD': 'BTC-USD'
                    }.get(symbol, symbol)

                    data = self.scrape_yahoo_finance_alternative(yahoo_symbol, start_date, end_date)

                elif source == 'financial_modeling_prep':
                    data = self.scrape_financial_modeling_prep(symbol, start_date, end_date)

                elif source == 'twelve_data':
                    data = self.scrape_twelve_data(symbol, start_date, end_date)

                elif source == 'alpha_vantage':
                    # Requires API key
                    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                    if api_key:
                        data = self.scrape_alpha_vantage(symbol, api_key)

                if data is not None and not data.empty:
                    print(f"Successfully obtained data for {symbol} from {source}")
                    return data

                # Wait between requests to be respectful
                time.sleep(2)

            except Exception as e:
                print(f"Error with {source} for {symbol}: {e}")
                continue

        print(f"Failed to scrape data for {symbol} from all sources")
        return None

    def save_data(self, df, symbol, filename=None):
        """
        Save scraped data to CSV file

        Args:
            df: DataFrame to save
            symbol: Trading symbol
            filename: Optional filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.csv"

        filepath = os.path.join(self.data_dir, filename)

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

        return filepath

# Convenience functions
def scrape_gold_data(start_date, end_date):
    """Scrape gold price data"""
    scraper = FinancialWebScraper()
    data = scraper.scrape_financial_data('XAUUSD', start_date, end_date)
    return data

def scrape_bitcoin_data(start_date, end_date):
    """Scrape Bitcoin price data"""
    scraper = FinancialWebScraper()
    data = scraper.scrape_financial_data('BTCUSD', start_date, end_date)
    return data

def scrape_all_data(start_date, end_date, save_data=True):
    """Scrape data for all configured trading pairs"""
    scraper = FinancialWebScraper()
    all_data = {}

    # Import config to get trading pairs
    try:
        import config
        trading_pairs = config.TRADING_PAIRS
    except ImportError:
        # Default pairs if config not available
        trading_pairs = {
            'XAUUSD': 'GLD',
            'BTCUSD': 'BTC-USD'
        }

    for pair_name in trading_pairs.keys():
        print(f"\n{'='*50}")
        print(f"Scraping {pair_name}")
        print(f"{'='*50}")

        data = scraper.scrape_financial_data(pair_name, start_date, end_date)

        if data is not None:
            all_data[pair_name] = data

            if save_data:
                scraper.save_data(data, pair_name)

        # Be respectful to APIs
        time.sleep(3)

    return all_data

if __name__ == "__main__":
    # Test the scraper
    print("Testing Financial Web Scraper")
    print("=" * 40)

    # Test gold data
    print("\nTesting Gold (XAUUSD) scraping...")
    gold_data = scrape_gold_data('2023-01-01', '2024-01-31')
    if gold_data is not None:
        print(f"Gold data shape: {gold_data.shape}")
        print(f"Date range: {gold_data['date'].min()} to {gold_data['date'].max()}")

    # Test Bitcoin data
    print("\nTesting Bitcoin (BTCUSD) scraping...")
    btc_data = scrape_bitcoin_data('2023-01-01', '2024-01-31')
    if btc_data is not None:
        print(f"BTC data shape: {btc_data.shape}")
        print(f"Date range: {btc_data['date'].min()} to {btc_data['date'].max()}")

    print("\nScraping test completed!")