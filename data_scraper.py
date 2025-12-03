"""
Data Scraper Module
Fetches historical price data for XAUUSD and BTCUSD using web scraping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import config
import os
import requests
import json
from financial_web_scraper import FinancialWebScraper


class DataScraper:
    """Scrape trading data from various sources"""
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.web_scraper = FinancialWebScraper()
        
    def load_huggingface_gold_data(self, start_date=None, end_date=None, interval='1d'):
        """
        Load gold price data from Hugging Face dataset
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        print("Loading gold price data from Hugging Face dataset...")
        
        try:
            # Prefer using Hugging Face Datasets if it is available (lazy import)
            # This avoids importing datasets at module load time which can fail
            # in some environments (torch DLL issues on Windows). If it's not
            # available or import fails, fall back to direct HTTP JSONL parsing.
            try:
                from datasets import load_dataset
                print("Loading HF dataset with datasets.load_dataset()...")
                # Map the requested interval to file names used in the HF dataset
                interval_map = {
                    '1d': 'XAU_1d_data.jsonl',
                    '1h': 'XAU_1h_data.jsonl',
                    '30m': 'XAU_30m_data.jsonl',
                    '15m': 'XAU_15m_data.jsonl',
                    '5m': 'XAU_5m_data.jsonl',
                    '1m': 'XAU_1m_data.jsonl',
                    '1w': 'XAU_1w_data.jsonl',
                    '1M': 'XAU_1Month_data.jsonl'
                }
                data_file = interval_map.get(interval, 'XAU_1d_data.jsonl')
                # Ask datasets to load a specific data file from the HF dataset repo
                ds = load_dataset('ZombitX64/xauusd-gold-price-historical-data-2004-2025', data_files={'train': data_file}, split='train')
                df = ds.to_pandas()
                # Some HF datasets may store Date column as 'Date' or 'date'
                if 'Date' in df.columns and 'date' not in df.columns:
                    df['date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M', errors='coerce')
                    df = df.drop(columns=['Date'])
            except Exception as exc:
                print(f"HF datasets import or load failed: {exc}. Falling back to HTTP JSONL parse.")
                # Download the specific interval file directly if datasets is not available
                interval_map = {
                    '1d': 'XAU_1d_data.jsonl',
                    '1h': 'XAU_1h_data.jsonl',
                    '30m': 'XAU_30m_data.jsonl',
                    '15m': 'XAU_15m_data.jsonl',
                    '5m': 'XAU_5m_data.jsonl',
                    '1m': 'XAU_1m_data.jsonl',
                    '1w': 'XAU_1w_data.jsonl',
                    '1M': 'XAU_1Month_data.jsonl'
                }
                chosen_file = interval_map.get(interval, 'XAU_1d_data.jsonl')
                url = f'https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/{chosen_file}'
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Failed to download data: {response.status_code}")
                    return None
                # Parse JSONL data
                data = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        record = json.loads(line)
                        data.append(record)
                df = pd.DataFrame(data)
            
            # Note: df is already set either from datasets or created above in fallback
            
            # Convert date format
            if 'Date' in df.columns and 'date' not in df.columns:
                # Try a few date formats; the HF dataset uses 'YYYY.MM.DD HH:MM'
                try:
                    df['date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M', errors='coerce')
                except Exception:
                    df['date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.drop('Date', axis=1)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure proper column order
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Filter by date range if specified
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"Successfully loaded {len(df)} records from Hugging Face dataset")
            return df
            
        except Exception as e:
            print(f"Error loading Hugging Face data: {e}")
            return None
        
    def fetch_financial_data(self, symbol, start_date, end_date=None, interval='1d', use_synthetic_fallback=True):
        """
        Fetch data using multiple sources with Hugging Face dataset as primary for XAUUSD
        
        Args:
            symbol: Trading symbol (e.g., 'XAUUSD', 'BTCUSD')
            start_date: Start date for data
            end_date: End date for data (None for today)
            interval: Data interval (1d, 1h, 15m, etc.)
            use_synthetic_fallback: Whether to generate synthetic data if scraping fails
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {symbol} data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Use Hugging Face dataset as primary source for XAUUSD
            if symbol == 'XAUUSD' or symbol == 'GLD' or symbol == 'GOLD':
                df = self.load_huggingface_gold_data(start_date, end_date, interval)
                if df is not None and not df.empty:
                    print(f"Successfully obtained {symbol} data from Hugging Face dataset")
                    return df
            
            # Fall back to web scraping for other symbols or if HF fails
            df = self.web_scraper.scrape_financial_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # Ensure we have the necessary columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Missing required columns for {symbol}")
                    if use_synthetic_fallback:
                        print(f"Generating synthetic data for {symbol}...")
                        return self.generate_synthetic_data(symbol, start_date, end_date)
                    return None
                
                df['date'] = pd.to_datetime(df['date'])
                print(f"Successfully fetched {len(df)} records for {symbol} from web scraping")
                return df
            else:
                print(f"No data found for {symbol}")
                if use_synthetic_fallback:
                    print(f"Generating synthetic data for {symbol}...")
                    return self.generate_synthetic_data(symbol, start_date, end_date)
                return None
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            if use_synthetic_fallback:
                print(f"Generating synthetic data for {symbol}...")
                return self.generate_synthetic_data(symbol, start_date, end_date)
            return None
    
    # Keep the old method for backward compatibility
    def fetch_yahoo_data(self, symbol, start_date, end_date=None, interval='1d', use_synthetic_fallback=True):
        """Backward compatibility method - now uses web scraping"""
        return self.fetch_financial_data(symbol, start_date, end_date, interval, use_synthetic_fallback)
    
    def generate_synthetic_data(self, symbol, start_date, end_date=None):
        """
        Generate synthetic OHLCV data for testing when real data is unavailable
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date (None for today)
        
        Returns:
            DataFrame with synthetic OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
        
        start_date = pd.to_datetime(start_date)
        days = (end_date - start_date).days
        
        if days <= 0:
            days = 365  # Default to 1 year
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        
        # Base prices for different assets
        if 'GC=F' in symbol or 'XAU' in symbol:
            base_price = 2000  # Gold around $2000
        elif 'BTC' in symbol:
            base_price = 50000  # Bitcoin around $50k
        else:
            base_price = 100  # Default
        
        # Generate random walk prices
        price_changes = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Generate OHLCV from prices
        high_mult = 1 + np.random.uniform(0, 0.03, len(dates))  # Up to 3% higher
        low_mult = 1 - np.random.uniform(0, 0.03, len(dates))   # Up to 3% lower
        
        # Volume based on asset type
        if 'GC=F' in symbol or 'XAU' in symbol:
            volume_base = 100000
        elif 'BTC' in symbol:
            volume_base = 1000000
        else:
            volume_base = 10000
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate OHLC
            high = close * high_mult[i]
            low = close * low_mult[i]
            open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.01))
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            volume = int(volume_base * (1 + np.random.normal(0, 0.5)))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic records for {symbol}")
        return df
    
    def fetch_all_pairs(self):
        """
        Fetch data for all configured trading pairs using web scraping
        
        Returns:
            Dictionary with pair names as keys and DataFrames as values
        """
        all_data = {}
        
        # Map config symbols to our scraper format
        symbol_mapping = {
            'XAUUSD': 'XAUUSD',  # Gold
            'BTCUSD': 'BTCUSD',  # Bitcoin
            'GLD': 'XAUUSD',     # Gold ETF -> Gold
            'BTC-USD': 'BTCUSD'  # BTC -> BTCUSD
        }
        
        for pair_name, config_symbol in config.TRADING_PAIRS.items():
            # Use the mapped symbol for scraping
            scrape_symbol = symbol_mapping.get(config_symbol, pair_name)
            
            print(f"\n{'='*50}")
            print(f"Fetching {pair_name} ({scrape_symbol})")
            print(f"{'='*50}")
            
            df = self.fetch_financial_data(
                symbol=scrape_symbol,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
                interval=config.INTERVAL
            )
            
            if df is not None:
                all_data[pair_name] = df
                # Save raw data
                self.save_data(df, pair_name, 'raw')
            
            # Be respectful to websites
            time.sleep(3)
        
        return all_data
    
    def save_data(self, df, pair_name, data_type='raw'):
        """
        Save DataFrame to CSV
        
        Args:
            df: DataFrame to save
            pair_name: Name of trading pair
            data_type: Type of data (raw, processed, etc.)
        """
        filename = f"{pair_name}_{data_type}_{config.INTERVAL}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Saved data to {filepath}")
    
    def load_data(self, pair_name, data_type='raw'):
        """
        Load DataFrame from CSV
        
        Args:
            pair_name: Name of trading pair
            data_type: Type of data (raw, processed, etc.)
        
        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = f"{pair_name}_{data_type}_{config.INTERVAL}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded data from {filepath}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None
    
    def update_data(self, pair_name):
        """
        Update existing data with new records
        
        Args:
            pair_name: Name of trading pair
        """
        existing_df = self.load_data(pair_name, 'raw')
        
        if existing_df is not None:
            last_date = existing_df['date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            print(f"Updating {pair_name} from {start_date}")
            
            symbol = config.TRADING_PAIRS[pair_name]
            new_df = self.fetch_yahoo_data(
                symbol=symbol,
                start_date=start_date,
                end_date=config.END_DATE,
                interval=config.INTERVAL
            )
            
            if new_df is not None and not new_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                
                self.save_data(combined_df, pair_name, 'raw')
                print(f"Added {len(new_df)} new records to {pair_name}")
                return combined_df
            else:
                print(f"No new data available for {pair_name}")
                return existing_df
        else:
            print(f"No existing data found. Fetching complete dataset...")
            return self.fetch_all_pairs().get(pair_name)
    
    def get_data_summary(self, df, pair_name):
        """
        Print summary statistics of the data
        
        Args:
            df: DataFrame to summarize
            pair_name: Name of trading pair
        """
        print(f"\n{'='*50}")
        print(f"Data Summary for {pair_name}")
        print(f"{'='*50}")
        print(f"Records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print("\nPrice Statistics:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())


if __name__ == "__main__":
    # Test the scraper
    scraper = DataScraper()
    
    # Fetch all pairs
    all_data = scraper.fetch_all_pairs()
    
    # Print summaries
    for pair_name, df in all_data.items():
        scraper.get_data_summary(df, pair_name)
