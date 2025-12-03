"""
Main Execution Script
Orchestrates the complete trading AI pipeline
"""

import argparse
import sys
from datetime import datetime

from data_scraper import DataScraper
from data_preprocessing import DataPreprocessor
from train import ModelTrainer
from predict import TradingPredictor
import config


def scrape_data():
    """Scrape data for all trading pairs"""
    print("\n" + "="*70)
    print("STEP 1: DATA SCRAPING")
    print("="*70)
    
    scraper = DataScraper()
    all_data = scraper.fetch_all_pairs()
    
    print(f"\nSuccessfully scraped data for {len(all_data)} pairs")
    
    for pair_name, df in all_data.items():
        scraper.get_data_summary(df, pair_name)
    
    return all_data


def update_data():
    """Update existing data with new records"""
    print("\n" + "="*70)
    print("UPDATING DATA")
    print("="*70)
    
    scraper = DataScraper()
    
    for pair_name in config.TRADING_PAIRS.keys():
        print(f"\nUpdating {pair_name}...")
        scraper.update_data(pair_name)


def train_models(pairs=None, model_types=None):
    """
    Train models for specified pairs
    
    Args:
        pairs: List of pair names (None = all pairs)
        model_types: List of model types (None = attention only)
    """
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    if pairs is None:
        pairs = list(config.TRADING_PAIRS.keys())
    
    if model_types is None:
        model_types = ['attention']
    
    results = {}
    
    for pair_name in pairs:
        for model_type in model_types:
            print(f"\n{'#'*70}")
            print(f"Training {model_type.upper()} model for {pair_name}")
            print(f"{'#'*70}\n")
            
            try:
                trainer = ModelTrainer(pair_name, model_type=model_type)
                metrics = trainer.run_full_training()
                results[f"{pair_name}_{model_type}"] = metrics
            except Exception as e:
                print(f"Error training {pair_name} with {model_type}: {str(e)}")
                results[f"{pair_name}_{model_type}"] = {"error": str(e)}
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
    
    return results


def backtest_models(pairs=None, model_types=None):
    """
    Backtest models for specified pairs
    
    Args:
        pairs: List of pair names (None = all pairs)
        model_types: List of model types (None = attention only)
    """
    print("\n" + "="*70)
    print("STEP 3: BACKTESTING")
    print("="*70)
    
    if pairs is None:
        pairs = list(config.TRADING_PAIRS.keys())
    
    if model_types is None:
        model_types = ['attention']
    
    scraper = DataScraper()
    results = {}
    
    for pair_name in pairs:
        for model_type in model_types:
            print(f"\n{'#'*70}")
            print(f"Backtesting {model_type.upper()} model for {pair_name}")
            print(f"{'#'*70}\n")
            
            try:
                # Load data
                df = scraper.load_data(pair_name, 'raw')
                
                if df is None:
                    print(f"No data available for {pair_name}")
                    continue
                
                # Create predictor and backtest
                predictor = TradingPredictor(pair_name, model_type=model_type)
                predictor.load_model()
                
                trades_df, portfolio_values, results_df = predictor.backtest_strategy(df, initial_capital=10000)
                
                # Plot results
                predictor.plot_backtest_results(trades_df, portfolio_values, results_df)
                
                # Save results
                if len(trades_df) > 0:
                    trades_path = f"{config.RESULTS_DIR}/trades_{pair_name}_{model_type}.csv"
                    trades_df.to_csv(trades_path, index=False)
                    
                    results[f"{pair_name}_{model_type}"] = {
                        "n_trades": len(trades_df),
                        "final_capital": portfolio_values[-1] if portfolio_values else 10000,
                        "return_pct": ((portfolio_values[-1] - 10000) / 10000 * 100) if portfolio_values else 0
                    }
                else:
                    results[f"{pair_name}_{model_type}"] = {"n_trades": 0, "final_capital": 10000, "return_pct": 0}
                
            except Exception as e:
                print(f"Error backtesting {pair_name} with {model_type}: {str(e)}")
                results[f"{pair_name}_{model_type}"] = {"error": str(e)}
    
    print("\n" + "="*70)
    print("BACKTESTING SUMMARY")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Total Trades: {result['n_trades']}")
            print(f"  Final Capital: ${result['final_capital']:,.2f}")
            print(f"  Return: {result['return_pct']:.2f}%")
    
    return results


def run_full_pipeline(pairs=None, model_types=None):
    """
    Run complete pipeline: scrape -> train -> backtest
    
    Args:
        pairs: List of pair names (None = all pairs)
        model_types: List of model types (None = attention only)
    """
    print("\n" + "#"*70)
    print("# LSTM TRADING AI - FULL PIPELINE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*70)
    
    start_time = datetime.now()
    
    # Step 1: Scrape data
    scrape_data()
    
    # Step 2: Train models
    train_results = train_models(pairs=pairs, model_types=model_types)
    
    # Step 3: Backtest models
    backtest_results = backtest_models(pairs=pairs, model_types=model_types)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "#"*70)
    print("# PIPELINE COMPLETED")
    print(f"# Duration: {duration}")
    print(f"# Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*70)


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(description='LSTM Trading AI - XAUUSD & BTCUSD')
    
    parser.add_argument(
        'action',
        choices=['scrape', 'update', 'train', 'backtest', 'full'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--pairs',
        nargs='+',
        choices=list(config.TRADING_PAIRS.keys()),
        help='Trading pairs to process (default: all)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['simple', 'bidirectional', 'attention', 'multi_head'],
        help='Model types to use (default: attention)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LSTM TRADING AI - XAUUSD & BTCUSD")
    print("="*70)
    print(f"Action: {args.action}")
    print(f"Pairs: {args.pairs if args.pairs else 'ALL'}")
    print(f"Models: {args.models if args.models else 'attention'}")
    print("="*70)
    
    if args.action == 'scrape':
        scrape_data()
    
    elif args.action == 'update':
        update_data()
    
    elif args.action == 'train':
        train_models(pairs=args.pairs, model_types=args.models)
    
    elif args.action == 'backtest':
        backtest_models(pairs=args.pairs, model_types=args.models)
    
    elif args.action == 'full':
        run_full_pipeline(pairs=args.pairs, model_types=args.models)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run full pipeline
        print("No arguments provided. Running full pipeline with default settings...")
        print("Usage: python main.py [scrape|update|train|backtest|full] [--pairs PAIR1 PAIR2] [--models MODEL1 MODEL2]")
        print("\nExamples:")
        print("  python main.py scrape")
        print("  python main.py train --pairs XAUUSD --models attention")
        print("  python main.py backtest --pairs BTCUSD")
        print("  python main.py full")
        print("\n" + "="*70)
        
        # Run with defaults
        run_full_pipeline()
    else:
        main()
