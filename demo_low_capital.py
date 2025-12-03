#!/usr/bin/env python3
"""
Low Capital Backtesting Demo
Test the AI system with $100-250 starting capital
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_low_capital_backtest():
    """Demonstrate backtesting with low capital amounts"""
    print("💰 LOW CAPITAL BACKTESTING DEMO")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Test basic functionality first
    try:
        from data_scraper import DataScraper
        from data_preprocessing import DataPreprocessor

        print("1. Testing Data Pipeline...")
        scraper = DataScraper()
        data = scraper.fetch_yahoo_data('GC=F', '2023-01-01', '2024-01-01')
        print(f"✓ Data collected: {len(data)} records")

        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_features(data)
        print(f"✓ Data processed: {processed_data.shape}")
        print()

    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        return

    # Test different capital amounts
    capital_amounts = [100, 150, 200, 250]

    print("2. Backtesting with Different Capital Amounts...")
    print("Note: This requires PyTorch to be working (Python 3.8-3.13)")
    print("Current Python version may have compatibility issues.")
    print()

    for capital in capital_amounts:
        print(f"Testing with ${capital} capital:")
        print("-" * 30)

        try:
            # Import low capital config
            import config_low_capital as config

            # Simulate basic backtest logic (without actual model)
            print(f"✓ Risk per trade: ${capital * config.RISK_PER_TRADE:.2f} ({config.RISK_PER_TRADE*100}%)")
            print(f"✓ Stop loss: {config.STOP_LOSS_PCT*100}%")
            print(f"✓ Take profit: {config.TAKE_PROFIT_PCT*100}%")
            print(f"✓ Min position size: ${config.MIN_POSITION_SIZE}")
            print(f"✓ Max daily loss: ${capital * config.MAX_DAILY_LOSS:.2f} ({config.MAX_DAILY_LOSS*100}%)")

            # Simulate some trades
            simulate_trades(capital, config)

        except ImportError as e:
            print(f"⚠️  Cannot run full backtest: {e}")
            print("   This is expected on Python 3.14")
            print("   Use Google Colab for full functionality")
        except Exception as e:
            print(f"✗ Backtest failed: {e}")

        print()

def simulate_trades(capital, config):
    """Simulate trading with given capital and config"""
    import numpy as np

    # Simulate 50 trades
    n_trades = 50
    win_rate = 0.55  # 55% win rate

    trades = []
    current_capital = capital

    for i in range(n_trades):
        position_size = min(
            current_capital * config.RISK_PER_TRADE,
            current_capital - config.MIN_POSITION_SIZE  # Don't go below minimum
        )

        # Simulate trade outcome
        if np.random.random() < win_rate:
            # Win
            profit = position_size * config.TAKE_PROFIT_PCT
        else:
            # Loss
            profit = -position_size * config.STOP_LOSS_PCT

        current_capital += profit
        trades.append(profit)

        # Check daily/weekly loss limits
        if current_capital < capital * (1 - config.MAX_DAILY_LOSS):
            print("   ⚠️  Daily loss limit reached - stopping for the day")
            break

    total_return = ((current_capital - capital) / capital) * 100
    n_wins = sum(1 for t in trades if t > 0)
    n_losses = sum(1 for t in trades if t < 0)

    print(f"   After {len(trades)} trades:")
    print("8"
          "8"
          "8")

def show_realistic_growth_projection():
    """Show realistic growth projections"""
    print("3. REALISTIC GROWTH PROJECTIONS")
    print("="*60)

    # Based on the low capital settings
    monthly_return_scenarios = [1.0, 2.0, 3.0, 5.0]  # Conservative estimates

    for capital in [100, 150, 200, 250]:
        print(f"\n💰 Starting with ${capital}:")

        for monthly_return in monthly_return_scenarios:
            # Calculate months to reach $10,000
            target = 10000
            if monthly_return > 0:
                months = np.log(target / capital) / np.log(1 + monthly_return/100)
                years = months / 12

                if years < 10:  # Only show realistic scenarios
                    print("4"
                          ".0f")

def show_risk_warnings():
    """Show important risk warnings"""
    print("\n⚠️  CRITICAL RISK WARNINGS")
    print("="*60)

    warnings = [
        "• Trading forex/crypto involves substantial risk of loss",
        "• Most retail traders lose money in the first year",
        "• Past backtest results do not guarantee future performance",
        "• Never risk money you cannot afford to lose completely",
        "• This system is for educational purposes only",
        "• Always test with paper trading first",
        "• Start with very small amounts while learning",
        "• Have a backup plan for complete loss",
        "• Consider professional financial advice",
        "• Trading can be addictive - set limits"
    ]

    for warning in warnings:
        print(warning)

def show_next_steps():
    """Show recommended next steps"""
    print("\n🚀 RECOMMENDED NEXT STEPS")
    print("="*60)

    steps = [
        "1. PAPER TRADING FIRST",
        "   • Practice with virtual money for 1-3 months",
        "   • Test the AI system on historical data",
        "   • Learn the psychology of trading",
        "",
        "2. START VERY SMALL",
        "   • Begin with $50-100 if possible",
        "   • Use the low capital config settings",
        "   • Focus on learning, not profits",
        "",
        "3. EDUCATION & IMPROVEMENT",
        "   • Study technical analysis fundamentals",
        "   • Learn risk management thoroughly",
        "   • Understand market psychology",
        "",
        "4. GRADUAL SCALING",
        "   • Only increase capital after consistent profits",
        "   • Never risk more than 1-2% per trade initially",
        "   • Reinvest profits conservatively",
        "",
        "5. LONG-TERM MINDSET",
        "   • Accept that this takes 1-3 years minimum",
        "   • Focus on 20-50% annual growth, not 1000%",
        "   • Consistency beats home runs"
    ]

    for step in steps:
        print(step)

def main():
    test_low_capital_backtest()
    show_realistic_growth_projection()
    show_risk_warnings()
    show_next_steps()

    print("\n" + "="*60)
    print("💡 FINAL THOUGHTS:")
    print("• Trading is a marathon, not a sprint")
    print("• The AI system gives you an edge, but not guarantees")
    print("• Risk management is more important than profits")
    print("• Most traders fail - be statistically aware")
    print("• Start small, learn big, grow slowly")
    print("="*60)

if __name__ == "__main__":
    main()