# LSTM Trading AI Research Paper

This directory contains the complete research paper for the LSTM-based algorithmic trading system.

## 📄 Paper Overview

**Title:** LSTM-Based Deep Learning for Algorithmic Trading: A Comprehensive Study on XAUUSD and BTCUSD

**Authors:** Jonus Nattapong

**Abstract:** This research presents a comprehensive deep learning approach for algorithmic trading using Long Short-Term Memory (LSTM) neural networks with attention mechanisms. The study focuses on two major financial instruments: Gold futures (XAUUSD) and Bitcoin (BTCUSD), implementing a sophisticated trading system that combines technical analysis with advanced machine learning techniques.

## 📁 Files

- `research_paper.tex` - Main LaTeX document
- `references.bib` - Bibliography database
- `compile_paper.py` - Compilation script
- `README.md` - This file

## 🔧 Compilation

### Prerequisites

Install a LaTeX distribution:
- **Windows:** MiKTeX or TeX Live
- **macOS:** MacTeX
- **Linux:** TeX Live

Required packages:
- amsmath, amssymb, amsthm
- graphicx, float
- hyperref, natbib
- geometry, setspace
- booktabs, multirow
- listings, xcolor
- algorithm, algpseudocode
- subcaption

### Compile the Paper

#### Option 1: Using the Python Script (Recommended)
```bash
python compile_paper.py
```

#### Option 2: Manual Compilation
```bash
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

## 📖 Paper Structure

### 1. Abstract
- Research overview and key contributions
- Keywords: LSTM, Deep Learning, Algorithmic Trading, Technical Analysis, Risk Management

### 2. Introduction
- Background and motivation
- Research objectives
- Key contributions

### 3. Literature Review
- Algorithmic trading evolution
- Machine learning in finance
- LSTM networks in trading
- Technical analysis integration
- Risk management approaches

### 4. Methodology
- System architecture
- Data collection and preprocessing
- LSTM model architecture
- Training methodology
- Risk management framework

### 5. Experimental Results
- Dataset description
- Model performance comparison
- Trading performance metrics
- Low capital trading analysis
- Feature importance analysis
- Market condition analysis

### 6. Discussion
- Model strengths and limitations
- Practical implementation considerations
- Future research directions

### 7. Conclusion
- Key findings summary
- Research implications
- Future work

### 8. References
- Comprehensive bibliography of related work

### Appendix
- Implementation details
- Code structure
- Configuration parameters
- Detailed performance metrics

## 🎯 Key Findings

### Model Performance
- Multi-Head Attention LSTM achieves 91.2% R² score
- Superior performance compared to traditional approaches
- Effective across different market conditions

### Trading Results (2023-2024)
- **XAUUSD:** 24.7% total return, 18.3% annual return
- **BTCUSD:** 31.2% total return, 22.1% annual return
- **Combined:** 27.8% total return, 20.1% annual return
- Sharpe Ratio: 1.89
- Win Rate: 59.7%

### Low Capital Projections
- Realistic growth scenarios for $100-250 starting capital
- Conservative risk management protocols
- 1-3 year growth timelines

## 🔬 Methodology Highlights

### Technical Indicators
- 40+ indicators across 5 categories
- Trend, Momentum, Volatility, Volume, Price features
- Systematic feature engineering

### Model Architecture
- Multi-layer LSTM with attention mechanisms
- Dropout regularization
- Early stopping and learning rate scheduling

### Risk Management
- 1\% risk per trade for low capital
- 1.5\% stop loss, 3\% take profit
- Daily/weekly loss limits
- Position sizing algorithms

## 🚀 Future Research Directions

- Integration of fundamental data and sentiment analysis
- Multi-timeframe analysis and ensemble methods
- Reinforcement learning for dynamic strategy adaptation
- Cross-asset portfolio optimization
- High-frequency trading applications

## 📊 Citation

If you use this research in your work, please cite:

```bibtex
@article{nattapong2025lstm,
  title={LSTM-Based Deep Learning for Algorithmic Trading: A Comprehensive Study on XAUUSD and BTCUSD},
  author={Nattapong, Jonus},
  year={2025},
  journal={Independent Research Publication}
}
```

## 🤝 Contributing

This research paper is part of the open-source LSTM Trading AI project. Contributions and feedback are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

This research paper is released under the MIT License. See the main project LICENSE file for details.

## 📞 Contact

**Jonus Nattapong**
- Email: jonus.nattapong@email.com
- GitHub: [JonusNattapong](https://github.com/JonusNattapong)
- LinkedIn: [Jonus Nattapong](https://linkedin.com/in/jonus-nattapong)

---

*This research represents a comprehensive study of deep learning applications in algorithmic trading, with practical implementations and rigorous validation.*