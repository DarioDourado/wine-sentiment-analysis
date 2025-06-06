# 🍷 Wine Sentiment Analysis System v2.0

Advanced sentiment analysis system for wine reviews using TextBlob with specialized wine lexicon.

## 🌟 Features

- ✅ **Enhanced sentiment analysis** with 400+ wine-specific terms
- ✅ **Interactive Streamlit dashboard** with advanced filters
- ✅ **11 visualization types** including word clouds and NLP analysis
- ✅ **Multi-language support** (English/Portuguese)
- ✅ **Modular architecture** for easy maintenance and extension
- ✅ **Comprehensive logging** and error handling
- ✅ **Automated testing** suite
- ✅ **Docker support** for easy deployment

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd wine-sentiment-analysis

# Setup project (creates directories and installs dependencies)
python scripts/setup_project.py

# Or manual setup:
pip install -r requirements.txt
```

### Usage

```bash
# Interactive mode (recommended)
python main.py

# Direct modes
python main.py --mode dashboard    # Web dashboard
python main.py --mode charts      # Generate all charts
python main.py --mode terminal    # Terminal analysis
python main.py --mode all         # All modes

# Custom input file
python main.py --input data/raw/my_wine_data.csv
```

## 📁 Project Structure

```
wine-sentiment-analysis/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── src/                       # Source code
│   ├── analysis/             # Sentiment analysis modules
│   │   ├── sentiment_analyzer.py
│   │   ├── wine_lexicon.py
│   │   └── nlp_processor.py
│   ├── data_processing/      # Data handling modules
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── data_enricher.py
│   ├── visualization/        # Charts and dashboard
│   │   ├── chart_generator.py
│   │   ├── dashboard.py
│   │   └── terminal_display.py
│   ├── utils/               # Utilities
│   │   ├── file_utils.py
│   │   ├── logger_config.py
│   │   └── validation_utils.py
│   └── config/              # Configuration
│       ├── settings.py
│       └── constants.py
├── data/                    # Data files
│   ├── raw/                # Original data
│   ├── processed/          # Processed data
│   └── exports/           # Exported results
├── assets/                 # Static assets
│   ├── images/            # Generated charts
│   └── translations/      # Language files
├── tests/                 # Test suite
├── logs/                  # Log files
├── docs/                  # Documentation
└── scripts/              # Utility scripts
```

## 🎯 Analysis Modes

### 1. 🌐 Interactive Dashboard

```bash
python main.py --mode dashboard
# or choose option 1 in interactive mode
```

- Web interface at `http://localhost:8501`
- Real-time filtering and visualization
- Multi-language support

### 2. 📊 Chart Generation

```bash
python main.py --mode charts
# or choose option 2 in interactive mode
```

Generates 11 comprehensive charts:

1. Polarity comparison (TextBlob vs Enhanced)
2. Sentiment distribution
3. Wine terms impact analysis
4. Top wine terms frequency
5. Rating vs sentiment correlation
6. Varietal analysis
7. Country analysis
8. Complete dashboard summary
9. Wine type analysis
10. Word cloud analysis
11. Advanced NLP analysis

### 3. 📋 Terminal Analysis

```bash
python main.py --mode terminal
# or choose option 3 in interactive mode
```

- Comprehensive text-based analysis
- Statistical summaries
- NLP insights and correlations
- Performance metrics

## 🔧 Configuration

Edit `src/config/settings.py` to customize:

- Sentiment thresholds
- Chart appearance
- File paths
- Language preferences

## 📊 Data Format

Expected CSV format:

```csv
wine,winery,varietal,review,rating,price,country
"Wine Name","Winery Name","Grape Variety","Review text...",95,50.0,"Country"
```

Minimum required columns:

- `review` (text): Wine review content
- `rating` (numeric, optional): Wine rating
- `price` (numeric, optional): Wine price

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_sentiment.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance

- **Processing speed**: ~1000 reviews/second
- **Memory usage**: ~50MB for 10k reviews
- **Chart generation**: ~30 seconds for all 11 charts
- **Dashboard startup**: ~5 seconds

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TextBlob library for base sentiment analysis
- Streamlit for dashboard framework
- Wine industry experts for terminology validation

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ and 🍷 by the Wine Analysis Team**
