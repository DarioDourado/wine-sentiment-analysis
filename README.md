# 🍷 Wine Sentiment Analysis Dashboard

> **Advanced sentiment analysis for wine reviews with interactive multilingual dashboard**

## 🌟 Features

- **Enhanced TextBlob Analysis** with wine-specific lexicon (700+ terms)
- **Multilingual Dashboard** (English & Portuguese)
- **Interactive Visualizations** with real-time filtering
- **Multiple Analysis Modes** (Dashboard, Terminal, Summary)
- **Export Options** (CSV downloads)

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv wine-env
source wine-env/bin/activate  # macOS/Linux
# wine-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python AS-TG-vinhos.py

# Choose option 1, then run:
streamlit run dashboard.py
```

**Dashboard URL:** http://localhost:8501

## 📦 Installation

**Prerequisites:** Python 3.8+

### **Option 1: Virtual Environment (Recommended)**

1. **Create and activate virtual environment:**

   ```bash
   # Create virtual environment
   python -m venv wine-env

   # Activate (macOS/Linux)
   source wine-env/bin/activate

   # Activate (Windows)
   wine-env\Scripts\activate

   # Verify activation (should show wine-env in prompt)
   which python  # macOS/Linux
   where python  # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **First run only (NLTK data):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### **Option 2: System-wide Installation**

1. **Download project files:**

   - `AS-TG-vinhos.py`
   - `dashboard.py`
   - `requirements.txt`
   - `translation/en.json`
   - `translation/pt.json`

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### **Option 3: Using Conda**

```bash
# Create conda environment
conda create -n wine-analysis python=3.9

# Activate environment
conda activate wine-analysis

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### **Activate Environment (if using virtual env)**

```bash
# Every time you use the project
source wine-env/bin/activate  # macOS/Linux
# wine-env\Scripts\activate   # Windows
```

### **Run Analysis**

```bash
python AS-TG-vinhos.py
```

**Available Options:**

1. 🌐 **Dashboard Interativo** - Web interface with real-time filtering
2. 📊 **Gráficos no Terminal** - Static charts saved as images
3. 📋 **Apenas Resumo** - Text-based analysis
4. 🚪 **Sair** - Exit

### **Deactivate Environment**

```bash
# When finished working
deactivate
```

## 📊 Input Data

**Supported files:**

- `data/avaliacao_vinhos.csv`
- `avaliacao_vinhos.csv`
- `data/wine_reviews.csv`
- `wine_reviews.csv`

**Required columns:**

- `description` - Wine review text
- `points` - Rating (optional)
- `country` - Country (optional)
- `variety` - Grape variety (optional)
- `winery` - Winery name (optional)

**Example CSV:**

```csv
description,points,country,variety,winery
"Excellent fruity aromas...",92,France,Chardonnay,Domaine Example
"Rich and complex...",88,Italy,Sangiovese,Castello Test
```

## 🌍 Dashboard Features

- **Language Selector** - Toggle EN/PT
- **Advanced Filters** - Wine type, sentiment, country, ratings
- **Key Metrics** - Total reviews, sentiment scores, wine terms
- **Interactive Charts** - Donut charts, bar charts, scatter plots
- **Data Explorer** - Sortable table with CSV export

## 📁 Project Structure

```
wine-sentiment-analysis/
├── AS-TG-vinhos.py              # Main analysis script
├── dashboard.py                 # Streamlit dashboard
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── wine-env/                    # Virtual environment (local)
├── translation/                 # Translations
│   ├── en.json                 # English
│   └── pt.json                 # Portuguese
├── data/                       # Generated data
└── imagens/                    # Generated charts
```

## 🐛 Troubleshooting

**Common Issues:**

1. **"No data found":**

   ```bash
   ls -la *.csv data/*.csv
   ```

2. **Streamlit won't start:**

   ```bash
   # Make sure virtual environment is activated
   source wine-env/bin/activate

   pip install --upgrade streamlit
   python -m streamlit run dashboard.py
   ```

3. **Missing dependencies:**

   ```bash
   # Activate environment first
   source wine-env/bin/activate
   pip install -r requirements.txt
   ```

4. **Virtual environment issues:**

   ```bash
   # Remove and recreate environment
   rm -rf wine-env
   python -m venv wine-env
   source wine-env/bin/activate
   pip install -r requirements.txt
   ```

5. **Port conflicts:**
   ```bash
   streamlit run dashboard.py --server.port 8502
   ```

**Environment Verification:**

```bash
# Check if environment is active
echo $VIRTUAL_ENV  # Should show path to wine-env

# Check Python location
which python  # Should point to wine-env/bin/python

# Check installed packages
pip list
```

## 🔧 Configuration

**Customize wine lexicon in `AS-TG-vinhos.py`:**

```python
wine_lexicon = {
    'positive': ['excellent', 'outstanding', 'complex'],
    'negative': ['poor', 'harsh', 'flat'],
    'neutral': ['oak', 'tannin', 'acidity']
}
```

**Edit translations in:**

- `translation/en.json`
- `translation/pt.json`

## 📈 Output Files

**Generated automatically:**

- `data/wine_sentiment_data_en.csv` - Full processed dataset
- `data/wine_analysis_summary_en.csv` - Summary statistics
- `imagens/*.png` - Visualization charts

## 🤝 Contributing

1. Fork the project
2. Create feature branch
3. **Setup development environment:**
   ```bash
   python -m venv wine-env
   source wine-env/bin/activate
   pip install -r requirements.txt
   ```
4. Make changes
5. Test thoroughly
6. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Made with ❤️ for wine enthusiasts and data scientists**

🍷 **Cheers to data-driven wine insights!** 🥂
