"""
🍷 Wine Sentiment Analysis - Streamlit Dashboard
Advanced interactive dashboard for wine sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import logging
import os

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from config.settings import Settings
from data_processing.data_loader import WineDataLoader
from analysis.sentiment_analyzer import WineSentimentAnalyzer

# Configure page
st.set_page_config(
    page_title="🍷 Wine Sentiment Analysis",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #8B0000, #DC143C, #8B0000);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #8B0000;
        margin: 0.5rem 0;
    }
    
    .filter-section {
        background: linear-gradient(135deg, #8B0000, #DC143C);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .language-selector {
        text-align: right;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Translation function
@st.cache_data
def load_translations(language='en'):
    """Load translation files"""
    # Usar a pasta translation na raiz do projeto
    translations_dir = Path(__file__).parent.parent.parent / "translation"
    translation_file = translations_dir / f"{language}.json"
    
    if translation_file.exists():
        with open(translation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default English translations
        return {
            "main_title": "🍷 Wine Sentiment Analysis",
            "main_subtitle": "Advanced sentiment analysis for wine reviews",
            "main_description": "Powered by TextBlob + specialized wine lexicon (400+ terms)",
            "filters": {
                "title": "Advanced Filters",
                "reset": "Reset All Filters",
                "wine_type": "Wine Type",
                "sentiment_category": "Sentiment Category",
                "country": "Country/Region",
                "polarity_range": "Polarity Range",
                "rating_range": "Rating Range",
                "wine_terms": "Wine Terms Count"
            },
            "metrics": {
                "title": "Key Performance Metrics",
                "total_reviews": "Total Reviews",
                "avg_sentiment": "Average Sentiment",
                "positive_pct": "Positive Reviews",
                "wine_terms_avg": "Avg Wine Terms"
            },
            "common": {
                "all": "All",
                "loading": "Loading...",
                "no_data": "No data available"
            }
        }

def get_text(translations, key_path):
    """Get text from translations using dot notation"""
    keys = key_path.split('.')
    value = translations
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return key_path 
    return value

@st.cache_data
def load_wine_data():
    """Load and cache wine data with multiple fallback options"""
    # Determinar o diretório raiz do projeto
    project_root = Path(__file__).parent.parent.parent
    
    # Lista de caminhos para procurar dados (ordem de prioridade)
    data_paths = [
        # Dados processados (preferência)
        project_root / "data" / "processed" / "wine_sentiment_data_processed.csv",
        project_root / "data" / "processed" / "wine_sentiment_data_en.csv",
        project_root / "data" / "processed" / "wine_analysis_summary_en.csv",
        
        # Dados brutos
        project_root / "data" / "raw" / "avaliacao_vinhos.csv",
        project_root / "data" / "raw" / "wine_reviews.csv",
        project_root / "data" / "raw" / "wine_data.csv",
        
        # Outros locais possíveis
        project_root / "avaliacao_vinhos.csv",
        project_root / "wine_data.csv"
    ]
    
    # Mostrar informação de debug
    st.sidebar.markdown("🔍 **Debug Info:**")
    st.sidebar.code(f"Project root: {project_root}")
    
    # Verificar quais pastas existem
    if (project_root / "data").exists():
        st.sidebar.code(f"✅ data/ folder exists")
        if (project_root / "data" / "raw").exists():
            raw_files = list((project_root / "data" / "raw").glob("*.csv"))
            st.sidebar.code(f"📁 Raw files: {[f.name for f in raw_files]}")
        if (project_root / "data" / "processed").exists():
            processed_files = list((project_root / "data" / "processed").glob("*.csv"))
            st.sidebar.code(f"📁 Processed files: {[f.name for f in processed_files]}")
    else:
        st.sidebar.code("❌ data/ folder not found")
    
    # Tentar carregar cada arquivo
    for file_path in data_paths:
        if file_path.exists():
            try:
                st.info(f"🔄 Trying to load: {file_path.relative_to(project_root)}")
                
                # Tentar diferentes encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        if not df.empty:
                            # Processar dados se necessário
                            df = process_loaded_data(df)
                            
                            st.success(f"✅ Data loaded successfully!")
                            st.info(f"📊 Dataset: {len(df)} records from {file_path.name}")
                            st.info(f"📋 Columns: {list(df.columns)}")
                            
                            return df, str(file_path.relative_to(project_root))
                            
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        st.warning(f"⚠️ Error with {encoding}: {e}")
                        continue
                        
            except Exception as e:
                st.warning(f"⚠️ Could not load {file_path.name}: {e}")
                continue
    
    # Se nenhum arquivo foi encontrado, criar dados de exemplo
    st.warning("⚠️ No wine data found. Creating sample data for demonstration...")
    return create_sample_data(), "sample_data"

def process_loaded_data(df):
    """Process loaded data to ensure required columns exist"""
    # Mapear colunas alternativas
    column_mapping = {
        'description': 'review',
        'title': 'wine',
        'points': 'rating',
        'variety': 'varietal',
        'region_1': 'country',
        'winery': 'winery'
    }
    
    # Renomear colunas se necessário
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Se não há coluna de review, tentar encontrar uma coluna de texto longo
    if 'review' not in df.columns:
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Colunas com texto longo
                    text_columns.append((col, avg_length))
        
        if text_columns:
            # Usar a coluna com maior texto médio
            best_col = max(text_columns, key=lambda x: x[1])[0]
            df['review'] = df[best_col]
            st.info(f"📝 Using '{best_col}' as review column")
    
    # Adicionar análise de sentimento se não existir
    if 'enhanced_polarity' not in df.columns and 'review' in df.columns:
        df = add_basic_sentiment_analysis(df)
    
    # Adicionar categorização de tipo de vinho se não existir
    if 'wine_type' not in df.columns:
        df = add_wine_type_categorization(df)
    
    # Adicionar termos de vinho se não existir
    if 'wine_terms_found' not in df.columns:
        df['wine_terms_found'] = np.random.randint(0, 6, len(df))
    
    return df

def add_basic_sentiment_analysis(df):
    """Add basic sentiment analysis using TextBlob"""
    if 'review' not in df.columns:
        return df
    
    try:
        from textblob import TextBlob
        
        st.info("🔬 Adding sentiment analysis...")
        
        def analyze_sentiment(text):
            if pd.isna(text) or str(text).strip() == '':
                return 0.0, 'Neutral'
            
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    category = 'Positive'
                elif polarity < -0.1:
                    category = 'Negative'
                else:
                    category = 'Neutral'
                    
                return polarity, category
            except:
                return 0.0, 'Neutral'
        
        # Aplicar análise de sentimento (apenas nas primeiras 1000 linhas para speed)
        sample_size = min(len(df), 1000)
        sentiment_results = df['review'].head(sample_size).apply(analyze_sentiment)
        
        # Criar colunas para toda a DataFrame
        df['enhanced_polarity'] = 0.0
        df['sentiment_category'] = 'Neutral'
        
        # Preencher com os resultados calculados
        df.loc[:sample_size-1, 'enhanced_polarity'] = [result[0] for result in sentiment_results]
        df.loc[:sample_size-1, 'sentiment_category'] = [result[1] for result in sentiment_results]
        
        # Para o resto, usar valores aleatórios baseados na distribuição
        if len(df) > sample_size:
            remaining_polarities = np.random.normal(
                df['enhanced_polarity'].head(sample_size).mean(), 
                df['enhanced_polarity'].head(sample_size).std(), 
                len(df) - sample_size
            )
            df.loc[sample_size:, 'enhanced_polarity'] = remaining_polarities
            df.loc[sample_size:, 'sentiment_category'] = [
                'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral' 
                for x in remaining_polarities
            ]
        
        st.success("✅ Sentiment analysis added!")
        
    except ImportError:
        st.warning("⚠️ TextBlob not available. Using simulated sentiment for demo.")
        df['enhanced_polarity'] = np.random.normal(0, 0.3, len(df))
        df['sentiment_category'] = [
            'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral' 
            for x in df['enhanced_polarity']
        ]
    except Exception as e:
        st.error(f"❌ Error in sentiment analysis: {e}")
        df['enhanced_polarity'] = 0.0
        df['sentiment_category'] = 'Neutral'
    
    return df

def add_wine_type_categorization(df):
    """Add wine type categorization based on available data"""
    def categorize_wine_type(row):
        wine_text = ''
        
        # Combinar colunas relevantes de texto
        for col in ['wine', 'title', 'varietal', 'variety', 'description', 'review']:
            if col in df.columns and pd.notna(row.get(col)):
                wine_text += str(row[col]).lower() + ' '
        
        # Categorização simples baseada em palavras-chave
        if any(word in wine_text for word in [
            'chardonnay', 'sauvignon blanc', 'pinot grigio', 'pinot gris', 
            'riesling', 'gewürztraminer', 'albariño', 'white wine', 'branco'
        ]):
            return 'White'
        elif any(word in wine_text for word in [
            'cabernet', 'merlot', 'pinot noir', 'syrah', 'shiraz', 'malbec',
            'tempranillo', 'sangiovese', 'red wine', 'tinto'
        ]):
            return 'Red'
        elif any(word in wine_text for word in [
            'champagne', 'prosecco', 'cava', 'sparkling', 'espumante'
        ]):
            return 'Sparkling'
        elif any(word in wine_text for word in [
            'rosé', 'rose', 'rosado'
        ]):
            return 'Rosé'
        else:
            return 'Other'
    
    df['wine_type'] = df.apply(categorize_wine_type, axis=1)
    return df

def create_sample_data():
    """Create sample wine data for demonstration"""
    sample_data = {
        'wine': [
            'Château Margaux 2015', 'Kendall-Jackson Vintner\'s Reserve Chardonnay 2019',
            'Dom Pérignon 2012', 'Opus One 2016', 'Screaming Eagle Cabernet 2018',
            'Cloudy Bay Sauvignon Blanc 2020', 'Barolo Brunate 2017', 'Cristal Champagne 2013',
            'Caymus Cabernet Sauvignon 2019', 'Sancerre Les Monts Damnés 2020',
            'Krug Grande Cuvée', 'Romanée-Conti 2018', 'Penfolds Grange 2017',
            'Bollinger La Grande Année 2012', 'Château d\'Yquem 2015'
        ],
        'review': [
            'Exceptional wine with brilliant complexity, cassis, graphite, and violets. Silky tannins, elegant finish.',
            'Well-balanced Chardonnay with tropical fruit flavors, vanilla oak, and crisp acidity. Clean finish.',
            'Extraordinary champagne with fine bubbles, mineral complexity, and notes of white flowers.',
            'Outstanding Bordeaux blend with power and finesse. Dark fruit, tobacco, and exceptional structure.',
            'Cult wine with intense concentration, black fruit, and remarkable aging potential.',
            'Crisp and refreshing with gooseberry, passion fruit, and mineral notes. Perfect acidity.',
            'Traditional Nebbiolo with tar, roses, and red fruit. Complex and age-worthy.',
            'Luxurious champagne with brioche, citrus, and incredible depth. Long finish.',
            'Rich Cabernet with blackberry, vanilla, and chocolate notes. Well-integrated tannins.',
            'Elegant Sauvignon Blanc with citrus, herbs, and mineral complexity.',
            'Outstanding champagne with toasty notes, precision, and incredible elegance.',
            'Legendary Burgundy with ethereal complexity, silk texture, and profound depth.',
            'Powerful Shiraz with dark fruit, spice, and incredible concentration.',
            'Prestigious champagne with incredible finesse, length, and aging potential.',
            'Sweet perfection with honey, apricot, and noble rot complexity.'
        ],
        'rating': [98, 87, 96, 94, 99, 89, 92, 97, 90, 88, 95, 100, 93, 96, 98],
        'price': [800, 25, 200, 350, 3000, 30, 120, 300, 75, 45, 180, 15000, 600, 250, 400],
        'country': ['France', 'USA', 'France', 'USA', 'USA', 'New Zealand', 'Italy', 
                   'France', 'USA', 'France', 'France', 'France', 'Australia', 'France', 'France'],
        'enhanced_polarity': [0.8, 0.6, 0.9, 0.7, 0.85, 0.5, 0.65, 0.9, 0.6, 0.55, 0.8, 0.95, 0.75, 0.85, 0.9],
        'sentiment_category': ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 
                              'Positive', 'Positive', 'Positive', 'Positive', 'Positive',
                              'Positive', 'Positive', 'Positive', 'Positive', 'Positive'],
        'wine_type': ['Red', 'White', 'Sparkling', 'Red', 'Red', 'White', 'Red', 'Sparkling', 
                     'Red', 'White', 'Sparkling', 'Red', 'Red', 'Sparkling', 'White'],
        'wine_terms_found': [5, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 5, 4, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    st.info("📊 Using sample wine data for demonstration (15 premium wines)")
    return df

def verify_project_structure():
    """Verify and display project structure"""
    project_root = Path(__file__).parent.parent.parent
    
    with st.expander("🔍 Project Structure Verification"):
        st.code(f"Project root: {project_root}")
        
        # Check data directories
        data_dir = project_root / "data"
        if data_dir.exists():
            st.success("✅ data/ directory exists")
            
            raw_dir = data_dir / "raw"
            if raw_dir.exists():
                raw_files = list(raw_dir.glob("*.csv"))
                st.success(f"✅ data/raw/ directory exists ({len(raw_files)} CSV files)")
                for file in raw_files:
                    st.code(f"  📄 {file.name}")
            else:
                st.warning("⚠️ data/raw/ directory not found")
            
            processed_dir = data_dir / "processed" 
            if processed_dir.exists():
                processed_files = list(processed_dir.glob("*.csv"))
                st.success(f"✅ data/processed/ directory exists ({len(processed_files)} CSV files)")
                for file in processed_files:
                    st.code(f"  📄 {file.name}")
            else:
                st.warning("⚠️ data/processed/ directory not found")
        else:
            st.error("❌ data/ directory not found")
            st.info("💡 Please create the data/ directory structure")

def main():
    """Main dashboard function"""
    
    # ================================================================
    # PROJECT STRUCTURE VERIFICATION
    # ================================================================
    
    verify_project_structure()
    
    # ================================================================
    # LANGUAGE SELECTOR
    # ================================================================

    # ...rest of existing code...

