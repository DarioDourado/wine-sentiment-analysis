import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns
from textblob import TextBlob
import os
import platform 
from collections import Counter
import nltk
from wordcloud import WordCloud
NLP_AVAILABLE = True


# Configure matplotlib to handle Unicode properly
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# For macOS specifically
if platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = ['Apple Symbols', 'Helvetica', 'Arial']

# ====================================================================
# SECTION 1: CARREGAR DADOS & LIMPEZA DE CARACTERES ESPECIAIS
# ====================================================================

def load_and_clean_data(csv_file):
    """Carregar e limpar caracteres especiais do CSV"""
    
    print(f"📁 A carregar ficheiro: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ {len(df)} linhas carregadas com sucesso")
    print(f"📋 Colunas disponíveis: {list(df.columns)}")
    
    # Função para limpar texto
    def clean_text(text):
        if pd.isnull(text):
            return ''
        text = str(text)
        text = re.sub(r'[^\w\s.,;:!?\'\"()\[\]-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Verificar se tem coluna de review
    if 'review' not in df.columns:
        print(f"❌ Coluna 'review' não encontrada!")
        print(f"📋 Colunas disponíveis: {list(df.columns)}")
        
        # Procurar coluna similar
        possible_review_cols = [
            'description', 'comment', 'text', 'avaliacao', 'avaliacoes', 
            'descricao', 'comentario', 'comentarios', 'note', 'notes'
        ]
        
        review_col = None
        for col in possible_review_cols:
            if col in df.columns:
                review_col = col
                print(f"✅ A usar coluna '{col}' como reviews")
                break
        
        if review_col is None:
            # Mostrar colunas disponíveis para escolha manual
            print("\n📋 Escolha a coluna que contém as avaliações:")
            for i, col in enumerate(df.columns, 1):
                sample_value = str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else "N/A"
                print(f"   {i}. '{col}' - Exemplo: {sample_value}...")
            
            try:
                choice = int(input(f"\n👉 Número da coluna (1-{len(df.columns)}): ")) - 1
                if 0 <= choice < len(df.columns):
                    review_col = df.columns[choice]
                    print(f"✅ Coluna '{review_col}' selecionada")
                else:
                    raise ValueError("Escolha inválida")
            except (ValueError, KeyboardInterrupt):
                raise ValueError("Nenhuma coluna de review selecionada!")
        
        # Renomear coluna para 'review'
        df = df.rename(columns={review_col: 'review'})
        print(f"🔄 Coluna '{review_col}' renomeada para 'review'")
    
    # Remover linhas sem avaliação
    original_count = len(df)
    df = df[~df['review'].isnull() & (df['review'].str.strip() != '')]
    print(f"🧹 Removidas {original_count - len(df)} linhas vazias")
    
    if len(df) == 0:
        raise ValueError("Nenhuma avaliação válida encontrada no ficheiro!")
    
    # Limpar colunas de texto
    text_columns = ['wine', 'winery', 'category', 'designation', 'varietal', 'appellation', 'reviewer', 'review']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            print(f"✅ Coluna '{col}' limpa")
    
    print(f"✅ Dados carregados e limpos: {len(df)} avaliações")
    return df

# ====================================================================
# SECTION 2: NORMALIZAÇÃO DE DADOS
# ====================================================================

def normalize_data(df):
    """Normalização de dados e conversão de tipos"""
    
    print("🔧 A normalizar dados...")
    
    # Normalizar varietal
    def normalize_varietal(text):
        if pd.isnull(text) or text == '':
            return 'Unknown'
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text.title()
    
    df['varietal'] = df['varietal'].apply(normalize_varietal)
    
    # Converter colunas numéricas
    numeric_cols = ['alcohol', 'price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
            if col == 'price':
                df[col] = df[col].str.replace(r'[$,€£¥₹]', '', regex=True)
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
            elif col == 'alcohol':
                df[col] = df[col].str.replace(r'[%°]', '', regex=True)
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Converter rating para número
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'] = df['rating'].round().astype('Int64')
    
    print("✅ Normalização concluída")
    return df

# ====================================================================
# SECTION 3: CATEGORIZAÇÃO DE TIPO DE VINHO
# ====================================================================

def categorize_wine_type(wine_name, varietal, category=None):
    """Categorizar tipo de vinho baseado no nome, varietal e categoria"""
    
    if pd.isnull(wine_name):
        wine_name = ''
    if pd.isnull(varietal):
        varietal = ''
    if pd.isnull(category):
        category = ''
    
    # Converter para string e minúsculas para análise
    wine_text = f"{wine_name} {varietal} {category}".lower()
    
    # Categorias de vinho tinto
    red_indicators = [
        'red', 'tinto', 'rouge', 'rosso',
        'cabernet', 'cabernet sauvignon', 'merlot', 'pinot noir', 'syrah', 'shiraz',
        'tempranillo', 'sangiovese', 'malbec', 'grenache', 'garnacha', 'nebbiolo',
        'barbera', 'zinfandel', 'petit verdot', 'carmenere', 'mourvedre', 'monastrell',
        'tannat', 'aglianico', 'montepulciano', 'nero d avola', 'blaufränkisch', 'baga'
    ]

    # Categorias de vinho branco
    white_indicators = [
        'white', 'branco', 'blanc', 'bianco',
        'chardonnay', 'sauvignon blanc', 'riesling', 'pinot grigio', 'pinot gris',
        'gewurztraminer', 'viognier', 'chenin blanc', 'semillon', 'muscadet',
        'albariño', 'alvarinho', 'vermentino', 'greco', 'fiano', 'trebbiano',
        'soave', 'vinho verde', 'grüner veltliner', 'godello', 'garganega'
    ]

    # Categorias de vinho rosé
    rose_indicators = [
        'rosé', 'rose', 'rosado', 'blush', 'pink', 'clairet'
    ]

    # Categorias de espumante
    sparkling_indicators = [
        'champagne', 'sparkling', 'espumante', 'cava', 'prosecco',
        'cremant', 'crémant', 'franciacorta', 'lambrusco', 'asti',
        'sekt', 'bubble', 'brut', 'extra brut', 'método clássico', 'methode traditionnelle'
    ]

    # Categorias de vinho doce/fortificado
    dessert_indicators = [
        'port', 'porto', 'sherry', 'madeira', 'sauternes', 'tokaj',
        'ice wine', 'eiswein', 'dessert', 'sweet', 'doce', 'moscatel',
        'late harvest', 'botrytis', 'passito', 'vin santo', 'banyuls',
        'recioto', 'liquoroso', 'vin doux naturel'
    ]
    
    # Verificar cada categoria
    if any(indicator in wine_text for indicator in sparkling_indicators):
        return 'Espumante'
    elif any(indicator in wine_text for indicator in dessert_indicators):
        return 'Doce/Fortificado'
    elif any(indicator in wine_text for indicator in rose_indicators):
        return 'Rosé'
    elif any(indicator in wine_text for indicator in red_indicators):
        return 'Tinto'
    elif any(indicator in wine_text for indicator in white_indicators):
        return 'Branco'
    else:
        return 'Outro/Indefinido'

# ====================================================================
# SECTION 4: ENRIQUECIMENTO DE DADOS
# ====================================================================

def enrich_data(df):
    """Enriquecer dados com categorização e extração de informação"""
    
    print("✨ A enriquecer dados...")
    
    # Categorizar rating técnico
    def rating_category(r):
        if pd.isnull(r):
            return 'Unknown'
        elif r >= 95:
            return 'Classic'
        elif r >= 90:
            return 'Outstanding'
        elif r >= 85:
            return 'Very Good'
        elif r >= 80:
            return 'Good'
        else:
            return 'Below Average'
    
    df['rating_category'] = df['rating'].apply(rating_category)
    
    # Extrair país da coluna 'appellation'
    df['country'] = df['appellation'].astype(str).str.split(',').apply(
        lambda x: x[-1].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )
    
    # Extração do ano do vinho da coluna 'wine'
    def extract_year(text):
        match = re.search(r'\b(19[0-9]{2}|20[0-2][0-9]|2025)\b', str(text))
        return int(match.group(0)) if match else pd.NA
    
    df['wine_year'] = df['wine'].apply(extract_year).astype('Int64')
    
    # Categorizar tipo de vinho
    print("🍷 A categorizar tipos de vinho...")
    df['wine_type'] = df.apply(
        lambda row: categorize_wine_type(
            row.get('wine', ''), 
            row.get('varietal', ''), 
            row.get('category', '')
        ), axis=1
    )
    
    # Tratamento de valores nulos
    df['wine_year'] = df['wine_year'].fillna(-1)
    df['winery'] = df['winery'].replace('', 'Unknown').fillna('Unknown')
    df['country'] = df['country'].replace('', 'Unknown').fillna('Unknown')
    df['varietal'] = df['varietal'].replace('', 'Unknown').fillna('Unknown')
    
    # Mostrar distribuição de tipos de vinho
    wine_type_dist = df['wine_type'].value_counts()
    print("🍷 Distribuição por tipo de vinho:")
    for wine_type, count in wine_type_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   • {wine_type}: {count} ({percentage:.1f}%)")
    
    print("✅ Enriquecimento concluído")
    return df

# ====================================================================
# SECTION 5: VARIETAL (Casta) HANDLER E LIMPEZA
# ====================================================================

def finalize_data_cleaning(df):
    """Varietal (Casta) contém por vezes mais de uma casta, assim o objetivo
    é adicionar uma coluna com a lista de castas normalizadas e remover duplicados.
    """
    
    print("🧹 A finalizar limpeza de dados...")
    
    def normalize_varietal(text):
        if pd.isnull(text) or text == '':
            return 'Unknown'
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text.title()
    
    df['varietal_list'] = df['varietal'].str.split(r',\s*')
    df_exploded = df.explode('varietal_list')
    df_exploded['varietal_list'] = df_exploded['varietal_list'].apply(normalize_varietal)
    data_clean = df_exploded.drop(columns=['varietal']).rename(columns={'varietal_list': 'varietal'})
    
    # Remover duplicados
    data_clean = data_clean.drop_duplicates(subset=['wine', 'winery', 'category', 'designation', 'varietal', 'appellation', 'reviewer', 'review'])
    
    print(f"✅ Limpeza finalizada: {len(data_clean)} registos únicos")
    return data_clean

# ====================================================================
# SECTION 6: ANÁLISE DE SENTIMENTO EM INGLÊS
# ====================================================================

def configure_wine_lexicon():
    """Léxico especializado para análise de sentimento em críticas de vinho."""
    return {
        # Positivos
        'exceptional': 1.0, 'outstanding': 0.9, 'brilliant': 0.9, 'sublime': 0.9,
        'magnificent': 0.9, 'superb': 0.8, 'excellent': 0.8, 'elegant': 0.8,
        'refined': 0.7, 'sophisticated': 0.8, 'complex': 0.7, 'balanced': 0.7,
        'harmonious': 0.6, 'finesse': 0.7, 'structured': 0.6, 'concentrated': 0.6,
        'intense': 0.6, 'rich': 0.6, 'full-bodied': 0.5, 'smooth': 0.6,
        'silky': 0.7, 'velvety': 0.7, 'crisp': 0.5, 'fresh': 0.6, 'vibrant': 0.7,
        'lively': 0.6, 'aromatic': 0.6, 'fragrant': 0.6, 'minerality': 0.3,
        'terroir': 0.4, 'premium': 0.6, 'reserve': 0.3, 'grand': 0.7, 
        'noble': 0.6, 'distinguished': 0.7, 'expressive': 0.6, 'profound': 0.7,
        'remarkable': 0.8, 'stunning': 0.9, 'delicious': 0.7, 'gorgeous': 0.8,
        'beautiful': 0.7, 'wonderful': 0.8, 'fantastic': 0.8, 'amazing': 0.8,
        'lovely': 0.6, 'delightful': 0.7, 'impressive': 0.7, 'spectacular': 0.9,
        'incredible': 0.8, 'perfect': 0.9, 'divine': 0.9, 'heavenly': 0.8,
        'floral': 0.5, 'juicy': 0.6, 'round': 0.5, 'succulent': 0.7,
        'lingering': 0.6, 'spicy': 0.4, 'peppery': 0.3, 'mature': 0.3,
        'polished': 0.6, 'nuanced': 0.5, 'fine': 0.5, 'bright': 0.5,
        'freshness': 0.5, 'depth': 0.5, 'charming': 0.6, 'savory': 0.6,
        'supple': 0.6, 'refreshed': 0.6, 'seductive': 0.7, 'alluring': 0.7,
        'focused': 0.6, 'elevated': 0.6,

        # Negativos
        'unbalanced': -0.8, 'harsh': -0.6, 'rough': -0.5, 'coarse': -0.5,
        'astringent': -0.4, 'bitter': -0.3, 'sour': -0.4, 'flat': -0.6,
        'dull': -0.4, 'lifeless': -0.6, 'thin': -0.4, 'watery': -0.6,
        'weak': -0.5, 'hollow': -0.5, 'cloying': -0.6, 'clumsy': -0.6,
        'awkward': -0.5, 'unripe': -0.4, 'overripe': -0.4, 'oxidized': -0.8,
        'corked': -0.9, 'faulty': -0.8, 'flawed': -0.7, 'disappointing': -0.6,
        'metallic': -0.5, 'vegetal': -0.3, 'musty': -0.6, 'brett': -0.8,
        'awful': -0.8, 'terrible': -0.8, 'horrible': -0.8, 'disgusting': -0.9,
        'unpleasant': -0.6, 'poor': -0.5, 'mediocre': -0.4, 'boring': -0.4,
        'burnt': -0.4, 'overwhelming': -0.4, 'tired': -0.5, 'muddy': -0.5,
        'stale': -0.6, 'acidic': -0.4, 'over-oaked': -0.5, 'imprecise': -0.4,
        'green': -0.4, 'volatile': -0.5, 'dry finish': -0.4,

        # Neutros técnicos
        'tannins': 0.0, 'acidity': 0.0, 'alcohol': 0.0, 'oak': 0.0,
        'barrel': 0.0, 'malolactic': 0.0, 'vintage': 0.0, 'appellation': 0.0,
        'nose': 0.0, 'palate': 0.0, 'finish': 0.0, 'bouquet': 0.0,
        'body': 0.0, 'structure': 0.0, 'texture': 0.0, 'color': 0.0,
        'grape': 0.0, 'blend': 0.0, 'fermentation': 0.0, 'aging': 0.0,
        'winery': 0.0, 'cellar': 0.0
    }
    
    
    """Léxico especializado para análise de sentimento em críticas de vinho."""
    return {
        # Positivos
        'exceptional': 1.0, 'outstanding': 0.9, 'brilliant': 0.9, 'sublime': 0.9,
        'magnificent': 0.9, 'superb': 0.8, 'excellent': 0.8, 'elegant': 0.8,
        'refined': 0.7, 'sophisticated': 0.8, 'complex': 0.7, 'balanced': 0.7,
        'harmonious': 0.6, 'finesse': 0.7, 'structured': 0.6, 'concentrated': 0.6,
        'intense': 0.6, 'rich': 0.6, 'full-bodied': 0.5, 'smooth': 0.6,
        'silky': 0.7, 'velvety': 0.7, 'crisp': 0.5, 'fresh': 0.6, 'vibrant': 0.7,
        'lively': 0.6, 'aromatic': 0.6, 'fragrant': 0.6, 'minerality': 0.3,
        'terroir': 0.4, 'premium': 0.6, 'reserve': 0.3, 'grand': 0.7, 
        'noble': 0.6, 'distinguished': 0.7, 'expressive': 0.6, 'profound': 0.7,
        'remarkable': 0.8, 'stunning': 0.9, 'delicious': 0.7, 'gorgeous': 0.8,
        'beautiful': 0.7, 'wonderful': 0.8, 'fantastic': 0.8, 'amazing': 0.8,
        'lovely': 0.6, 'delightful': 0.7, 'impressive': 0.7, 'spectacular': 0.9,
        'incredible': 0.8, 'perfect': 0.9, 'divine': 0.9, 'heavenly': 0.8,
        'floral': 0.5, 'juicy': 0.6, 'round': 0.5, 'succulent': 0.7,
        'lingering': 0.6, 'spicy': 0.4, 'peppery': 0.3, 'mature': 0.3,
        'polished': 0.6, 'nuanced': 0.5, 'fine': 0.5, 'bright': 0.5,
        'freshness': 0.5, 'depth': 0.5, 'charming': 0.6, 'savory': 0.6,
        'supple': 0.6, 'refreshed': 0.6, 'seductive': 0.7, 'alluring': 0.7,
        'focused': 0.6, 'elevated': 0.6,

        # Negativos
        'unbalanced': -0.8, 'harsh': -0.6, 'rough': -0.5, 'coarse': -0.5,
        'astringent': -0.4, 'bitter': -0.3, 'sour': -0.4, 'flat': -0.6,
        'dull': -0.4, 'lifeless': -0.6, 'thin': -0.4, 'watery': -0.6,
        'weak': -0.5, 'hollow': -0.5, 'cloying': -0.6, 'clumsy': -0.6,
        'awkward': -0.5, 'unripe': -0.4, 'overripe': -0.4, 'oxidized': -0.8,
        'corked': -0.9, 'faulty': -0.8, 'flawed': -0.7, 'disappointing': -0.6,
        'metallic': -0.5, 'vegetal': -0.3, 'musty': -0.6, 'brett': -0.8,
        'awful': -0.8, 'terrible': -0.8, 'horrible': -0.8, 'disgusting': -0.9,
        'unpleasant': -0.6, 'poor': -0.5, 'mediocre': -0.4, 'boring': -0.4,
        'burnt': -0.4, 'overwhelming': -0.4, 'tired': -0.5, 'muddy': -0.5,
        'stale': -0.6, 'acidic': -0.4, 'over-oaked': -0.5, 'imprecise': -0.4,
        'green': -0.4, 'volatile': -0.5, 'dry finish': -0.4,

        # Neutros técnicos
        'tannins': 0.0, 'acidity': 0.0, 'alcohol': 0.0, 'oak': 0.0,
        'barrel': 0.0, 'malolactic': 0.0, 'vintage': 0.0, 'appellation': 0.0,
        'nose': 0.0, 'palate': 0.0, 'finish': 0.0, 'bouquet': 0.0,
        'body': 0.0, 'structure': 0.0, 'texture': 0.0, 'color': 0.0,
        'grape': 0.0, 'blend': 0.0, 'fermentation': 0.0, 'aging': 0.0,
        'winery': 0.0, 'cellar': 0.0
    }

def enhanced_sentiment_analysis(text, wine_lexicon):
    """Análise de sentimento aprimorada com termos específicos de vinho"""
    if pd.isnull(text) or str(text).strip() == '':
        return {
            'polarity': 0.0, 
            'subjectivity': 0.0, 
            'wine_boost': 0.0, 
            'wine_terms_found': 0,
            'enhanced_polarity': 0.0,
            'wine_terms_list': []
        }
    
    text_str = str(text).lower()
    
    # Análise padrão TextBlob
    blob = TextBlob(text_str)
    original_polarity = blob.sentiment.polarity
    original_subjectivity = blob.sentiment.subjectivity
    
    # Análise adicional de termos específicos de vinho
    wine_boost = 0.0
    wine_terms_found = 0
    wine_terms_list = []
    
    for term, score in wine_lexicon.items():
        if term in text_str:
            wine_boost += score
            wine_terms_found += 1
            wine_terms_list.append(term)
    
    if wine_terms_found > 0:
        wine_boost = wine_boost / wine_terms_found
    
    # Combinar análises
    if wine_terms_found > 0:
        boost_factor = min(0.2, wine_terms_found * 0.05)
        wine_contribution = wine_boost * boost_factor
        enhanced_polarity = original_polarity + wine_contribution
        enhanced_polarity = max(-1.0, min(1.0, enhanced_polarity))
    else:
        enhanced_polarity = original_polarity
    
    return {
        'polarity': original_polarity,
        'subjectivity': original_subjectivity,
        'wine_boost': wine_boost,
        'wine_terms_found': wine_terms_found,
        'enhanced_polarity': enhanced_polarity,
        'wine_terms_list': wine_terms_list
    }

def enhanced_sentiment_category(enhanced_polarity):
    """Categorização de sentimento"""
    if enhanced_polarity > 0.1:
        return 'Positive'
    elif enhanced_polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# ====================================================================
# SECTION 7: APLICAR ANÁLISE DE SENTIMENTO
# ====================================================================

def apply_sentiment_analysis(data_clean):
    """Aplicar análise de sentimento aprimorada aos dados"""
    
    wine_lexicon = configure_wine_lexicon()
    
    print("🔬 A aplicar análise de sentimento aprimorada (Inglês)...")
    print(f"📊 A processar {len(data_clean):,} avaliações...")
    
    # Aplicar análise aprimorada
    sentiment_results = data_clean['review'].apply(lambda x: enhanced_sentiment_analysis(x, wine_lexicon))
    
    # Extrair todos os resultados
    data_clean['original_polarity'] = [result['polarity'] for result in sentiment_results]
    data_clean['subjectivity'] = [result['subjectivity'] for result in sentiment_results]
    data_clean['wine_boost'] = [result['wine_boost'] for result in sentiment_results]
    data_clean['wine_terms_found'] = [result['wine_terms_found'] for result in sentiment_results]
    data_clean['enhanced_polarity'] = [result['enhanced_polarity'] for result in sentiment_results]
    data_clean['wine_terms_list'] = [result['wine_terms_list'] for result in sentiment_results]
    
    # Categorização final
    data_clean['sentiment_category'] = data_clean['enhanced_polarity'].apply(enhanced_sentiment_category)
    
    print("✅ Análise de sentimento concluída!")
    
    return data_clean, wine_lexicon

# ====================================================================
# SECTION 8: CALCULAR ESTATÍSTICAS E COMPARAÇÕES
# ====================================================================

def calculate_statistics(data_clean):
    """Calcular estatísticas da análise"""
    
    print("📊 A calcular estatísticas...")
    
    # Avaliações impactadas
    impacted_reviews = data_clean[data_clean['wine_terms_found'] > 0]
    
    # Distribuições de sentimento
    original_categories = data_clean['original_polarity'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    original_dist = original_categories.value_counts()
    enhanced_dist = data_clean['sentiment_category'].value_counts()
    
    # Frequência de termos
    all_wine_terms = []
    for terms_list in data_clean['wine_terms_list']:
        all_wine_terms.extend(terms_list)
    term_frequency = pd.Series(all_wine_terms).value_counts()
    
    print("✅ Estatísticas calculadas")
    
    return {
        'impacted_reviews': impacted_reviews,
        'original_dist': original_dist,
        'enhanced_dist': enhanced_dist,
        'term_frequency': term_frequency
    }

# ====================================================================
# SECTION 9: FUNÇÃO PARA GRÁFICO DE TIPO DE VINHO
# ====================================================================

def generate_wine_type_analysis_chart(data_clean, stats, wine_lexicon, output_dir):
    """Gerar análise específica por tipo de vinho"""
    
    if 'wine_type' not in data_clean.columns:
        print("⚠️  Gráfico 9/9: Dados de tipo de vinho não disponíveis")
        return
    
    print("📊 A criar análise por tipo de vinho...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribuição por tipo de vinho
    wine_type_counts = data_clean['wine_type'].value_counts()
    colors = ['#e74c3c', '#f39c12', '#e91e63', '#9b59b6', '#3498db', '#27ae60']
    ax1.pie(wine_type_counts.values, labels=wine_type_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(wine_type_counts)], startangle=90)
    ax1.set_title('Distribuição por Tipo de Vinho', fontsize=14, fontweight='bold')
    
    # 2. Sentimento médio por tipo
    wine_type_sentiment = data_clean.groupby('wine_type')['enhanced_polarity'].mean().sort_values(ascending=False)
    colors2 = ['#27ae60' if x > 0.1 else '#e74c3c' if x < -0.1 else '#f39c12' 
              for x in wine_type_sentiment.values]
    bars = ax2.bar(range(len(wine_type_sentiment)), wine_type_sentiment.values, color=colors2, alpha=0.8)
    ax2.set_xticks(range(len(wine_type_sentiment)))
    ax2.set_xticklabels(wine_type_sentiment.index, rotation=45, ha='right')
    ax2.set_ylabel('Polaridade Aprimorada Média')
    ax2.set_title('Sentimento Médio por Tipo de Vinho', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # 3. Número de avaliações por tipo
    bars3 = ax3.bar(range(len(wine_type_counts)), wine_type_counts.values, 
                    color='#9b59b6', alpha=0.8)
    ax3.set_xticks(range(len(wine_type_counts)))
    ax3.set_xticklabels(wine_type_counts.index, rotation=45, ha='right')
    ax3.set_ylabel('Número de Avaliações')
    ax3.set_title('Número de Avaliações por Tipo', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Termos de vinho médios por tipo
    wine_type_terms = data_clean.groupby('wine_type')['wine_terms_found'].mean().sort_values(ascending=False)
    bars4 = ax4.bar(range(len(wine_type_terms)), wine_type_terms.values, 
                    color='#e67e22', alpha=0.8)
    ax4.set_xticks(range(len(wine_type_terms)))
    ax4.set_xticklabels(wine_type_terms.index, rotation=45, ha='right')
    ax4.set_ylabel('Termos de Vinho Médios')
    ax4.set_title('Termos de Vinho Médios por Tipo', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🍷 Análise por Tipo de Vinho', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_wine_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 9/9: Análise por tipo de vinho guardado")

# ====================================================================
# SECTION 10: VISUALIZAÇÕES NO TERMINAL - VERSÃO COMPLETA
# ====================================================================

def generate_terminal_visualizations(data_clean, stats, wine_lexicon):
    """Gerar visualizações abrangentes para o terminal (matplotlib)"""
    
    print("🎨 A GERAR VISUALIZAÇÕES ABRANGENTES...")
    print("=" * 60)
    
    # Configurar estilo matplotlib
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    
    output_dir = 'imagens'
    os.makedirs(output_dir, exist_ok=True)
    
    # ================================================================
    # 1. COMPARAÇÃO DE POLARIDADE
    # ================================================================
    print("📊 A criar gráficos de comparação de polaridade...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Polaridade original TextBlob
    sns.histplot(data_clean['original_polarity'], kde=True, alpha=0.7, 
                 color='#3498db', ax=ax1, bins=30)
    ax1.axvline(data_clean['original_polarity'].mean(), color='red', 
                linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_title('TextBlob Original\nDistribuição de Polaridade', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Polaridade', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Polaridade aprimorada
    sns.histplot(data_clean['enhanced_polarity'], kde=True, alpha=0.7, 
                 color='#e74c3c', ax=ax2, bins=30)
    ax2.axvline(data_clean['enhanced_polarity'].mean(), color='red', 
                linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_title('Análise Aprimorada\nDistribuição de Polaridade', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Polaridade', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('🍷 Comparação da Análise de Sentimento de Vinhos', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_polarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 1/9: Comparação de polaridade guardado")

    # ================================================================
    # 2. DISTRIBUIÇÃO DE CATEGORIAS DE SENTIMENTO
    # ================================================================
    print("📊 A criar distribuição de categorias de sentimento...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Categorias de sentimento originais
    original_categories = data_clean['original_polarity'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    original_counts = original_categories.value_counts()
    colors_orig = ['#2ecc71' if cat == 'Positive' else '#e74c3c' if cat == 'Negative' else '#95a5a6' 
                   for cat in original_counts.index]
    
    wedges1, texts1, autotexts1 = ax1.pie(original_counts.values, labels=original_counts.index, 
                                          autopct='%1.1f%%', colors=colors_orig, startangle=90)
    ax1.set_title('TextBlob Original\nDistribuição de Sentimento', fontsize=14, fontweight='bold')
    
    # Categorias de sentimento aprimoradas
    enhanced_counts = data_clean['sentiment_category'].value_counts()
    colors_enh = ['#2ecc71' if cat == 'Positive' else '#e74c3c' if cat == 'Negative' else '#95a5a6' 
                  for cat in enhanced_counts.index]
    
    wedges2, texts2, autotexts2 = ax2.pie(enhanced_counts.values, labels=enhanced_counts.index, 
                                          autopct='%1.1f%%', colors=colors_enh, startangle=90)
    ax2.set_title('Análise Aprimorada\nDistribuição de Sentimento', fontsize=14, fontweight='bold')
    
    plt.suptitle('📊 Comparação da Distribuição de Categorias de Sentimento', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 2/9: Distribuição de sentimento guardado")

    # ================================================================
    # 3. ANÁLISE DE IMPACTO DOS TERMOS DE VINHO
    # ================================================================
    print("📊 A criar análise de impacto dos termos de vinho...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Frequência de termos de vinho
    terms_impact = data_clean['wine_terms_found'].value_counts().sort_index()
    ax1.bar(terms_impact.index, terms_impact.values, color='#8e44ad', alpha=0.7)
    ax1.set_title('Distribuição de Termos de Vinho\nEncontrados por Avaliação', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Número de Termos de Vinho Encontrados', fontsize=12)
    ax1.set_ylabel('Número de Avaliações', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Distribuição do boost de vinho
    wine_boost_data = data_clean[data_clean['wine_boost'] != 0]['wine_boost']
    if len(wine_boost_data) > 0:
        sns.histplot(wine_boost_data, kde=True, alpha=0.7, color='#f39c12', ax=ax2, bins=20)
        ax2.axvline(wine_boost_data.mean(), color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_title('Distribuição do Boost\nde Termos de Vinho', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Boost de Termos de Vinho', fontsize=12)
        ax2.set_ylabel('Frequência', fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Nenhum boost de termos de vinho encontrado', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Distribuição do Boost\nde Termos de Vinho', fontsize=14, fontweight='bold')
    
    plt.suptitle('🍇 Análise de Impacto dos Termos de Vinho', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_wine_terms_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 3/9: Impacto dos termos de vinho guardado")

    # ================================================================
    # 4. TOP TERMOS DE VINHO ENCONTRADOS
    # ================================================================
    print("📊 A criar gráfico dos top termos de vinho...")
    
    if len(stats['term_frequency']) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        top_terms = stats['term_frequency'].head(15)
        colors = ['#27ae60' if wine_lexicon[term] > 0 else '#e74c3c' if wine_lexicon[term] < 0 else '#34495e' 
                  for term in top_terms.index]
        
        bars = ax.barh(range(len(top_terms)), top_terms.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_terms)))
        ax.set_yticklabels(top_terms.index)
        ax.set_xlabel('Frequência', fontsize=12)
        ax.set_title('🍇 Top 15 Termos de Vinho Encontrados nas Avaliações', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adicionar etiquetas de valores nas barras
        for i, (bar, value) in enumerate(zip(bars, top_terms.values)):
            ax.text(value + max(top_terms.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                   str(value), va='center', fontsize=10)
        
        # Adicionar legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#27ae60', alpha=0.8, label='Positivo'),
                          Patch(facecolor='#e74c3c', alpha=0.8, label='Negativo'),
                          Patch(facecolor='#34495e', alpha=0.8, label='Neutro')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_top_wine_terms.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 4/9: Top termos de vinho guardado")
    else:
        print("⚠️  Gráfico 4/9: Nenhum termo de vinho encontrado para mostrar")

    # ================================================================
    # 5. CORRELAÇÃO RATING vs SENTIMENTO
    # ================================================================
    if 'rating' in data_clean.columns and not data_clean['rating'].isna().all():
        print("📊 A criar correlação rating vs sentimento...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de dispersão: Rating vs Polaridade Aprimorada
        valid_data = data_clean.dropna(subset=['rating', 'enhanced_polarity'])
        if len(valid_data) > 0:
            scatter = ax1.scatter(valid_data['rating'], valid_data['enhanced_polarity'], 
                                alpha=0.6, c=valid_data['wine_terms_found'], 
                                cmap='viridis', s=30)
            ax1.set_xlabel('Rating', fontsize=12)
            ax1.set_ylabel('Polaridade Aprimorada', fontsize=12)
            ax1.set_title('Rating vs Sentimento Aprimorado\n(Cor = Termos de Vinho Encontrados)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Termos de Vinho Encontrados')
            
            # Adicionar linha de tendência
            z = np.polyfit(valid_data['rating'], valid_data['enhanced_polarity'], 1)
            p = np.poly1d(z)
            ax1.plot(valid_data['rating'], p(valid_data['rating']), "r--", alpha=0.8, linewidth=2)
        
        # Distribuição de rating por categoria de sentimento
        if 'sentiment_category' in data_clean.columns:
            sentiment_rating = []
            labels = []
            for category in ['Positive', 'Neutral', 'Negative']:
                if category in data_clean['sentiment_category'].values:
                    ratings = data_clean[data_clean['sentiment_category'] == category]['rating'].dropna()
                    if len(ratings) > 0:
                        sentiment_rating.append(ratings)
                        labels.append(f'{category}\n(n={len(ratings)})')
            
            if sentiment_rating:
                bp = ax2.boxplot(sentiment_rating, labels=labels, patch_artist=True)
                colors = ['#2ecc71', '#95a5a6', '#e74c3c']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel('Rating', fontsize=12)
                ax2.set_title('Distribuição de Rating\npor Categoria de Sentimento', 
                             fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        plt.suptitle('⭐ Análise Rating vs Sentimento', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_rating_sentiment_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 5/9: Correlação rating vs sentimento guardado")
    else:
        print("⚠️  Gráfico 5/9: Dados de rating não disponíveis")

    # ================================================================
    # 6. ANÁLISE POR VARIETAL
    # ================================================================
    if 'varietal' in data_clean.columns:
        print("📊 A criar análise por varietal...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top varietals por contagem
        top_varietals = data_clean['varietal'].value_counts().head(15)
        if len(top_varietals) > 0:
            bars1 = ax1.barh(range(len(top_varietals)), top_varietals.values, 
                            color='#9b59b6', alpha=0.8)
            ax1.set_yticks(range(len(top_varietals)))
            ax1.set_yticklabels(top_varietals.index)
            ax1.set_xlabel('Número de Avaliações', fontsize=12)
            ax1.set_title('Top 15 Varietals por Número de Avaliações', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Adicionar etiquetas de valores
            for i, (bar, value) in enumerate(zip(bars1, top_varietals.values)):
                ax1.text(value + max(top_varietals.values) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        str(value), va='center', fontsize=10)
        
        # Top varietals por sentimento
        varietal_sentiment = data_clean.groupby('varietal')['enhanced_polarity'].agg(['mean', 'count'])
        varietal_sentiment = varietal_sentiment[varietal_sentiment['count'] >= 3].sort_values('mean', ascending=False).head(15)
        
        if len(varietal_sentiment) > 0:
            colors2 = ['#27ae60' if x > 0.1 else '#e74c3c' if x < -0.1 else '#f39c12' 
                      for x in varietal_sentiment['mean']]
            bars2 = ax2.barh(range(len(varietal_sentiment)), varietal_sentiment['mean'], 
                            color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(varietal_sentiment)))
            ax2.set_yticklabels(varietal_sentiment.index)
            ax2.set_xlabel('Polaridade Aprimorada Média', fontsize=12)
            ax2.set_title('Top 15 Varietals por Sentimento Médio (mín. 3 avaliações)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Adicionar etiquetas de valores
            for i, (bar, value) in enumerate(zip(bars2, varietal_sentiment['mean'])):
                ax2.text(value + (max(varietal_sentiment['mean']) - min(varietal_sentiment['mean'])) * 0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', fontsize=10)
        
        plt.suptitle('🍇 Análise por Varietal', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_varietal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 6/9: Análise por varietal guardado")
    else:
        print("⚠️  Gráfico 6/9: Dados de varietal não disponíveis")

    # ================================================================
    # 7. ANÁLISE POR PAÍS
    # ================================================================
    if 'country' in data_clean.columns:
        print("📊 A criar análise por país...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top países por contagem
        top_countries = data_clean['country'].value_counts().head(15)
        if len(top_countries) > 0:
            bars1 = ax1.barh(range(len(top_countries)), top_countries.values, 
                            color='#3498db', alpha=0.8)
            ax1.set_yticks(range(len(top_countries)))
            ax1.set_yticklabels(top_countries.index)
            ax1.set_xlabel('Número de Avaliações', fontsize=12)
            ax1.set_title('Top 15 Países por Número de Avaliações', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Adicionar etiquetas de valores
            for i, (bar, value) in enumerate(zip(bars1, top_countries.values)):
                ax1.text(value + max(top_countries.values) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        str(value), va='center', fontsize=10)
        
        # Top países por sentimento
        country_sentiment = data_clean.groupby('country')['enhanced_polarity'].agg(['mean', 'count'])
        country_sentiment = country_sentiment[country_sentiment['count'] >= 5].sort_values('mean', ascending=False).head(15)
        
        if len(country_sentiment) > 0:
            colors2 = ['#27ae60' if x > 0.1 else '#e74c3c' if x < -0.1 else '#f39c12' 
                      for x in country_sentiment['mean']]
            bars2 = ax2.barh(range(len(country_sentiment)), country_sentiment['mean'], 
                            color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(country_sentiment)))
            ax2.set_yticklabels(country_sentiment.index)
            ax2.set_xlabel('Polaridade Aprimorada Média', fontsize=12)
            ax2.set_title('Top 15 Países por Sentimento Médio (mín. 5 avaliações)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Adicionar etiquetas de valores
            for i, (bar, value) in enumerate(zip(bars2, country_sentiment['mean'])):
                ax2.text(value + (max(country_sentiment['mean']) - min(country_sentiment['mean'])) * 0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', fontsize=10)
        
        plt.suptitle('🌍 Análise por País', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_country_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 7/9: Análise por país guardado")
    else:
        print("⚠️  Gráfico 7/9: Dados de país não disponíveis")

    # ================================================================
    # 8. DASHBOARD RESUMO ABRANGENTE
    # ================================================================
    print("📊 A criar dashboard de resumo abrangente...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Estatísticas de resumo
    ax1 = fig.add_subplot(gs[0, :2])
    summary_data = {
        'Total de Avaliações': len(data_clean),
        'Polaridade Original Média': data_clean['original_polarity'].mean(),
        'Polaridade Aprimorada Média': data_clean['enhanced_polarity'].mean(),
        'Avaliações com Termos de Vinho': (data_clean['wine_terms_found'] > 0).sum(),
        'Termos de Vinho Únicos': len(stats['term_frequency']),
        'Avaliações Positivas (%)': (data_clean['sentiment_category'] == 'Positive').mean() * 100,
        'Avaliações Negativas (%)': (data_clean['sentiment_category'] == 'Negative').mean() * 100,
        'Avaliações Neutras (%)': (data_clean['sentiment_category'] == 'Neutral').mean() * 100
    }
    
    ax1.axis('off')
    summary_text = '\n'.join([f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v:,}' 
                             for k, v in summary_data.items()])
    ax1.text(0.1, 0.9, '📊 RESUMO DA ANÁLISE', fontsize=16, fontweight='bold', 
             transform=ax1.transAxes)
    ax1.text(0.1, 0.1, summary_text, fontsize=12, transform=ax1.transAxes, 
             verticalalignment='bottom')
    
    # Comparação de polaridade (mini)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(data_clean['original_polarity'], bins=20, alpha=0.7, label='Original', color='#3498db')
    ax2.hist(data_clean['enhanced_polarity'], bins=20, alpha=0.7, label='Aprimorado', color='#e74c3c')
    ax2.set_title('Comparação da Distribuição de Polaridade', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico circular de sentimento
    ax3 = fig.add_subplot(gs[1, :2])
    sentiment_counts = data_clean['sentiment_category'].value_counts()
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    ax3.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax3.set_title('Distribuição de Sentimento Aprimorado', fontweight='bold')
    
    # Top termos (mini)
    ax4 = fig.add_subplot(gs[1, 2:])
    if len(stats['term_frequency']) > 0:
        top_terms_mini = stats['term_frequency'].head(8)
        ax4.barh(range(len(top_terms_mini)), top_terms_mini.values, color='#9b59b6', alpha=0.8)
        ax4.set_yticks(range(len(top_terms_mini)))
        ax4.set_yticklabels(top_terms_mini.index, fontsize=10)
        ax4.set_title('Top Termos de Vinho', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # Impacto dos termos de vinho
    ax5 = fig.add_subplot(gs[2, :])
    comparison_data = pd.DataFrame({
        'Original': data_clean['original_polarity'],
        'Aprimorado': data_clean['enhanced_polarity'],
        'Termos_Vinho': data_clean['wine_terms_found']
    })
    
    # Criar subgráficos para análise de impacto
    impact_with_terms = data_clean[data_clean['wine_terms_found'] > 0]
    impact_without_terms = data_clean[data_clean['wine_terms_found'] == 0]
    
    if len(impact_with_terms) > 0 and len(impact_without_terms) > 0:
        ax5.hist(impact_without_terms['enhanced_polarity'], bins=15, alpha=0.7, 
                label=f'Sem Termos de Vinho (n={len(impact_without_terms)})', color='#95a5a6')
        ax5.hist(impact_with_terms['enhanced_polarity'], bins=15, alpha=0.7, 
                label=f'Com Termos de Vinho (n={len(impact_with_terms)})', color='#e74c3c')
        ax5.set_title('Impacto dos Termos de Vinho no Sentimento', fontweight='bold')
        ax5.set_xlabel('Polaridade Aprimorada')
        ax5.set_ylabel('Frequência')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle('🍷 Análise de Sentimento de Vinhos - Dashboard Completo', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(f'{output_dir}/08_complete_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 8/9: Dashboard completo guardado")

    # ================================================================
    # 9. ANÁLISE POR TIPO DE VINHO
    # ================================================================
    generate_wine_type_analysis_chart(data_clean, stats, wine_lexicon, output_dir)
    
    # ================================================================
    # 10-11. ANÁLISE NLP AVANÇADA
    # ================================================================
    print("🔬 A executar análise NLP avançada...")
    
    # Análise de descritores por rating
    descriptor_analysis = analyze_descriptors_by_rating(data_clean)
    
    # Análise de preço vs qualidade
    price_analysis = analyze_price_quality_correlation(data_clean)
    
    # Gerar word clouds
    generate_wordcloud_analysis(data_clean, descriptor_analysis)
    
    # Gerar gráficos avançados
    generate_advanced_analysis_charts(data_clean, descriptor_analysis, price_analysis)
    
    # ================================================================
    # RESUMO FINAL ATUALIZADO
    # ================================================================
    print("\n" + "="*60)
    print("🎯 GERAÇÃO DE VISUALIZAÇÕES CONCLUÍDA!")
    print("="*60)
    print(f"📁 Diretório de saída: {output_dir}/")
    print("📊 Gráficos gerados:")
    print("   1. 01_polarity_comparison.png")
    print("   2. 02_sentiment_distribution.png") 
    print("   3. 03_wine_terms_impact.png")
    print("   4. 04_top_wine_terms.png")
    print("   5. 05_rating_sentiment_correlation.png")
    print("   6. 06_varietal_analysis.png")
    print("   7. 07_country_analysis.png")
    print("   8. 08_complete_dashboard.png")
    print("   9. 09_wine_type_analysis.png")
    print("  10. 10_wordcloud_analysis.png")
    print("  11. 11_advanced_nlp_analysis.png")
    print("="*60)
    
    return output_dir

# ====================================================================
# SECTION 11: GUARDAR DADOS E RESUMOS
# ====================================================================

def save_data_and_summaries(data_clean, stats):
    """Guardar dados processados e resumos estatísticos"""
    
    # Criar pasta data se não existir
    os.makedirs('data', exist_ok=True)
    
    # Guardar dados processados na pasta data
    data_file = 'data/wine_sentiment_data_en.csv'
    data_clean.to_csv(data_file, index=False)
    print(f"💾 Dados guardados: {data_file}")
    
    # Criar resumo estatístico enriquecido
    summary_stats = {
        'total_reviews': len(data_clean),
        'avg_original_polarity': data_clean['original_polarity'].mean(),
        'avg_enhanced_polarity': data_clean['enhanced_polarity'].mean(),
        'impacted_reviews': len(stats['impacted_reviews']),
        'impact_percentage': (len(stats['impacted_reviews'])/len(data_clean)*100),
        'wine_terms_found': len(stats['term_frequency']),
        'most_common_varietal': data_clean['varietal'].value_counts().index[0] if len(data_clean) > 0 else 'N/A',
        'most_common_wine_type': data_clean['wine_type'].value_counts().index[0] if 'wine_type' in data_clean.columns and len(data_clean) > 0 else 'N/A',
        'positive_sentiment_pct': (data_clean['sentiment_category'] == 'Positive').sum() / len(data_clean) * 100,
        'negative_sentiment_pct': (data_clean['sentiment_category'] == 'Negative').sum() / len(data_clean) * 100,
        'neutral_sentiment_pct': (data_clean['sentiment_category'] == 'Neutral').sum() / len(data_clean) * 100
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = 'data/wine_analysis_summary_en.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Resumo guardado: {summary_file}")
    
    return summary_stats

# ====================================================================
# SECTION 12: RESUMO NO TERMINAL ENRIQUECIDO
# ====================================================================

def show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon):
    """Mostrar resumo completo e enriquecido da análise no terminal"""
    
    print("\n" + "🍷" * 35)
    print("🎯 ANÁLISE DE SENTIMENTO DE VINHOS - RELATÓRIO COMPLETO")
    print("🍷" * 70)
    print(f"📅 Data da análise: {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}")
    print(f"🌍 Idioma das avaliações: Inglês (EN)")
    print(f"🔬 Motor de análise: TextBlob + Léxico especializado de vinhos")
    print("=" * 70)
    
    print("\n📈 COMPARAÇÃO: TEXTBLOB vs APRIMORADO")
    print("-" * 50)
    print(f"Polaridade média original (TextBlob): {data_clean['original_polarity'].mean():.3f}")
    print(f"Polaridade média aprimorada: {data_clean['enhanced_polarity'].mean():.3f}")
    print(f"Boost médio dos termos de vinho: {data_clean['wine_boost'].mean():.3f}")
    
    print("\n📊 DISTRIBUIÇÃO DE SENTIMENTO")
    print("-" * 50)
    print("TextBlob Original:")
    for sentiment, count in stats['original_dist'].items():
        print(f"- {sentiment}: {count} ({count/len(data_clean)*100:.1f}%)")
    
    print("\nAprimorado com termos de vinho:")
    for sentiment, count in stats['enhanced_dist'].items():
        print(f"- {sentiment}: {count} ({count/len(data_clean)*100:.1f}%)")
    
    if len(stats['term_frequency']) > 0:
        print("\n🍇 TOP 20 TERMOS DE VINHO ENCONTRADOS")
        print("-" * 50)
        top_terms = stats['term_frequency'].head(20)
        for term, count in top_terms.items():
            score = wine_lexicon.get(term, 0)
            sentiment_type = "Positivo" if score > 0 else ("Negativo" if score < 0 else "Neutro")
            print(f"- '{term}': {count} ocorrências (pontuação: {score:+.1f}, {sentiment_type})")
    
    # ================================================================
    # NOVA SECÇÃO: ANÁLISE POR TIPO DE VINHO
    # ================================================================
    if 'wine_type' in data_clean.columns:
        print("\n🍷 ANÁLISE POR TIPO DE VINHO")
        print("-" * 50)
        
        wine_type_analysis = data_clean.groupby('wine_type').agg({
            'enhanced_polarity': ['mean', 'std', 'count'],
            'wine_terms_found': 'mean',
            'rating': 'mean'
        }).round(3)
        
        wine_type_analysis.columns = ['sentiment_mean', 'sentiment_std', 'count', 'wine_terms_avg', 'rating_avg']
        wine_type_analysis = wine_type_analysis.sort_values('sentiment_mean', ascending=False)
        
        print("   Tipo de Vinho         | Avs. | Sentimento | ±Desvio | Termos | Rating")
        print("-" * 70)
        
        for wine_type, row in wine_type_analysis.iterrows():
            sentiment_emoji = "😊" if row['sentiment_mean'] > 0.1 else "😞" if row['sentiment_mean'] < -0.1 else "😐"
            rating_str = f"{row['rating_avg']:5.1f}" if not pd.isna(row['rating_avg']) else "  N/A"
            print(f"   {wine_type[:20]:20} | {row['count']:3.0f} | {row['sentiment_mean']:+7.3f} {sentiment_emoji} | ±{row['sentiment_std']:5.3f} | {row['wine_terms_avg']:5.1f} | {rating_str}")
        
        # Estatísticas adicionais por tipo
        print(f"\n🏆 Tipo com melhor sentimento: {wine_type_analysis.index[0]} ({wine_type_analysis.iloc[0]['sentiment_mean']:+.3f})")
        
        # Distribuição de tipos
        wine_type_dist = data_clean['wine_type'].value_counts()
        print(f"\n📊 Distribuição por tipo:")
        for wine_type, count in wine_type_dist.head(6).items():
            percentage = (count / len(data_clean)) * 100
            print(f"   • {wine_type}: {count} avaliações ({percentage:.1f}%)")
    
    print(f"\n💾 FICHEIROS GERADOS")
    print("-" * 50)
    print("✅ data/wine_sentiment_data_en.csv - Dados completos processados")
    print("✅ data/wine_analysis_summary_en.csv - Resumo estatístico")
    
    # Informações técnicas atualizadas
    print(f"\n🔧 INFORMAÇÕES TÉCNICAS:")
    print(f"   • Colunas geradas: {len(data_clean.columns)} colunas")
    print(f"   • Tamanho do dataset: {data_clean.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"   • Termos especializados: {len(wine_lexicon)} termos")
    print(f"   • Tipos de vinho identificados: {data_clean['wine_type'].nunique()}")
    print(f"   • Avaliações por segundo: ~{len(data_clean)/1:.0f} (estimativa)")
    
    print("\n📖 COMO USAR OS DADOS:")
    print("   import pandas as pd")
    print("   df = pd.read_csv('data/wine_sentiment_data_en.csv')")
    print("   print(df[['wine_type', 'varietal', 'enhanced_polarity', 'sentiment_category']].head())")
    
    print("\n" + "🍷" * 70)
    print("🎯 ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("🍷" * 70)

def show_enhanced_terminal_summary(data_clean, stats, summary_stats, wine_lexicon, 
                                  descriptor_analysis=None, price_analysis=None, nlp_available=True):
    """Resumo aprimorado no terminal com análise NLP (sem gráficos)"""
    
    # Mostrar resumo básico primeiro
    show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
    
    # ================================================================
    # ANÁLISE NLP AVANÇADA (TEXTUAL)
    # ================================================================
    if nlp_available and descriptor_analysis and 'high_rated_words' in descriptor_analysis:
        print("\n" + "🔬" * 35)
        print("🧠 ANÁLISE NLP AVANÇADA - INSIGHTS DE VOCABULÁRIO")
        print("🔬" * 70)
        
        print("\n🏆 TOP 15 PALAVRAS ASSOCIADAS A RATINGS ALTOS")
        print("-" * 70)
        
        top_words = sorted(descriptor_analysis['high_rated_words'].items(), 
                          key=lambda x: x[1]['score'], reverse=True)[:15]
        
        if top_words:
            print("   Palavra          | Score  | Alto% | Médio% | Baixo% | Contagem")
            print("-" * 70)
            
            for word, data in top_words:
                print(f"   {word[:15]:15} | {data['score']:+5.3f} | {data['high_ratio']*100:5.1f} | "
                     f"{data['medium_ratio']*100:6.1f} | {data['low_ratio']*100:6.1f} | {data['count']:8}")
        
        # Estatísticas de análise de palavras
        stats_data = descriptor_analysis.get('stats', {})
        print(f"\n📊 ESTATÍSTICAS DA ANÁLISE:")
        print(f"   • Avaliações com rating alto (≥90): {stats_data.get('high_rated_count', 'N/A')}")
        print(f"   • Avaliações com rating médio (80-89): {stats_data.get('medium_rated_count', 'N/A')}")
        print(f"   • Avaliações com rating baixo (<80): {stats_data.get('low_rated_count', 'N/A')}")
        print(f"   • Palavras distintivas encontradas: {len(descriptor_analysis.get('high_rated_words', {}))}")
    
    # ================================================================
    # ANÁLISE PREÇO vs QUALIDADE (TEXTUAL)
    # ================================================================
    if price_analysis and 'correlation' in price_analysis:
        print("\n" + "💰" * 35)
        print("💎 ANÁLISE PREÇO vs QUALIDADE")
        print("💰" * 70)
        
        correlation = price_analysis['correlation']
        stats_price = price_analysis.get('stats', {})
        
        print(f"\n📈 CORRELAÇÃO PREÇO vs RATING: {correlation:+.3f}")
        
        # Interpretação da correlação
        if correlation > 0.5:
            interpretation = "🟢 Forte correlação positiva - preços altos tendem a ter ratings altos"
        elif correlation > 0.3:
            interpretation = "🟡 Correlação moderada - alguma tendência preço-qualidade"
        elif correlation > 0.1:
            interpretation = "🟠 Correlação fraca - pouca relação preço-qualidade"
        else:
            interpretation = "🔴 Sem correlação significativa - preço não prediz qualidade"
        
        print(f"📊 Interpretação: {interpretation}")
        
        print(f"\n📋 ESTATÍSTICAS:")
        print(f"   • Total de vinhos com preço: {stats_price.get('total_with_price', 'N/A')}")
        
        if 'price_range' in stats_price:
            price_min, price_max = stats_price['price_range']



            print(f"   • Faixa de preços: ${price_min:.2f} - ${price_max:.2f}")
        
        if 'rating_range' in stats_price:
            rating_min, rating_max = stats_price['rating_range']
            print(f"   • Faixa de ratings: {rating_min} - {rating_max}")
        
        # Análise por categoria de preço
        if 'price_analysis' in price_analysis:
            print(f"\n💎 ANÁLISE POR CATEGORIA DE PREÇO:")
            print("-" * 50)
            print("   Categoria    | Rating Médio | Sentimento | Preço Médio")
            print("-" * 50)
            
            price_cats = price_analysis['price_analysis']
            for category in price_cats.index:
                rating_mean = price_cats.loc[category, ('rating', 'mean')]
                sentiment_mean = price_cats.loc[category, ('enhanced_polarity', 'mean')]
                price_mean = price_cats.loc[category, ('price', 'mean')]
                
                sentiment_emoji = "😊" if sentiment_mean > 0.1 else "😞" if sentiment_mean < -0.1 else "😐"
                
                print(f"   {category:12} | {rating_mean:11.1f} | {sentiment_mean:+7.3f} {sentiment_emoji} | ${price_mean:9.2f}")
    
    # ================================================================
    # INSIGHTS E RECOMENDAÇÕES
    # ================================================================
    print("\n" + "🎯" * 35)
    print("💡 INSIGHTS E RECOMENDAÇÕES")
    print("🎯" * 70)
    
    # Análise de sentimento geral
    positive_pct = (data_clean['sentiment_category'] == 'Positive').mean() * 100
    negative_pct = (data_clean['sentiment_category'] == 'Negative').mean() * 100
    
    print(f"\n📊 PANORAMA GERAL:")
    if positive_pct > 60:
        print("🟢 Sentimento predominantemente positivo - boa qualidade geral dos vinhos")
    elif positive_pct > 40:
        print("🟡 Sentimento equilibrado - qualidade variada")
    else:
        print("🔴 Sentimento predominantemente neutro/negativo - considerar melhor curadoria")
    
    # Insights sobre termos de vinho
    avg_wine_terms = data_clean['wine_terms_found'].mean()
    print(f"\n🍇 ANÁLISE DE VOCABULÁRIO:")
    print(f"   • Média de termos de vinho por avaliação: {avg_wine_terms:.1f}")
    
    if avg_wine_terms > 3:
        print("🟢 Avaliações ricas em vocabulário técnico")
    elif avg_wine_terms > 1.5:
        print("🟡 Vocabulário técnico moderado")
    else:
        print("🔴 Vocabulário técnico limitado - avaliações mais simples")
    
    # Recomendações baseadas em tipos de vinho
    if 'wine_type' in data_clean.columns:
        wine_type_sentiment = data_clean.groupby('wine_type')['enhanced_polarity'].mean().sort_values(ascending=False)
        best_type = wine_type_sentiment.index[0]
        best_sentiment = wine_type_sentiment.iloc[0]
        
        print(f"\n🏆 RECOMENDAÇÃO:")
        print(f"   • Melhor categoria: {best_type} (sentimento: {best_sentiment:+.3f})")
        print(f"   • Considere focar nesta categoria para maximizar satisfação")
    
    # Instruções para análise mais profunda
    print(f"\n🔍 PARA ANÁLISE MAIS PROFUNDA:")
    print("   • Execute opção 2 para gráficos detalhados")
    print("   • Execute opção 1 para dashboard interativo")
    print("   • Verifique ficheiros CSV gerados na pasta data/")
    
    print("\n" + "🎯" * 70)
    print("💡 ANÁLISE TEXTUAL COMPLETA CONCLUÍDA!")
    print("🎯" * 70)

# ====================================================================
# SECTION 13: ANÁLISE NLP AVANÇADA
# ====================================================================

def analyze_descriptors_by_rating(data_clean):
    """Analisar quais palavras estão associadas a melhores notas"""
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    print("🔍 A analisar descritores por rating...")
    
    # Dividir avaliações por rating
    if 'rating' not in data_clean.columns:
        print("⚠️  Coluna 'rating' não disponível")
        return {}
    
    # Definir grupos de rating
    high_rated = data_clean[data_clean['rating'] >= 90]['review'].dropna()
    medium_rated = data_clean[(data_clean['rating'] >= 80) & (data_clean['rating'] < 90)]['review'].dropna()
    low_rated = data_clean[data_clean['rating'] < 80]['review'].dropna()
    
    def extract_words(reviews):
        all_words = []
        for review in reviews:
            try:
                words = word_tokenize(str(review).lower())
                words = [word for word in words if word.isalpha() and len(word) > 3 and word not in stop_words]
                all_words.extend(words)
            except:
                continue
        return Counter(all_words)
    
    high_words = extract_words(high_rated)
    medium_words = extract_words(medium_rated)
    low_words = extract_words(low_rated)
    
    # Palavras mais associadas a ratings altos
    high_distinctive = {}
    for word, count in high_words.most_common(100):
        high_ratio = count / len(high_rated) if len(high_rated) > 0 else 0
        medium_ratio = medium_words.get(word, 0) / len(medium_rated) if len(medium_rated) > 0 else 0
        low_ratio = low_words.get(word, 0) / len(low_rated) if len(low_rated) > 0 else 0
        
        if high_ratio > medium_ratio and high_ratio > low_ratio and count >= 10:
            high_distinctive[word] = {
                'count': count,
                'high_ratio': high_ratio,
                'medium_ratio': medium_ratio,
                'low_ratio': low_ratio,
                'score': high_ratio - max(medium_ratio, low_ratio)
            }
    
    print(f"✅ Análise concluída: {len(high_distinctive)} palavras distintivas encontradas")
    
    return {
        'high_rated_words': high_distinctive,
        'high_word_counts': high_words,
        'medium_word_counts': medium_words,
        'low_word_counts': low_words,
        'stats': {
            'high_rated_count': len(high_rated),
            'medium_rated_count': len(medium_rated),
            'low_rated_count': len(low_rated)
        }
    }

def generate_wordcloud_analysis(data_clean, descriptor_analysis):
    """Gerar word clouds comparativos"""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        print("☁️  A gerar word clouds...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Word cloud para vinhos bem avaliados (rating >= 90)
        if 'rating' in data_clean.columns:
            high_rated_text = ' '.join(data_clean[data_clean['rating'] >= 90]['review'].dropna().astype(str))
            if len(high_rated_text) > 100:
                wc_high = WordCloud(width=800, height=400, background_color='white', 
                                   colormap='Greens', max_words=100).generate(high_rated_text)
                axes[0,0].imshow(wc_high, interpolation='bilinear')
                axes[0,0].set_title('📈 Vinhos Bem Avaliados (Rating ≥ 90)', fontsize=14, fontweight='bold')
                axes[0,0].axis('off')
            
            # Word cloud para vinhos mal avaliados (rating < 80)
            low_rated_text = ' '.join(data_clean[data_clean['rating'] < 80]['review'].dropna().astype(str))
            if len(low_rated_text) > 100:
                wc_low = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='Reds', max_words=100).generate(low_rated_text)
                axes[0,1].imshow(wc_low, interpolation='bilinear')
                axes[0,1].set_title('📉 Vinhos Mal Avaliados (Rating < 80)', fontsize=14, fontweight='bold')
                axes[0,1].axis('off')
        
        # Word cloud por sentimento
        positive_text = ' '.join(data_clean[data_clean['sentiment_category'] == 'Positive']['review'].dropna().astype(str))
        if len(positive_text) > 100:
            wc_pos = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Blues', max_words=100).generate(positive_text)
            axes[1,0].imshow(wc_pos, interpolation='bilinear')
            axes[1,0].set_title('😊 Sentimento Positivo', fontsize=14, fontweight='bold')
            axes[1,0].axis('off')
        
        negative_text = ' '.join(data_clean[data_clean['sentiment_category'] == 'Negative']['review'].dropna().astype(str))
        if len(negative_text) > 100:
            wc_neg = WordCloud(width=800, height=400, background_color='white', 
                              colormap='Oranges', max_words=100).generate(negative_text)
            axes[1,1].imshow(wc_neg, interpolation='bilinear')
            axes[1,1].set_title('😞 Sentimento Negativo', fontsize=14, fontweight='bold')
            axes[1,1].axis('off')
        
        plt.suptitle('☁️  Análise de Palavras-Chave por Categoria', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('imagens/10_wordcloud_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 10: Word clouds guardados")
        
    except ImportError:
        print("⚠️  WordCloud não instalado. Instale com: pip install wordcloud")
        print("📊 A gerar análise alternativa...")
        
        # Análise alternativa sem wordcloud
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if descriptor_analysis and 'high_rated_words' in descriptor_analysis:
            # Top palavras em vinhos bem avaliados
            top_high_words = sorted(descriptor_analysis['high_rated_words'].items(), 
                                  key=lambda x: x[1]['score'], reverse=True)[:15]
            
            if top_high_words:
                words, scores = zip(*[(word, data['score']) for word, data in top_high_words])
                ax1.barh(range(len(words)), scores, color='#2ecc71', alpha=0.8)
                ax1.set_yticks(range(len(words)))
                ax1.set_yticklabels(words)
                ax1.set_title('🏆 Palavras Mais Associadas a Ratings Altos', fontweight='bold')
                ax1.set_xlabel('Score de Associação')
                ax1.grid(True, alpha=0.3, axis='x')
        
        # Frequência de termos por categoria de sentimento
        sentiment_terms = {}
        for category in ['Positive', 'Negative', 'Neutral']:
            category_text = ' '.join(data_clean[data_clean['sentiment_category'] == category]['review'].dropna().astype(str))
            words = category_text.lower().split()
            sentiment_terms[category] = len([w for w in words if len(w) > 4])
        
        categories = list(sentiment_terms.keys())
        counts = list(sentiment_terms.values())
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('📊 Distribuição de Palavras por Sentimento', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('imagens/10_word_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Gráfico 10: Análise de palavras guardada")

def analyze_price_quality_correlation(data_clean):
    """Analisar correlação entre preço e qualidade"""
    print("💰 A analisar correlação preço vs qualidade...")
    
    if 'price' not in data_clean.columns or 'rating' not in data_clean.columns:
        print("⚠️  Colunas 'price' ou 'rating' não disponíveis")
        return {}
    
    # Filtrar dados válidos
    valid_data = data_clean.dropna(subset=['price', 'rating'])
    valid_data = valid_data[(valid_data['price'] > 0) & (valid_data['rating'] > 0)]
    
    if len(valid_data) < 10:
        print("⚠️  Dados insuficientes para análise preço vs qualidade")
        return {}
    
    # Calcular correlação
    correlation = valid_data['price'].corr(valid_data['rating'])
    
    # Criar categorias de preço
    price_percentiles = valid_data['price'].quantile([0.25, 0.5, 0.75])
    
    def categorize_price(price):
        if price <= price_percentiles[0.25]:
            return 'Económico'
        elif price <= price_percentiles[0.5]:
            return 'Médio'
        elif price <= price_percentiles[0.75]:
            return 'Premium'
        else:
            return 'Luxo'
    
    valid_data['price_category'] = valid_data['price'].apply(categorize_price)
    
    # Análise por categoria de preço
    price_analysis = valid_data.groupby('price_category').agg({
        'rating': ['mean', 'std', 'count'],
        'enhanced_polarity': ['mean', 'std'],
        'price': ['mean', 'min', 'max']
    }).round(3)
    
    print(f"✅ Correlação preço vs rating: {correlation:.3f}")
    
    return {
        'correlation': correlation,
        'valid_data': valid_data,
        'price_analysis': price_analysis,
        'price_percentiles': price_percentiles,
        'stats': {
            'total_with_price': len(valid_data),
            'price_range': (valid_data['price'].min(), valid_data['price'].max()),
            'rating_range': (valid_data['rating'].min(), valid_data['rating'].max())
        }
    }

def generate_advanced_analysis_charts(data_clean, descriptor_analysis, price_analysis):
    """Gerar gráficos de análise avançada"""
    print("📊 A criar gráficos de análise avançada...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top palavras associadas a ratings altos
    if descriptor_analysis and 'high_rated_words' in descriptor_analysis:
        ax1 = fig.add_subplot(gs[0, :2])
        top_words = sorted(descriptor_analysis['high_rated_words'].items(), 
                          key=lambda x: x[1]['score'], reverse=True)[:12]
        
        if top_words:
            words, data = zip(*top_words)
            scores = [d['score'] for d in data]
            
            bars = ax1.barh(range(len(words)), scores, color='#2ecc71', alpha=0.8)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_title('🏆 Palavras Mais Associadas a Ratings Altos', fontweight='bold')
            ax1.set_xlabel('Score de Associação')
            ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Distribuição de ratings
    ax2 = fig.add_subplot(gs[0, 2])
    if 'rating' in data_clean.columns:
        ratings = data_clean['rating'].dropna()
        ax2.hist(ratings, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_title('📊 Distribuição de Ratings', fontweight='bold')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Frequência')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(ratings.mean(), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # 3. Correlação preço vs qualidade
    if price_analysis and 'valid_data' in price_analysis:
        ax3 = fig.add_subplot(gs[1, :2])
        valid_data = price_analysis['valid_data']
        
        scatter = ax3.scatter(valid_data['price'], valid_data['rating'], 
                             c=valid_data['enhanced_polarity'], cmap='RdYlGn', 
                             alpha=0.6, s=30)
        ax3.set_xlabel('Preço')
        ax3.set_ylabel('Rating')
        ax3.set_title(f'💰 Preço vs Rating (Correlação: {price_analysis["correlation"]:.3f})', 
                     fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Sentimento')
        
        # Linha de tendência
        z = np.polyfit(valid_data['price'], valid_data['rating'], 1)
        p = np.poly1d(z)
        ax3.plot(valid_data['price'], p(valid_data['price']), "r--", alpha=0.8, linewidth=2)
    
    # 4. Análise por categoria de preço
    if price_analysis and 'price_analysis' in price_analysis:
        ax4 = fig.add_subplot(gs[1, 2])
        price_cats = price_analysis['price_analysis']
        
        categories = price_cats.index
        ratings = price_cats[('rating', 'mean')]
        
        bars = ax4.bar(range(len(categories)), ratings, 
                      color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:len(categories)], 
                      alpha=0.8)
        ax4.set_xticks(range(len(categories)))
        ax4.set_xticklabels(categories, rotation=45)
        ax4.set_title('💎 Rating Médio por Categoria de Preço', fontweight='bold')
        ax4.set_ylabel('Rating Médio')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Termos de vinho por categoria de sentimento
    ax5 = fig.add_subplot(gs[2, :])
    sentiment_terms = data_clean.groupby('sentiment_category')['wine_terms_found'].agg(['mean', 'sum', 'count'])
    
    x = np.arange(len(sentiment_terms))
    width = 0.25
    
    bars1 = ax5.bar(x - width, sentiment_terms['mean'], width, label='Média por Avaliação', 
                   color='#3498db', alpha=0.8)
    bars2 = ax5.bar(x, sentiment_terms['sum']/sentiment_terms['sum'].max()*sentiment_terms['mean'].max(), 
                   width, label='Total (Normalizado)', color='#e74c3c', alpha=0.8)
    bars3 = ax5.bar(x + width, sentiment_terms['count']/sentiment_terms['count'].max()*sentiment_terms['mean'].max(), 
                   width, label='Número de Avaliações (Norm.)', color='#2ecc71', alpha=0.8)
    
    ax5.set_xlabel('Categoria de Sentimento')
    ax5.set_ylabel('Termos de Vinho')
    ax5.set_title('🍇 Termos de Vinho por Categoria de Sentimento', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(sentiment_terms.index)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🔬 Análise NLP Avançada - Insights de Qualidade e Vocabulário', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig('imagens/11_advanced_nlp_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico 11: Análise NLP avançada guardada")

# ====================================================================
# SECTION 11: FUNÇÃO PRINCIPAL E EXECUÇÃO
# ====================================================================

def main():
    """Função principal que executa toda a análise"""
    
    print("🚀 ANÁLISE DE SENTIMENTO DE VINHOS")
    print("=" * 60)
    print("🌍 Idioma: Inglês")
    print("🔬 Motor: TextBlob + Léxico específico de vinhos")
    print("=" * 60)
    
    # 1. Processamento de dados
    df = load_and_clean_data("data/avaliacao_vinhos.csv")
    df = normalize_data(df)
    df = enrich_data(df)
    data_clean = finalize_data_cleaning(df)
    
    # 2. Análise de sentimento
    data_clean, wine_lexicon = apply_sentiment_analysis(data_clean)
    
    # 3. Calcular estatísticas
    stats = calculate_statistics(data_clean)
    
    # 4. Guardar dados
    summary_stats = save_data_and_summaries(data_clean, stats)
    
    return data_clean, stats, summary_stats, wine_lexicon

def user_choice():
    """Interface do utilizador para escolha (atualizada)"""
    
    print("\n" + "="*70)
    print("🎨 COMO GOSTARIA DE VER OS RESULTADOS?")
    print("="*70)
    print("1️⃣  🌐 Dashboard Interativo")
    print("    └─ Interface web com filtros e gráficos interativos")
    print()
    print("2️⃣  📊 Visualizações Avançadas (Com gráficos)")
    print("    └─ 11 gráficos incluindo word clouds e análise NLP")
    print()
    print("3️⃣  📋 Análise Completa no Terminal (Sem gráficos)")
    print("    └─ Resumo detalhado com insights NLP e correlações")
    print()
    print("4️⃣  🚪 Sair")
    print("    └─ Apenas processar dados e sair")
    print("="*70)
    
    while True:
        try:
            choice = input("\n👉 Introduza a sua escolha (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Entrada inválida. Introduza 1, 2, 3 ou 4.")
        except KeyboardInterrupt:
            print("\n👋 Programa interrompido pelo utilizador.")
            return "4"
        except:
            print("❌ Entrada inválida. Introduza 1, 2, 3 ou 4.")

import subprocess
import os
import sys

def launch_streamlit_dashboard():
    """Mostrar comando simples para lançar dashboard"""
    print("\n🌐 DASHBOARD INTERATIVO")
    print("=" * 50)

    dashboard_file = "dashboard.py"
    current_path = os.getcwd()

    # Verificar se ficheiro existe
    if not os.path.exists(dashboard_file):
        print(f"❌ Ficheiro '{dashboard_file}' não encontrado!")
        print(f"📁 Pasta atual: {current_path}")
        return False

    # Verificar se pasta translation existe
    if not os.path.exists("translation"):
        print("❌ Pasta translation não encontrada!")
        return False

    print("✅ Todos os ficheiros necessários encontrados!")
    print("\n🚀 Para lançar o dashboard, execute o seguinte comando num novo terminal:")
    print("=" * 60)
    print(f"cd {current_path}")
    print(f"streamlit run {dashboard_file}")
    print("=" * 60)
    print("📱 O dashboard abrirá automaticamente em: http://localhost:8501")
    print("💡 Para parar o dashboard, pressione Ctrl+C no terminal onde está a correr")
    print("\n📋 ALTERNATIVA: Se o comando acima não funcionar, tente:")
    print(f"python -m streamlit run {dashboard_file}")
    
    return True

if __name__ == "__main__":
    import subprocess
    import sys
    import time
    
    # Executar análise principal
    data_clean, stats, summary_stats, wine_lexicon = main()
    
    # Escolha do utilizador
    choice = user_choice()
    
    if choice == "1":
        # DASHBOARD INTERATIVO
        success = launch_streamlit_dashboard()
        if success:
            print("\n📊 Entretanto, aqui está o resumo no terminal:")
            print("-" * 50)
            show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
        else:
            print("\n⚠️  Erro na verificação de ficheiros")
            print("📊 A mostrar resumo no terminal...")
            show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
        
    elif choice == "2":
        # VISUALIZAÇÕES NO TERMINAL (ATUALIZADA)
        print("\n📊 A gerar visualizações avançadas...")
        print("🔬 Incluindo análise NLP, word clouds e correlações")
        try:
            # Executar análise completa com todas as funcionalidades
            output_dir = generate_terminal_visualizations(data_clean, stats, wine_lexicon)
            
            # Verificar se análise NLP foi executada
            nlp_files_generated = [
                f"{output_dir}/10_wordcloud_analysis.png",
                f"{output_dir}/11_advanced_nlp_analysis.png"
            ]
            
            nlp_success = any(os.path.exists(f) for f in nlp_files_generated)
            
            print(f"\n✅ Visualizações guardadas em: {output_dir}/")
            print(f"📊 Total de gráficos gerados: 11")
            
            if nlp_success:
                print("🔬 Análise NLP avançada incluída:")
                print("   • Word clouds por categoria")
                print("   • Análise de palavras associadas a qualidade")
                print("   • Correlações preço vs qualidade")
            else:
                print("⚠️  Análise NLP limitada (instale: pip install nltk wordcloud)")
            
            # Mostrar resumo após gráficos
            show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
            
        except Exception as e:
            print(f"❌ Erro ao gerar gráficos: {e}")
            print("📊 A mostrar resumo detalhado no terminal...")
            show_enhanced_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
    
    elif choice == "3":
        # ANÁLISE MEGA DETALHADA NO TERMINAL
        print("\n📋 A executar análise mega detalhada...")
        print("🔍 Incluindo estatísticas avançadas, correlações e insights")
        
        try:
            # Executar análises NLP se disponível
            descriptor_analysis = analyze_descriptors_by_rating(data_clean)
            price_analysis = analyze_price_quality_correlation(data_clean)
            
            # Mostrar resumo mega detalhado
            show_enhanced_terminal_summary(data_clean, stats, summary_stats, wine_lexicon, 
                                          descriptor_analysis, price_analysis, nlp_available=True)
        except Exception as e:
            print(f"⚠️  Análise NLP limitada: {e}")
            print("📊 A mostrar análise detalhada básica...")
            show_enhanced_terminal_summary(data_clean, stats, summary_stats, wine_lexicon, 
                                          nlp_available=False)
    
    elif choice == "4":
        # MOSTRAR COMANDOS PARA ESTRUTURA DO PROJETO
        print_project_structure_commands()
        
        # Depois mostrar resumo
        print("\n📊 A mostrar resumo da análise atual...")
        show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
    
    elif choice == "5":
        # SAIR
        print("\n👋 Análise concluída!")
        print("📄 Dados guardados na pasta data/")
    
    else:
        print("❌ Opção inválida. A mostrar resumo...")
        show_terminal_summary(data_clean, stats, summary_stats, wine_lexicon)
    
    print("\n🎯 Obrigado por usar o analisador de sentimento de vinhos!")
    
    # Instruções finais se o dashboard não funcionou
    if choice == "1":
        print("\n" + "="*60)
        print("🚀 COMO EXECUTAR O DASHBOARD MANUALMENTE:")
        print("="*60)
        print("1. Abra um novo terminal")
        print("2. Navegue para a pasta do projeto:")
        print("   cd /Users/dariodourado/Desktop/SI")
        print("3. Execute o comando:")
        print("   streamlit run dashboard.py")
        print("4. O dashboard abrirá em: http://localhost:8501")
        print("5. Para parar o dashboard, pressione Ctrl+C no terminal onde está a correr")
        print("="*60)
    
    if choice == "4":
        print("\n📋 PRÓXIMOS PASSOS APÓS CRIAR ESTRUTURA:")
        print("=" * 50)
        print("1. Execute os comandos mostrados acima")
        print("2. Mova o AS-TG-vinhos.py atual para a nova estrutura")
        print("3. Gradualmente refatore o código para usar módulos")
        print("4. Teste cada módulo independentemente")
        print("5. Atualize a documentação conforme necessário")
        print("=" * 50)