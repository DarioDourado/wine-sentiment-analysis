import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json
from datetime import datetime

# ====================================================================
# CONFIGURAÇÃO DA PÁGINA
# ====================================================================

st.set_page_config(
    page_title="🍷 Wine Analytics Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Trabalho desenvolvido por Dario Dourado e Renato Ruivo"
    }
)

# ====================================================================
# SISTEMA DE TRADUÇÕES
# ====================================================================

@st.cache_data
def load_translations(language):
    """Carregar traduções do ficheiro JSON na pasta translation/"""
    try:
        translation_file = f"translation/{language}.json"
        if os.path.exists(translation_file):
            with open(translation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            st.warning(f"Translation file {translation_file} not found. Using English as fallback.")
            with open("translation/en.json", 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading translations: {e}")
        return {
            "main_title": "🍷 Wine Analytics Dashboard",
            "messages": {
                "no_data": "❌ No data found!"
            }
        }

def get_text(translations, key_path, **kwargs):
    """Obter texto traduzido usando caminho de chaves aninhadas"""
    keys = key_path.split('.')
    value = translations
    
    try:
        for key in keys:
            value = value[key]
        
        if kwargs:
            return value.format(**kwargs)
        return value
    except (KeyError, TypeError):
        return key_path  

# ====================================================================
# CSS CUSTOMIZADO
# ====================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #8B0000 0%, #DC143C 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(139, 0, 0, 0.3);
    }
    
    /* Language selector styling */
    .language-selector {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Filter section styling */
    .filter-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #e17055;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ====================================================================

@st.cache_data(show_spinner=False)
def load_wine_data():
    """Carregar dados de vinho com cache otimizado"""
    possible_files = [
        'data/wine_sentiment_data_en.csv',
        'wine_sentiment_data_en.csv',
        'data/avaliacao_vinhos.csv',
        'avaliacao_vinhos.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    return df, file_path
            except Exception as e:
                continue
    
    return pd.DataFrame(), None

@st.cache_data(show_spinner=False)
def load_advanced_analysis_data():
    """Carregar dados de análise avançada se disponíveis"""
    try:
        summary_files = [
            'data/wine_analysis_summary_en.csv',
            'wine_analysis_summary_en.csv'
        ]
        
        for file_path in summary_files:
            if os.path.exists(file_path):
                summary_df = pd.read_csv(file_path)
                return summary_df
        
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def check_nlp_analysis_available():
    """Verificar se análise NLP avançada está disponível"""
    advanced_files = [
        'imagens/10_wordcloud_analysis.png',
        'imagens/11_advanced_nlp_analysis.png'
    ]
    
    return any(os.path.exists(f) for f in advanced_files)

# ====================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ====================================================================

def create_modern_donut_chart(df, column, title, colors=None):
    """Criar gráfico donut moderno com legendas na parte inferior"""
    if column not in df.columns:
        return None
    
    value_counts = df[column].value_counts()
    
    if colors is None:
        colors = px.colors.qualitative.Set3
    
    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index,
        values=value_counts.values,
        hole=0.6,
        marker_colors=colors[:len(value_counts)],
        textinfo='label+percent',
        textfont_size=11,
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        pull=[0.01] * len(value_counts)  # Pequena separação entre fatias
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='#2c3e50'),
            pad=dict(t=10, b=5)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",        
            yanchor="top",          
            y=-0.1,                
            xanchor="center",       
            x=0.5,                   
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=10),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemsizing="constant",
            itemwidth=80,           
            traceorder="normal"
        ),
        height=420,                  
        margin=dict(
            t=50,    
            b=80,    
            l=20,    
            r=20    
        ),
        autosize=True,
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(
        domain=dict(
            x=[0.1, 0.9],    
            y=[0.15, 0.85]  
        ),
        textfont_size=11,
        marker=dict(
            line=dict(color='#FFFFFF', width=1.5)
        )
    )
    
    return fig

def create_modern_bar_chart(df, x_col, y_col, title, orientation='v', top_n=15):
    """Criar gráfico de barras moderno"""
    if x_col not in df.columns:
        return None
    
    if y_col == 'count':
        data = df[x_col].value_counts().head(top_n)
        x_data = data.index
        y_data = data.values
    else:
        grouped = df.groupby(x_col, observed=False)[y_col].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 3].sort_values('mean', ascending=False).head(top_n)
        x_data = grouped[x_col]
        y_data = grouped['mean']
    
    if orientation == 'h':
        fig = px.bar(
            x=y_data,
            y=x_data,
            orientation='h',
            title=title,
            color=y_data,
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.bar(
            x=x_data,
            y=y_data,
            title=title,
            color=y_data,
            color_continuous_scale='Viridis'
        )
    
    fig.update_layout(
        title=dict(
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_comparison_chart(df, translations):
    """Criar gráfico de comparação de polaridade"""
    if 'original_polarity' not in df.columns or 'enhanced_polarity' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            get_text(translations, 'visualizations.original_textblob'),
            get_text(translations, 'visualizations.enhanced_analysis')
        ),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Histograma original
    fig.add_trace(
        go.Histogram(
            x=df['original_polarity'],
            name='Original',
            marker_color='#3498db',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=1
    )
    
    # Histograma aprimorado
    fig.add_trace(
        go.Histogram(
            x=df['enhanced_polarity'],
            name='Enhanced',
            marker_color='#e74c3c',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=get_text(translations, 'visualizations.polarity_comparison'),
        title_x=0.5,
        title_font=dict(size=20, color='#2c3e50'),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col, title):
    """Criar scatter plot moderno"""
    if not all(col in df.columns for col in [x_col, y_col, color_col]):
        return None
    
    valid_data = df.dropna(subset=[x_col, y_col, color_col])
    
    if len(valid_data) == 0:
        return None
    
    fig = px.scatter(
        valid_data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=color_col,
        size_max=15,
        title=title,
        color_continuous_scale='Viridis',
        hover_data=[col for col in ['wine_type', 'varietal', 'country'] if col in df.columns]
    )
    
    try:
        z = np.polyfit(valid_data[x_col], valid_data[y_col], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_data[x_col],
            y=p(valid_data[x_col]),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash', width=2)
        ))
    except:
        pass
    
    fig.update_layout(
        title=dict(
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_word_frequency_chart(df, translations):
    """Criar gráfico de frequência de palavras de vinho"""
    if 'wine_terms_found' not in df.columns:
        return None
    
    terms_dist = df['wine_terms_found'].value_counts().sort_index()
    
    fig = px.bar(
        x=terms_dist.index,
        y=terms_dist.values,
        title=get_text(translations, 'visualizations.wine_terms_distribution'),
        labels={
            'x': get_text(translations, 'visualizations.terms_found'),
            'y': get_text(translations, 'visualizations.review_count')
        },
        color=terms_dist.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18, color='#2c3e50')),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_price_quality_correlation(df, translations):
    """Criar análise de correlação preço vs qualidade"""
    if 'price' not in df.columns or 'rating' not in df.columns:
        return None
    
    # Filtrar dados válidos
    valid_data = df.dropna(subset=['price', 'rating', 'enhanced_polarity'])
    
    if len(valid_data) < 10:
        return None
    
    # Calcular correlação
    correlation = valid_data['price'].corr(valid_data['rating'])
    
    fig = px.scatter(
        valid_data,
        x='price',
        y='rating',
        color='enhanced_polarity',
        size='wine_terms_found',
        size_max=15,
        title=f"{get_text(translations, 'visualizations.price_quality_correlation')} (r={correlation:.3f})",
        color_continuous_scale='RdYlGn',
        hover_data=['wine_type', 'varietal'] if 'wine_type' in df.columns else None
    )
    
    # Adicionar linha de tendência
    try:
        z = np.polyfit(valid_data['price'], valid_data['rating'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_data['price'],
            y=p(valid_data['price']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash', width=2)
        ))
    except:
        pass
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=18, color='#2c3e50')),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_rating_distribution_advanced(df, translations):
    """Criar análise avançada de distribuição de ratings"""
    if 'rating' not in df.columns:
        return None
    
    valid_ratings = df['rating'].dropna()
    
    if len(valid_ratings) == 0:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            get_text(translations, 'visualizations.rating_histogram'),
            get_text(translations, 'visualizations.rating_by_sentiment')
        ),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Histograma de ratings
    fig.add_trace(
        go.Histogram(
            x=valid_ratings,
            name='Rating Distribution',
            marker_color='#3498db',
            opacity=0.7,
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # Box plot por categoria de sentimento
    if 'sentiment_category' in df.columns:
        for i, category in enumerate(['Positive', 'Neutral', 'Negative']):
            if category in df['sentiment_category'].values:
                category_ratings = df[df['sentiment_category'] == category]['rating'].dropna()
                if len(category_ratings) > 0:
                    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
                    fig.add_trace(
                        go.Box(
                            y=category_ratings,
                            name=category,
                            marker_color=colors[category],
                            x=[category] * len(category_ratings)
                        ),
                        row=1, col=2
                    )
    
    fig.update_layout(
        title_text=get_text(translations, 'visualizations.rating_analysis'),
        title_x=0.5,
        title_font=dict(size=20, color='#2c3e50'),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def display_advanced_images(translations):
    """Exibir imagens de análise avançada se disponíveis"""
    advanced_images = {
        'imagens/10_wordcloud_analysis.png': get_text(translations, 'visualizations.wordcloud_analysis'),
        'imagens/11_advanced_nlp_analysis.png': get_text(translations, 'visualizations.advanced_nlp_analysis')
    }
    
    available_images = [(path, title) for path, title in advanced_images.items() if os.path.exists(path)]
    
    if available_images:
        st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.advanced_analysis")}</h3>', unsafe_allow_html=True)
        
        for i in range(0, len(available_images), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(available_images):
                    image_path, image_title = available_images[i + j]
                    with col:
                        st.markdown(f"**{image_title}**")
                        st.image(image_path, use_container_width=True)
    
    return len(available_images) > 0

# ====================================================================
# INTERFACE PRINCIPAL
# ====================================================================

def main():
    # ================================================================
    # SELETOR DE IDIOMA NO TOPO
    # ================================================================
    
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col3:
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        language = st.selectbox(
            "🌐",
            ["en", "pt"],
            format_func=lambda x: "🇺🇸 EN" if x == "en" else "🇵🇹 PT",
            key="language_selector"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Carregar traduções
    translations = load_translations(language)
    
    # ================================================================
    # HEADER PRINCIPAL
    # ================================================================
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{get_text(translations, 'main_title')}</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            {get_text(translations, 'main_subtitle')}
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
            {get_text(translations, 'main_description')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================================================
    # CARREGAR DADOS
    # ================================================================
    
    with st.spinner(get_text(translations, 'messages.loading_data')):
        df, file_path = load_wine_data()
    
    if df.empty:
        st.error(get_text(translations, 'messages.no_data'))
        st.stop()
        return
    
    # ================================================================
    # SIDEBAR - FILTROS AVANÇADOS
    # ================================================================
    
    st.sidebar.markdown(f"""
    <div class="filter-section">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            {get_text(translations, 'filters.title')}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset filters button
    if st.sidebar.button(get_text(translations, 'filters.reset'), type="primary"):
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    # Backup original dataframe
    original_df = df.copy()
    
    # Filter 1: Wine Type
    if 'wine_type' in df.columns:
        wine_types = [get_text(translations, 'common.all')] + sorted(df['wine_type'].unique().tolist())
        selected_wine_type = st.sidebar.selectbox(
            get_text(translations, 'filters.wine_type'),
            wine_types,
            help=get_text(translations, 'filters.wine_type_help')
        )
        
        if selected_wine_type != get_text(translations, 'common.all'):
            df = df[df['wine_type'] == selected_wine_type]
    
    # Filter 2: Sentiment Category
    if 'sentiment_category' in df.columns:
        sentiments = [get_text(translations, 'common.all')] + sorted(df['sentiment_category'].unique().tolist())
        selected_sentiment = st.sidebar.selectbox(
            get_text(translations, 'filters.sentiment_category'),
            sentiments,
            help=get_text(translations, 'filters.sentiment_help')
        )
        
        if selected_sentiment != get_text(translations, 'common.all'):
            df = df[df['sentiment_category'] == selected_sentiment]
    
    # Filter 3: Country
    if 'country' in df.columns:
        top_countries = df['country'].value_counts().head(20).index.tolist()
        countries = [get_text(translations, 'common.all')] + sorted(top_countries)
        selected_country = st.sidebar.selectbox(
            get_text(translations, 'filters.country'),
            countries,
            help=get_text(translations, 'filters.country_help')
        )
        
        if selected_country != get_text(translations, 'common.all'):
            df = df[df['country'] == selected_country]
    
    # Filter 4: Polarity Range
    if 'enhanced_polarity' in df.columns:
        min_pol = float(df['enhanced_polarity'].min())
        max_pol = float(df['enhanced_polarity'].max())
        
        polarity_range = st.sidebar.slider(
            get_text(translations, 'filters.polarity_range'),
            min_pol, max_pol,
            (min_pol, max_pol),
            step=0.01,
            help=get_text(translations, 'filters.polarity_help')
        )
        
        df = df[
            (df['enhanced_polarity'] >= polarity_range[0]) & 
            (df['enhanced_polarity'] <= polarity_range[1])
        ]
    
    # Filter 5: Rating Range
    if 'rating' in df.columns and not df['rating'].isna().all():
        valid_ratings = df['rating'].dropna()
        if len(valid_ratings) > 0:
            min_rating = int(valid_ratings.min())
            max_rating = int(valid_ratings.max())
            
            rating_range = st.sidebar.slider(
                get_text(translations, 'filters.rating_range'),
                min_rating, max_rating,
                (min_rating, max_rating),
                help=get_text(translations, 'filters.rating_help')
            )
            
            df = df[
                (df['rating'] >= rating_range[0]) & 
                (df['rating'] <= rating_range[1])
            ]
    
    # Filter 6: Wine Terms
    if 'wine_terms_found' in df.columns:
        min_terms = int(df['wine_terms_found'].min())
        max_terms = int(df['wine_terms_found'].max())
        
        terms_range = st.sidebar.slider(
            get_text(translations, 'filters.wine_terms'),
            min_terms, max_terms,
            (min_terms, max_terms),
            help=get_text(translations, 'filters.wine_terms_help')
        )
        
        df = df[
            (df['wine_terms_found'] >= terms_range[0]) & 
            (df['wine_terms_found'] <= terms_range[1])
        ]
    
    # Filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_text(translations, 'filters.filtered_dataset')}")
    st.sidebar.metric(get_text(translations, 'filters.records'), f"{len(df):,}")
    st.sidebar.metric(get_text(translations, 'filters.percent_total'), f"{len(df)/len(original_df)*100:.1f}%")
    
    if len(df) == 0:
        st.warning(get_text(translations, 'messages.no_data_filters'))
        return
    
    # ================================================================
    # MÉTRICAS PRINCIPAIS EXPANDIDAS
    # ================================================================

    st.markdown(f'<h2 class="section-header">{get_text(translations, "metrics.title")}</h2>', unsafe_allow_html=True)

    # Row 1: Métricas básicas
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            get_text(translations, 'metrics.total_reviews'),
            f"{len(df):,}",
            delta=f"{len(df) - len(original_df):+,}" if len(df) != len(original_df) else None
        )

    with col2:
        if 'enhanced_polarity' in df.columns:
            avg_sentiment = df['enhanced_polarity'].mean()
            orig_avg = original_df['enhanced_polarity'].mean() if len(df) != len(original_df) else None
            delta = f"{avg_sentiment - orig_avg:+.3f}" if orig_avg else None
            st.metric(
                get_text(translations, 'metrics.avg_sentiment'),
                f"{avg_sentiment:.3f}",
                delta=delta
            )

    with col3:
        if 'sentiment_category' in df.columns:
            positive_pct = (df['sentiment_category'] == 'Positive').mean() * 100
            st.metric(
                get_text(translations, 'metrics.positive_percent'),
                f"{positive_pct:.1f}%"
            )

    with col4:
        if 'wine_terms_found' in df.columns:
            total_terms = df['wine_terms_found'].sum()
            avg_terms = df['wine_terms_found'].mean()
            st.metric(
                get_text(translations, 'metrics.wine_terms_metric'),
                f"{total_terms:,}",
                delta=f"Avg: {avg_terms:.1f}"
            )

    with col5:
        if 'wine_type' in df.columns:
            unique_types = df['wine_type'].nunique()
            st.metric(
                get_text(translations, 'metrics.wine_types_metric'),
                f"{unique_types}"
            )

    # Row 2: Métricas avançadas
    if check_nlp_analysis_available():
        st.markdown("### 🔬 Advanced NLP Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'rating' in df.columns:
                valid_ratings = df['rating'].dropna()
                if len(valid_ratings) > 0:
                    st.metric(
                        "📊 Avg Rating",
                        f"{valid_ratings.mean():.1f}",
                        delta=f"Range: {valid_ratings.min():.0f}-{valid_ratings.max():.0f}"
                    )
        
        with col2:
            if 'price' in df.columns:
                valid_prices = df['price'].dropna()
                if len(valid_prices) > 0:
                    st.metric(
                        "💰 Avg Price",
                        f"${valid_prices.mean():.0f}",
                        delta=f"Range: ${valid_prices.min():.0f}-${valid_prices.max():.0f}"
                    )
        
        with col3:
            if 'wine_boost' in df.columns:
                boosted_reviews = df[df['wine_boost'] != 0]
                if len(boosted_reviews) > 0:
                    st.metric(
                        "🚀 Boosted Reviews",
                        f"{len(boosted_reviews):,}",
                        delta=f"{len(boosted_reviews)/len(df)*100:.1f}% of total"
                    )
        
        with col4:
            if 'country' in df.columns:
                unique_countries = df['country'].nunique()
                st.metric(
                    "🌍 Countries",
                    f"{unique_countries}",
                    delta=f"Most: {df['country'].value_counts().index[0]}"
                )
    
    # ================================================================
    # GRÁFICOS PRINCIPAIS
    # ================================================================
    
    st.markdown(f'<h2 class="section-header">{get_text(translations, "visualizations.title")}</h2>', unsafe_allow_html=True)
    
    # Row 1: Sentiment Distribution e Polarity Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sentiment_category' in df.columns:
            fig_sentiment = create_modern_donut_chart(
                df, 
                'sentiment_category', 
                get_text(translations, 'visualizations.sentiment_distribution'),
                colors=['#2ecc71', '#95a5a6', '#e74c3c']
            )
            if fig_sentiment:
                st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        fig_comparison = create_comparison_chart(df, translations)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Row 2: Wine Type Analysis
    if 'wine_type' in df.columns:
        st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.wine_type_analysis")}</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_wine_type = create_modern_donut_chart(
                df,
                'wine_type',
                get_text(translations, 'visualizations.wine_type_distribution')
            )
            if fig_wine_type:
                st.plotly_chart(fig_wine_type, use_container_width=True)
        
        with col2:
            if 'enhanced_polarity' in df.columns:
                fig_wine_sentiment = create_modern_bar_chart(
                    df,
                    'wine_type',
                    'enhanced_polarity',
                    get_text(translations, 'visualizations.avg_sentiment_wine_type'),
                    orientation='v'
                )
                if fig_wine_sentiment:
                    st.plotly_chart(fig_wine_sentiment, use_container_width=True)
    
    # Row 3: Geographic Analysis
    if 'country' in df.columns:
        st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.geographic_analysis")}</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_countries = create_modern_bar_chart(
                df,
                'country',
                'count',
                get_text(translations, 'visualizations.top_countries_count'),
                orientation='h',
                top_n=10
            )
            if fig_countries:
                st.plotly_chart(fig_countries, use_container_width=True)
        
        with col2:
            if 'enhanced_polarity' in df.columns:
                fig_country_sentiment = create_modern_bar_chart(
                    df,
                    'country',
                    'enhanced_polarity',
                    get_text(translations, 'visualizations.top_countries_sentiment'),
                    orientation='h',
                    top_n=10
                )
                if fig_country_sentiment:
                    st.plotly_chart(fig_country_sentiment, use_container_width=True)
    
    # Row 4: Varietal Analysis
    if 'varietal' in df.columns:
        st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.varietal_analysis")}</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_varietals = create_modern_bar_chart(
                df,
                'varietal',
                'count',
                get_text(translations, 'visualizations.top_varietals_count'),
                orientation='h',
                top_n=15
            )
            if fig_varietals:
                st.plotly_chart(fig_varietals, use_container_width=True)
        
        with col2:
            if 'enhanced_polarity' in df.columns:
                fig_varietal_sentiment = create_modern_bar_chart(
                    df,
                    'varietal',
                    'enhanced_polarity',
                    get_text(translations, 'visualizations.top_varietals_sentiment'),
                    orientation='h',
                    top_n=15
                )
                if fig_varietal_sentiment:
                    st.plotly_chart(fig_varietal_sentiment, use_container_width=True)
    
    # Row 5: Rating vs Sentiment Correlation - REMOVIDO
    # if 'rating' in df.columns and 'enhanced_polarity' in df.columns:
    #     st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.rating_sentiment_correlation")}</h3>', unsafe_allow_html=True)
    #     
    #     fig_scatter = create_scatter_plot(
    #         df,
    #         'rating',
    #         'enhanced_polarity',
    #         'wine_terms_found',
    #         get_text(translations, 'visualizations.rating_vs_sentiment')
    #     )
    #     if fig_scatter:
    #         st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ================================================================
    # ANÁLISE AVANÇADA NLP (REFORMULADA)
    # ================================================================

    if check_nlp_analysis_available():
        st.markdown(f'<h2 class="section-header">🔬 {get_text(translations, "advanced.title", default="Advanced NLP Analysis")}</h2>', unsafe_allow_html=True)
        
        # Tabs para diferentes análises
        tab1, tab2, tab3, tab4 = st.tabs([
            get_text(translations, "tabs.wine_terms"),
            get_text(translations, "tabs.price_quality"), 
            get_text(translations, "tabs.rating_distribution"),
            get_text(translations, "tabs.advanced_charts")
        ])
        
        with tab1:
            st.markdown("#### 🍇 Wine Terms Impact Analysis")
            
            if 'wine_terms_found' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de distribuição de termos
                    fig_terms = create_word_frequency_chart(df, translations)
                    if fig_terms:
                        st.plotly_chart(fig_terms, use_container_width=True)
                    
                    # Estatísticas principais
                    st.markdown("##### 📈 Key Statistics")
                    terms_stats = {
                        "Total wine terms": df['wine_terms_found'].sum(),
                        "Average per review": f"{df['wine_terms_found'].mean():.2f}",
                        "Reviews with terms": f"{(df['wine_terms_found'] > 0).sum():,} ({(df['wine_terms_found'] > 0).mean()*100:.1f}%)",
                        "Maximum in one review": df['wine_terms_found'].max()
                    }
                    
                    for stat, value in terms_stats.items():
                        st.metric(stat, value)
                
                with col2:
                    # Análise de impacto comparativo
                    if 'wine_boost' in df.columns and 'enhanced_polarity' in df.columns:
                        st.markdown("##### 🚀 Sentiment Impact Comparison")
                        
                        with_terms = df[df['wine_terms_found'] > 0]
                        without_terms = df[df['wine_terms_found'] == 0]
                        
                        if len(with_terms) > 0 and len(without_terms) > 0:
                            # Criar comparação visual
                            comparison_data = pd.DataFrame({
                                'Category': ['With Wine Terms', 'Without Wine Terms'],
                                'Count': [len(with_terms), len(without_terms)],
                                'Avg_Sentiment': [with_terms['enhanced_polarity'].mean(), without_terms['enhanced_polarity'].mean()],
                                'Avg_Boost': [with_terms['wine_boost'].mean() if 'wine_boost' in with_terms.columns else 0, 0]
                            })
                            
                            fig_impact = px.bar(
                                comparison_data,
                                x='Category',
                                y='Avg_Sentiment',
                                color='Avg_Boost',
                                title="Sentiment Enhancement by Wine Terms",
                                color_continuous_scale='Viridis',
                                text='Avg_Sentiment'
                            )
                            
                            fig_impact.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig_impact.update_layout(
                                title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig_impact, use_container_width=True)
                            
                            # Tabela de comparação
                            impact_summary = pd.DataFrame({
                                'Metric': ['Reviews Count', 'Avg Sentiment', 'Sentiment Boost', 'Std Deviation'],
                                'With Wine Terms': [
                                    f"{len(with_terms):,}",
                                    f"{with_terms['enhanced_polarity'].mean():.3f}",
                                    f"{with_terms['wine_boost'].mean():.3f}" if 'wine_boost' in with_terms.columns else "0.000",
                                    f"{with_terms['enhanced_polarity'].std():.3f}"
                                ],
                                'Without Wine Terms': [
                                    f"{len(without_terms):,}",
                                    f"{without_terms['enhanced_polarity'].mean():.3f}",
                                    "0.000",
                                    f"{without_terms['enhanced_polarity'].std():.3f}"
                                ]
                            })
                            
                            st.dataframe(impact_summary, use_container_width=True)
        
        with tab2:
            st.markdown("#### 💎 Price vs Quality Correlation Analysis")
            
            # Análise preço vs qualidade
            fig_price_quality = create_price_quality_correlation(df, translations)
            if fig_price_quality:
                st.plotly_chart(fig_price_quality, use_container_width=True)
                
                # Estatísticas de correlação
                if 'price' in df.columns and 'rating' in df.columns:
                    valid_data = df.dropna(subset=['price', 'rating'])
                    if len(valid_data) >= 10:
                        correlation = valid_data['price'].corr(valid_data['rating'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 Correlation", f"{correlation:.3f}")
                        
                        with col2:
                            interpretation = ""
                            color = ""
                            if correlation > 0.5:
                                interpretation = "Strong +"
                                color = "🟢"
                            elif correlation > 0.3:
                                interpretation = "Moderate"
                                color = "🟡"
                            elif correlation > 0.1:
                                interpretation = "Weak"
                                color = "🟠"
                            else:
                                interpretation = "None"
                                color = "🔴"
                            st.metric("🎯 Strength", f"{color} {interpretation}")
                        
                        with col3:
                            st.metric("📈 Data Points", f"{len(valid_data):,}")
                        
                        with col4:
                            price_range = valid_data['price'].max() - valid_data['price'].min()
                            st.metric("💰 Price Range", f"${price_range:.0f}")
                        
                        # Análise por categoria de preço
                        if len(valid_data) > 50:
                            st.markdown("##### 💎 Price Category Analysis")
                            
                            # Criar quartis de preço - usando .copy() para evitar warning
                            price_data = valid_data.copy()
                            price_data['price_quartile'] = pd.qcut(
                                price_data['price'], 
                                4, 
                                labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
                            )
                            
                            price_analysis = price_data.groupby('price_quartile', observed=False).agg({
                                'rating': ['count', 'mean', 'std'],
                                'enhanced_polarity': ['mean', 'std'],
                                'price': ['mean', 'min', 'max']
                            }).round(3)
                            
                            # Flatten column names
                            price_analysis.columns = ['_'.join(col).strip() for col in price_analysis.columns]
                            price_analysis = price_analysis.reset_index()
                            
                            st.dataframe(price_analysis, use_container_width=True)
            else:
                st.info("💡 Price vs Quality analysis requires both price and rating columns in the dataset.")
        
        with tab3:
            st.markdown("#### ⭐ Advanced Rating Distribution Analysis")
            
            # Distribuição de ratings avançada
            fig_rating_dist = create_rating_distribution_advanced(df, translations)
            if fig_rating_dist:
                st.plotly_chart(fig_rating_dist, use_container_width=True)
            
            # Análise detalhada por categoria de sentimento
            if 'rating' in df.columns and 'sentiment_category' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 📊 Rating Statistics by Sentiment")
                    rating_analysis = df.groupby('sentiment_category', observed=False)['rating'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
                    rating_analysis.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
                    st.dataframe(rating_analysis, use_container_width=True)
                
                with col2:
                    # Distribuição percentual
                    if 'wine_type' in df.columns:
                        st.markdown("##### 🍷 Rating Distribution by Wine Type")
                        wine_rating = df.groupby(['wine_type', 'sentiment_category'], observed=False).size().unstack(fill_value=0)
                        wine_rating_pct = wine_rating.div(wine_rating.sum(axis=1), axis=0) * 100
                        st.dataframe(wine_rating_pct.round(1), use_container_width=True)
        
        with tab4:
            st.markdown("#### 🖼️ Generated Analysis Charts")
            
            # Verificar e exibir imagens de análise avançada
            advanced_images = {
                'imagens/10_wordcloud_analysis.png': {
                    'title': '☁️ Word Cloud Comparative Analysis',
                    'description': 'Word clouds comparing vocabulary in high-rated vs low-rated wines, and positive vs negative sentiment reviews.'
                },
                'imagens/11_advanced_nlp_analysis.png': {
                    'title': '🔬 Advanced NLP Insights',
                    'description': 'Comprehensive analysis including top words associated with quality ratings, price correlations, and sentiment distributions.'
                }
            }
            
            images_found = 0
            for image_path, image_info in advanced_images.items():
                if os.path.exists(image_path):
                    st.markdown(f"##### {image_info['title']}")
                    st.markdown(f"*{image_info['description']}*")
                    st.image(image_path, use_container_width=True)  # Corrigido aqui
                    st.markdown("---")
                    images_found += 1
            
            if images_found == 0:
                st.info("""
                🖼️ **Advanced analysis charts will appear here after running the full analysis.**
                
                **To generate these charts:**
                1. Run `python AS-TG-vinhos.py`
                2. Choose option 2 (Terminal Visualizations)
                3. Charts will be automatically saved and displayed here
                
                **Generated charts include:**
                - Word cloud comparisons (high vs low ratings)
                - Advanced NLP analysis with vocabulary insights
                - Price vs quality correlation visualizations
                - Wine term impact analysis
                """)
            else:
                st.success(f"✅ {images_found} advanced analysis charts loaded successfully!")

    else:
        st.info("""
        🔬 **Advanced NLP analysis not yet available.**
        
        Run the main analysis script first to unlock advanced features:
        - Wine vocabulary analysis
        - Price vs quality correlations  
        - Rating prediction insights
        - Word cloud visualizations
        """)

    # ================================================================
    # INSIGHTS E ESTATÍSTICAS (REFORMULADO)
    # ================================================================
    
    st.markdown(f'<h2 class="section-header">{get_text(translations, "insights.title")}</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>{get_text(translations, "insights.dataset_overview")}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas essenciais do dataset
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Columns", f"{len(df.columns)}")
            
        with col_b:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            st.metric("Data Types", f"{len(df.dtypes.unique())}")

    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>{get_text(translations, "insights.data_quality")}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Verificações de qualidade
        col_a, col_b = st.columns(2)
        
        with col_a:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_values:,}")
            
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicate_rows:,}")
            
        with col_b:
            if 'enhanced_polarity' in df.columns:
                outliers = len(df[(df['enhanced_polarity'] < -2) | (df['enhanced_polarity'] > 2)])
                st.metric("Sentiment Outliers", f"{outliers:,}")
            
            if 'sentiment_category' in df.columns:
                balance = df['sentiment_category'].value_counts()
                most_common_pct = (balance.iloc[0] / len(df)) * 100
                st.metric("Class Balance", f"{most_common_pct:.1f}% dominant")

    # Resumo de insights principais
    if 'enhanced_polarity' in df.columns and 'sentiment_category' in df.columns:
        st.markdown(f"#### {get_text(translations, 'insights.key_insights')}")
        
        positive_pct = (df['sentiment_category'] == 'Positive').mean() * 100
        avg_sentiment = df['enhanced_polarity'].mean()
        
        insights = []
        
        if positive_pct > 60:
            insights.append("🟢 **Predominantly positive sentiment** - High wine quality indicators")
        elif positive_pct > 40:
            insights.append("🟡 **Balanced sentiment distribution** - Mixed quality reviews")
        else:
            insights.append("🔴 **Lower positive sentiment** - Quality concerns may exist")
        
        if 'wine_terms_found' in df.columns:
            avg_terms = df['wine_terms_found'].mean()
            if avg_terms > 3:
                insights.append("🍇 **Rich wine vocabulary** - Expert-level reviews detected")
            elif avg_terms > 1.5:
                insights.append("🍇 **Moderate wine vocabulary** - Semi-professional reviews")
            else:
                insights.append("🍇 **Basic wine vocabulary** - Casual reviewer language")
        
        if 'rating' in df.columns and 'enhanced_polarity' in df.columns:
            correlation = df[['rating', 'enhanced_polarity']].corr().iloc[0, 1]
            if correlation > 0.5:
                insights.append("📈 **Strong rating-sentiment correlation** - Consistent quality perception")
            elif correlation > 0.3:
                insights.append("📈 **Moderate rating-sentiment correlation** - Generally aligned reviews")
            else:
                insights.append("📈 **Weak rating-sentiment correlation** - Mixed review patterns")
        
        for insight in insights:
            st.markdown(insight)

    # ================================================================
    # CONCLUSÃO SIMPLIFICADA
    # ================================================================

    st.markdown(f'<h2 class="section-header">{get_text(translations, "insights.analysis_summary")}</h2>', unsafe_allow_html=True)

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(f"### {get_text(translations, 'insights.what_discovered')}")
        
        st.markdown("""
        This wine sentiment analysis reveals patterns in wine reviews through:
        - **Enhanced sentiment analysis** with wine-specific vocabulary
        - **Quality correlations** between ratings and sentiment
        - **Geographic and varietal insights** 
        - **Wine terminology impact** on review sentiment
        """)

    with summary_col2:
        st.markdown(f"### {get_text(translations, 'insights.next_steps')}")
        
        st.markdown("""
        To expand this analysis:
        - Run the main script for **advanced NLP features**
        - Explore **price vs quality correlations**
        - Generate **word cloud visualizations**
        - Apply **machine learning models** for prediction
        """)

    # Informações do projeto
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
        <h4>{get_text(translations, "footer.title")}</h4>
        <p>{get_text(translations, "footer.developers")}</p>
        <p><em>{get_text(translations, "footer.course")}</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
