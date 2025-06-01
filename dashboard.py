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
            # Fallback para inglês se o ficheiro não existir
            st.warning(f"Translation file {translation_file} not found. Using English as fallback.")
            with open("translation/en.json", 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading translations: {e}")
        # Traduções básicas de emergência
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
        
        # Formatar com parâmetros se fornecidos
        if kwargs:
            return value.format(**kwargs)
        return value
    except (KeyError, TypeError):
        return key_path  # Retorna a chave se não encontrar tradução

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

# ====================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ====================================================================

def create_modern_donut_chart(df, column, title, colors=None):
    """Criar gráfico donut moderno"""
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
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        height=400,
        margin=dict(t=60, b=20, l=20, r=120)
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
        grouped = df.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
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
    
    # Adicionar linha de tendência
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
    # MÉTRICAS PRINCIPAIS
    # ================================================================
    
    st.markdown(f'<h2 class="section-header">{get_text(translations, "metrics.title")}</h2>', unsafe_allow_html=True)
    
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
            st.metric(
                get_text(translations, 'metrics.wine_terms_metric'),
                f"{total_terms:,}"
            )
    
    with col5:
        if 'wine_type' in df.columns:
            unique_types = df['wine_type'].nunique()
            st.metric(
                get_text(translations, 'metrics.wine_types_metric'),
                f"{unique_types}"
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
    
    # Row 5: Rating vs Sentiment Correlation
    if 'rating' in df.columns and 'enhanced_polarity' in df.columns:
        st.markdown(f'<h3 class="section-header">{get_text(translations, "visualizations.rating_sentiment_correlation")}</h3>', unsafe_allow_html=True)
        
        fig_scatter = create_scatter_plot(
            df,
            'rating',
            'enhanced_polarity',
            'wine_terms_found',
            get_text(translations, 'visualizations.rating_vs_sentiment')
        )
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ================================================================
    # DATA EXPLORER
    # ================================================================
    
    st.markdown(f'<h2 class="section-header">{get_text(translations, "data_explorer.title")}</h2>', unsafe_allow_html=True)
    
    # Column selector
    available_cols = [col for col in df.columns if col not in ['wine_terms_list']]
    default_cols = []
    
    # Priority columns
    priority_cols = ['wine', 'winery', 'varietal', 'wine_type', 'country', 'enhanced_polarity', 'sentiment_category', 'rating']
    for col in priority_cols:
        if col in available_cols and len(default_cols) < 8:
            default_cols.append(col)
    
    selected_cols = st.multiselect(
        get_text(translations, 'data_explorer.select_columns'),
        available_cols,
        default=default_cols,
        help=get_text(translations, 'data_explorer.select_columns_help')
    )
    
    if selected_cols:
        # Sorting options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            sort_col = st.selectbox(
                get_text(translations, 'data_explorer.sort_by'),
                selected_cols,
                index=0 if 'enhanced_polarity' not in selected_cols else selected_cols.index('enhanced_polarity')
            )
        
        with col2:
            sort_order = st.radio(
                get_text(translations, 'data_explorer.order'), 
                [get_text(translations, 'data_explorer.descending'), get_text(translations, 'data_explorer.ascending')], 
                horizontal=True
            )
        
        with col3:
            sample_size = st.selectbox(
                get_text(translations, 'data_explorer.show_rows'),
                [50, 100, 250, 500, 1000],
                index=1
            )
        
        # Display filtered and sorted data
        display_df = df[selected_cols].copy()
        
        if sort_order == get_text(translations, 'data_explorer.descending'):
            display_df = display_df.sort_values(sort_col, ascending=False)
        else:
            display_df = display_df.sort_values(sort_col, ascending=True)
        
        # Format numeric columns
        for col in display_df.columns:
            if col == 'enhanced_polarity' and col in display_df.columns:
                display_df[col] = display_df[col].round(3)
            elif col == 'rating' and col in display_df.columns:
                display_df[col] = display_df[col].round(1)
        
        st.dataframe(
            display_df.head(sample_size),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.head(sample_size).to_csv(index=False)
        st.download_button(
            label=get_text(translations, 'data_explorer.download_data'),
            data=csv,
            file_name=f'wine_analysis_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    # ================================================================
    # INSIGHTS E ESTATÍSTICAS
    # ================================================================
    
    st.markdown(f'<h2 class="section-header">{get_text(translations, "insights.title")}</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>{get_text(translations, 'insights.dataset_overview')}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        insights_data = {
            get_text(translations, 'insights.total_records'): f"{len(df):,}",
            get_text(translations, 'insights.memory_usage'): f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
            get_text(translations, 'insights.columns'): f"{len(df.columns)}",
            get_text(translations, 'insights.data_types'): len(df.dtypes.unique())
        }
        
        for key, value in insights_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>{get_text(translations, 'insights.analysis_summary')}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if 'enhanced_polarity' in df.columns:
            analysis_data = {
                get_text(translations, 'insights.avg_polarity'): f"{df['enhanced_polarity'].mean():.3f}",
                get_text(translations, 'insights.polarity_std'): f"{df['enhanced_polarity'].std():.3f}",
                get_text(translations, 'insights.most_positive'): f"{df['enhanced_polarity'].max():.3f}",
                get_text(translations, 'insights.most_negative'): f"{df['enhanced_polarity'].min():.3f}"
            }
            
            for key, value in analysis_data.items():
                st.write(f"**{key}:** {value}")
    
    # ================================================================
    # FOOTER
    # ================================================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
        <p>{get_text(translations, 'footer.text')}</p>
        <p><em>{get_text(translations, 'footer.data_processed')} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    </div>
    """, unsafe_allow_html=True)

# ====================================================================
# EXECUTAR APLICAÇÃO
# ====================================================================

if __name__ == "__main__":
    main()