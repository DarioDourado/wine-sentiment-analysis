"""
Chart generation for wine sentiment analysis
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

# Configure matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate all charts for wine sentiment analysis"""
    
    def __init__(self, settings):
        self.settings = settings
        self.output_dir = settings.images_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(settings.chart_style)
        sns.set_palette("husl")
        
    def generate_all_charts(self, df: pd.DataFrame) -> List[str]:
        """Generate all visualization charts"""
        logger.info("🎨 Generating all visualization charts...")
        
        generated_files = []
        
        try:
            # 1. Polarity comparison
            file1 = self.create_polarity_comparison(df)
            if file1: generated_files.append(file1)
            
            # 2. Sentiment distribution
            file2 = self.create_sentiment_distribution(df)
            if file2: generated_files.append(file2)
            
            # 3. Wine terms impact
            file3 = self.create_wine_terms_impact(df)
            if file3: generated_files.append(file3)
            
            # 4. Top wine terms
            file4 = self.create_top_wine_terms(df)
            if file4: generated_files.append(file4)
            
            # 5. Rating correlation
            file5 = self.create_rating_correlation(df)
            if file5: generated_files.append(file5)
            
            # 6. Varietal analysis
            file6 = self.create_varietal_analysis(df)
            if file6: generated_files.append(file6)
            
            # 7. Country analysis
            file7 = self.create_country_analysis(df)
            if file7: generated_files.append(file7)
            
            # 8. Complete dashboard
            file8 = self.create_dashboard_summary(df)
            if file8: generated_files.append(file8)
            
            # 9. Wine type analysis
            file9 = self.create_wine_type_analysis(df)
            if file9: generated_files.append(file9)
            
            # 10. Word cloud analysis
            file10 = self.create_wordcloud_analysis(df)
            if file10: generated_files.append(file10)
            
            # 11. Advanced NLP analysis
            file11 = self.create_advanced_nlp_analysis(df)
            if file11: generated_files.append(file11)
            
            logger.info(f"✅ Generated {len(generated_files)} charts successfully")
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ Error generating charts: {e}")
            return generated_files
    
    def create_polarity_comparison(self, df: pd.DataFrame) -> Optional[str]:
        """Create polarity comparison chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original polarity
            sns.histplot(df['original_polarity'], kde=True, alpha=0.7, 
                        color='#3498db', ax=ax1, bins=30)
            ax1.axvline(df['original_polarity'].mean(), color='red', 
                       linestyle='--', alpha=0.8, linewidth=2)
            ax1.set_title('TextBlob Original\nPolarity Distribution', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Polarity')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Enhanced polarity
            sns.histplot(df['enhanced_polarity'], kde=True, alpha=0.7, 
                        color='#e74c3c', ax=ax2, bins=30)
            ax2.axvline(df['enhanced_polarity'].mean(), color='red', 
                       linestyle='--', alpha=0.8, linewidth=2)
            ax2.set_title('Enhanced Analysis\nPolarity Distribution', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Polarity')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('🍷 Wine Sentiment Analysis Comparison', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = self.output_dir / '01_polarity_comparison.png'
            plt.savefig(filename, dpi=self.settings.chart_dpi, bbox_inches='tight')
            plt.show()
            
            logger.info("✅ Chart 1/11: Polarity comparison saved")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Error creating polarity comparison: {e}")
            return None
    
    def create_sentiment_distribution(self, df: pd.DataFrame) -> Optional[str]:
        """Create sentiment distribution chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original sentiment categories
            original_categories = df['original_polarity'].apply(
                lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
            )
            original_counts = original_categories.value_counts()
            colors_orig = ['#2ecc71' if cat == 'Positive' else '#e74c3c' if cat == 'Negative' else '#95a5a6' 
                          for cat in original_counts.index]
            
            ax1.pie(original_counts.values, labels=original_counts.index, 
                   autopct='%1.1f%%', colors=colors_orig, startangle=90)
            ax1.set_title('TextBlob Original\nSentiment Distribution', fontweight='bold')
            
            # Enhanced sentiment categories
            enhanced_counts = df['sentiment_category'].value_counts()
            colors_enh = ['#2ecc71' if cat == 'Positive' else '#e74c3c' if cat == 'Negative' else '#95a5a6' 
                         for cat in enhanced_counts.index]
            
            ax2.pie(enhanced_counts.values, labels=enhanced_counts.index, 
                   autopct='%1.1f%%', colors=colors_enh, startangle=90)
            ax2.set_title('Enhanced Analysis\nSentiment Distribution', fontweight='bold')
            
            plt.suptitle('📊 Sentiment Category Distribution Comparison', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = self.output_dir / '02_sentiment_distribution.png'
            plt.savefig(filename, dpi=self.settings.chart_dpi, bbox_inches='tight')
            plt.show()
            
            logger.info("✅ Chart 2/11: Sentiment distribution saved")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment distribution: {e}")
            return None
    
    # Continue with other chart methods...
    def create_wine_terms_impact(self, df: pd.DataFrame) -> Optional[str]:
        """Create wine terms impact analysis"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Wine terms frequency
            terms_impact = df['wine_terms_found'].value_counts().sort_index()
            ax1.bar(terms_impact.index, terms_impact.values, color='#8e44ad', alpha=0.7)
            ax1.set_title('Wine Terms Distribution\nFound per Review', fontweight='bold')
            ax1.set_xlabel('Number of Wine Terms Found')
            ax1.set_ylabel('Number of Reviews')
            ax1.grid(True, alpha=0.3)
            
            # Wine boost distribution
            wine_boost_data = df[df['wine_boost'] != 0]['wine_boost']
            if len(wine_boost_data) > 0:
                sns.histplot(wine_boost_data, kde=True, alpha=0.7, color='#f39c12', ax=ax2, bins=20)
                ax2.axvline(wine_boost_data.mean(), color='red', linestyle='--', alpha=0.8, linewidth=2)
                ax2.set_title('Wine Terms Boost\nDistribution', fontweight='bold')
                ax2.set_xlabel('Wine Terms Boost')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            
            plt.suptitle('🍇 Wine Terms Impact Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = self.output_dir / '03_wine_terms_impact.png'
            plt.savefig(filename, dpi=self.settings.chart_dpi, bbox_inches='tight')
            plt.show()
            
            logger.info("✅ Chart 3/11: Wine terms impact saved")
            return str(filename)
            
        except Exception as e:
            logger.error(f"❌ Error creating wine terms impact: {e}")
            return None
    
    def create_wordcloud_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create word cloud analysis"""
        try:
            from wordcloud import WordCloud
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Positive sentiment wordcloud
            positive_text = ' '.join(df[df['sentiment_category'] == 'Positive']['review'].dropna().astype(str))
            if len(positive_text) > 100:
                wc_pos = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='Greens', max_words=100).generate(positive_text)
                axes[0,0].imshow(wc_pos, interpolation='bilinear')
                axes[0,0].set_title('😊 Positive Sentiment', fontweight='bold')
                axes[0,0].axis('off')
            
            # Negative sentiment wordcloud
            negative_text = ' '.join(df[df['sentiment_category'] == 'Negative']['review'].dropna().astype(str))
            if len(negative_text) > 100:
                wc_neg = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='Reds', max_words=100).generate(negative_text)
                axes[0,1].imshow(wc_neg, interpolation='bilinear')
                axes[0,1].set_title('😞 Negative Sentiment', fontweight='bold')
                axes[0,1].axis('off')
            
            # High-rated wines wordcloud
            if 'rating' in df.columns:
                high_rated_text = ' '.join(df[df['rating'] >= 90]['review'].dropna().astype(str))
                if len(high_rated_text) > 100:
                    wc_high = WordCloud(width=800, height=400, background_color='white', 
                                       colormap='Blues', max_words=100).generate(high_rated_text)
                    axes[1,0].imshow(wc_high, interpolation='bilinear')
                    axes[1,0].set_title('⭐ High-Rated Wines (≥90)', fontweight='bold')
                    axes[1,0].axis('off')
            
            # All reviews wordcloud
            all_text = ' '.join(df['review'].dropna().astype(str))
            if len(all_text) > 100:
                wc_all = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='viridis', max_words=100).generate(all_text)
                axes[1,1].imshow(wc_all, interpolation='bilinear')
                axes[1,1].set_title('🍷 All Reviews', fontweight='bold')
                axes[1,1].axis('off')
            
            plt.suptitle('☁️ Word Cloud Analysis by Category', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = self.output_dir / '10_wordcloud_analysis.png'
            plt.savefig(filename, dpi=self.settings.chart_dpi, bbox_inches='tight')
            plt.show()
            
            logger.info("✅ Chart 10/11: Word cloud analysis saved")
            return str(filename)
            
        except ImportError:
            logger.warning("⚠️ WordCloud not installed. Skipping word cloud generation.")
            return None
        except Exception as e:
            logger.error(f"❌ Error creating word cloud analysis: {e}")
            return None
    
    # Add placeholder methods for other charts
    def create_top_wine_terms(self, df: pd.DataFrame) -> Optional[str]:
        """Create top wine terms chart"""
        # Implementation here...
        return None
    
    def create_rating_correlation(self, df: pd.DataFrame) -> Optional[str]:
        """Create rating correlation chart"""
        # Implementation here...
        return None
    
    def create_varietal_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create varietal analysis chart"""
        # Implementation here...
        return None
    
    def create_country_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create country analysis chart"""
        # Implementation here...
        return None
    
    def create_dashboard_summary(self, df: pd.DataFrame) -> Optional[str]:
        """Create dashboard summary chart"""
        # Implementation here...
        return None
    
    def create_wine_type_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create wine type analysis chart"""
        # Implementation here...
        return None
    
    def create_advanced_nlp_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Create advanced NLP analysis chart"""
        # Implementation here...
        return None