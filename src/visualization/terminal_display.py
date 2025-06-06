"""
Terminal display utilities for Wine Sentiment Analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class TerminalDisplay:
    """Terminal-based display and analysis"""
    
    def __init__(self, settings):
        self.settings = settings
    
    def show_complete_analysis(self, df: pd.DataFrame):
        """Show complete analysis in terminal"""
        logger.info("📋 Displaying complete terminal analysis...")
        
        self._show_header()
        self._show_dataset_overview(df)
        self._show_sentiment_analysis(df)
        self._show_wine_terms_analysis(df)
        self._show_correlations(df)
        self._show_insights_and_recommendations(df)
    
    def _show_header(self):
        """Show analysis header"""
        print("\n" + "🍷" * 35)
        print("🍷 WINE SENTIMENT ANALYSIS - COMPLETE REPORT")
        print("🍷" * 70)
        print("📊 Advanced sentiment analysis using TextBlob + Wine Lexicon")
        print("🔬 Enhanced with 400+ specialized wine terms")
        print("="*70)
    
    def _show_dataset_overview(self, df: pd.DataFrame):
        """Show dataset overview"""
        print(f"\n📋 DATASET OVERVIEW:")
        print("="*50)
        print(f"   • Total reviews: {len(df):,}")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Missing values: {df.isnull().sum().sum():,}")
        
        if 'country' in df.columns:
            unique_countries = df['country'].nunique()
            print(f"   • Unique countries: {unique_countries}")
        
        if 'wine_type' in df.columns:
            wine_types = df['wine_type'].value_counts()
            print(f"   • Wine types: {dict(wine_types)}")
    
    def _show_sentiment_analysis(self, df: pd.DataFrame):
        """Show sentiment analysis results"""
        print(f"\n🔬 SENTIMENT ANALYSIS RESULTS:")
        print("="*50)
        
        if 'sentiment_category' in df.columns:
            sentiment_dist = df['sentiment_category'].value_counts()
            total = len(df)
            
            for category, count in sentiment_dist.items():
                percentage = (count / total) * 100
                emoji = "😊" if category == "Positive" else "😞" if category == "Negative" else "😐"
                print(f"   {emoji} {category}: {count:,} ({percentage:.1f}%)")
        
        if 'enhanced_polarity' in df.columns:
            avg_polarity = df['enhanced_polarity'].mean()
            std_polarity = df['enhanced_polarity'].std()
            print(f"\n   📊 Average polarity: {avg_polarity:+.3f}")
            print(f"   📊 Polarity std dev: {std_polarity:.3f}")
    
    def _show_wine_terms_analysis(self, df: pd.DataFrame):
        """Show wine terms analysis"""
        if 'wine_terms_found' not in df.columns:
            return
            
        print(f"\n🍇 WINE TERMS ANALYSIS:")
        print("="*50)
        
        avg_terms = df['wine_terms_found'].mean()
        max_terms = df['wine_terms_found'].max()
        reviews_with_terms = (df['wine_terms_found'] > 0).sum()
        
        print(f"   • Average wine terms per review: {avg_terms:.1f}")
        print(f"   • Maximum wine terms in single review: {max_terms}")
        print(f"   • Reviews with wine terms: {reviews_with_terms:,} ({reviews_with_terms/len(df)*100:.1f}%)")
        
        # Show most common wine terms if available
        if 'wine_terms_list' in df.columns:
            all_terms = []
            for terms_list in df['wine_terms_list']:
                if isinstance(terms_list, list):
                    all_terms.extend(terms_list)
            
            if all_terms:
                from collections import Counter
                top_terms = Counter(all_terms).most_common(10)
                print(f"\n   🏆 TOP 10 WINE TERMS:")
                for i, (term, count) in enumerate(top_terms, 1):
                    print(f"      {i:2d}. {term}: {count} times")
    
    def _show_correlations(self, df: pd.DataFrame):
        """Show correlation analysis"""
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print("="*50)
        
        # Rating vs sentiment correlation
        if 'rating' in df.columns and 'enhanced_polarity' in df.columns:
            correlation = df['rating'].corr(df['enhanced_polarity'])
            print(f"   📊 Rating vs Sentiment: {correlation:.3f}")
            
            if correlation > 0.3:
                print("      🟢 Strong positive correlation")
            elif correlation > 0.1:
                print("      🟡 Moderate positive correlation")
            else:
                print("      🔴 Weak correlation")
        
        # Price vs sentiment correlation
        if 'price' in df.columns and 'enhanced_polarity' in df.columns:
            price_corr = df['price'].corr(df['enhanced_polarity'])
            print(f"   💰 Price vs Sentiment: {price_corr:.3f}")
        
        # Wine terms vs sentiment correlation
        if 'wine_terms_found' in df.columns and 'enhanced_polarity' in df.columns:
            terms_corr = df['wine_terms_found'].corr(df['enhanced_polarity'])
            print(f"   🍇 Wine Terms vs Sentiment: {terms_corr:.3f}")
    
    def _show_insights_and_recommendations(self, df: pd.DataFrame):
        """Show insights and recommendations"""
        print(f"\n💡 INSIGHTS & RECOMMENDATIONS:")
        print("="*50)
        
        # Sentiment insights
        if 'sentiment_category' in df.columns:
            positive_pct = (df['sentiment_category'] == 'Positive').mean() * 100
            
            if positive_pct > 60:
                print("🟢 Sentiment predominantly positive - good overall wine quality")
            elif positive_pct > 40:
                print("🟡 Balanced sentiment - varied quality range")
            else:
                print("🔴 Sentiment predominantly neutral/negative - consider quality curation")
        
        # Wine terms insights
        if 'wine_terms_found' in df.columns:
            avg_terms = df['wine_terms_found'].mean()
            
            if avg_terms > 3:
                print("🟢 Reviews rich in technical vocabulary")
            elif avg_terms > 1.5:
                print("🟡 Moderate technical vocabulary")
            else:
                print("🔴 Limited technical vocabulary - simpler reviews")
        
        # Best performing categories
        if 'wine_type' in df.columns and 'enhanced_polarity' in df.columns:
            wine_type_sentiment = df.groupby('wine_type')['enhanced_polarity'].mean().sort_values(ascending=False)
            if not wine_type_sentiment.empty:
                best_type = wine_type_sentiment.index[0]
                print(f"🏆 Best performing wine type: {best_type}")
        
        print(f"\n✅ Analysis completed successfully!")
        print("="*70)