"""
Wine sentiment analysis with specialized lexicon
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import Dict, List, Tuple, Any
import json
import logging
from pathlib import Path

from .wine_lexicon import WineLexicon

logger = logging.getLogger(__name__)

class WineSentimentAnalyzer:
    """Advanced sentiment analyzer for wine reviews"""
    
    def __init__(self, settings):
        self.settings = settings
        self.wine_lexicon = WineLexicon()
        self.lexicon_dict = self.wine_lexicon.get_lexicon()
        
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sentiment analysis to wine reviews"""
        logger.info(f"🔬 Analyzing sentiment for {len(df)} reviews...")
        
        if 'review' not in df.columns:
            raise ValueError("Column 'review' not found in dataframe")
        
        # Apply enhanced sentiment analysis
        sentiment_results = df['review'].apply(self._enhanced_sentiment_analysis)
        
        # Extract results into separate columns
        df['original_polarity'] = [r['polarity'] for r in sentiment_results]
        df['subjectivity'] = [r['subjectivity'] for r in sentiment_results]
        df['wine_boost'] = [r['wine_boost'] for r in sentiment_results]
        df['wine_terms_found'] = [r['wine_terms_found'] for r in sentiment_results]
        df['enhanced_polarity'] = [r['enhanced_polarity'] for r in sentiment_results]
        df['wine_terms_list'] = [r['wine_terms_list'] for r in sentiment_results]
        
        # Categorize sentiment
        df['sentiment_category'] = df['enhanced_polarity'].apply(self._categorize_sentiment)
        
        logger.info("✅ Sentiment analysis completed")
        return df
    
    def _enhanced_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with wine-specific terms"""
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
        
        # Standard TextBlob analysis
        blob = TextBlob(text_str)
        original_polarity = blob.sentiment.polarity
        original_subjectivity = blob.sentiment.subjectivity
        
        # Wine-specific analysis
        wine_boost = 0.0
        wine_terms_found = 0
        wine_terms_list = []
        
        for term, score in self.lexicon_dict.items():
            if term in text_str:
                wine_boost += score
                wine_terms_found += 1
                wine_terms_list.append(term)
        
        # Calculate average boost
        if wine_terms_found > 0:
            wine_boost = wine_boost / wine_terms_found
        
        # Combine analyses
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
    
    def _categorize_sentiment(self, polarity: float) -> str:
        """Categorize sentiment based on polarity"""
        if polarity > self.settings.sentiment_threshold_positive:
            return 'Positive'
        elif polarity < self.settings.sentiment_threshold_negative:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sentiment statistics"""
        if 'sentiment_category' not in df.columns:
            return {}
        
        total = len(df)
        sentiment_counts = df['sentiment_category'].value_counts()
        
        stats = {
            'total_reviews': total,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': (sentiment_counts / total * 100).to_dict(),
            'average_polarity': df['enhanced_polarity'].mean(),
            'polarity_std': df['enhanced_polarity'].std(),
            'reviews_with_wine_terms': (df['wine_terms_found'] > 0).sum(),
            'average_wine_terms': df['wine_terms_found'].mean(),
            'most_common_wine_terms': self._get_most_common_terms(df)
        }
        
        return stats
    
    def _get_most_common_terms(self, df: pd.DataFrame, top_n: int = 20) -> List[Tuple[str, int]]:
        """Get most common wine terms found in reviews"""
        all_terms = []
        for terms_list in df['wine_terms_list']:
            all_terms.extend(terms_list)
        
        from collections import Counter
        term_counts = Counter(all_terms)
        return term_counts.most_common(top_n)