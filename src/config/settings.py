"""
Configuration settings for Wine Sentiment Analysis
"""

import os
from pathlib import Path
import json
from typing import Dict, Any

class Settings:
    """Configuration settings class"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.assets_dir = self.base_dir / "assets"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.assets_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Data paths
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.exports_dir = self.data_dir / "exports"
        
        # Asset paths
        self.images_dir = self.assets_dir / "images"
        self.translations_dir = self.assets_dir / "translations"
        
        # Analysis settings
        self.sentiment_threshold_positive = 0.1
        self.sentiment_threshold_negative = -0.1
        self.min_wine_terms_for_analysis = 1
        self.max_charts_to_generate = 11
        
        # Visualization settings
        self.chart_dpi = 300
        self.chart_style = 'seaborn-v0_8'
        self.default_figsize = (12, 8)
        
        # Dashboard settings
        self.dashboard_port = 8501
        self.dashboard_host = 'localhost'
        
        # Language settings
        self.default_language = 'en'
        self.supported_languages = ['en', 'pt']
        
        # Create subdirectories
        for subdir in [self.raw_data_dir, self.processed_data_dir, self.exports_dir, 
                      self.images_dir, self.translations_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
    
    def get_translation_file(self, language: str) -> Path:
        """Get translation file path for given language"""
        return self.translations_dir / f"{language}.json"
    
    def load_translations(self, language: str = None) -> Dict[str, Any]:
        """Load translations for given language"""
        if language is None:
            language = self.default_language
            
        translation_file = self.get_translation_file(language)
        
        if translation_file.exists():
            with open(translation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return default English translations
            return self._get_default_translations()
    
    def _get_default_translations(self) -> Dict[str, Any]:
        """Get default English translations"""
        return {
            "main_title": "🍷 Wine Sentiment Analysis",
            "main_subtitle": "Advanced sentiment analysis for wine reviews",
            "filters": {
                "title": "Advanced Filters",
                "reset": "Reset Filters",
                "wine_type": "Wine Type",
                "sentiment_category": "Sentiment",
                "country": "Country",
                "polarity_range": "Polarity Range",
                "rating_range": "Rating Range"
            },
            "metrics": {
                "title": "Key Metrics",
                "total_reviews": "Total Reviews",
                "avg_sentiment": "Avg Sentiment",
                "positive_pct": "Positive %",
                "wine_terms": "Wine Terms"
            },
            "charts": {
                "sentiment_distribution": "Sentiment Distribution",
                "wine_types": "Wine Types Analysis",
                "top_countries": "Top Countries",
                "rating_correlation": "Rating vs Sentiment"
            }
        }
    
    def get_wine_lexicon_path(self) -> Path:
        """Get wine lexicon file path"""
        return self.base_dir / "src" / "analysis" / "wine_lexicon.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'base_dir': str(self.base_dir),
            'data_dir': str(self.data_dir),
            'sentiment_threshold_positive': self.sentiment_threshold_positive,
            'sentiment_threshold_negative': self.sentiment_threshold_negative,
            'chart_dpi': self.chart_dpi,
            'default_language': self.default_language
        }