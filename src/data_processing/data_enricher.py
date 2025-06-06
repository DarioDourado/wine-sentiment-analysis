"""
Data enrichment utilities for Wine Sentiment Analysis
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class WineDataEnricher:
    """Wine data enrichment with additional features"""
    
    def __init__(self, settings):
        self.settings = settings
        
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with additional features"""
        logger.info(f"✨ Starting data enrichment for {len(df)} records...")
        
        df_enriched = df.copy()
        
        # Add wine type categorization
        df_enriched = self._categorize_wine_type(df_enriched)
        
        # Extract vintage year
        df_enriched = self._extract_vintage_year(df_enriched)
        
        # Categorize price ranges
        df_enriched = self._categorize_price_ranges(df_enriched)
        
        # Add text statistics
        df_enriched = self._add_text_statistics(df_enriched)
        
        # Normalize country names
        df_enriched = self._normalize_country_names(df_enriched)
        
        logger.info("✅ Data enrichment completed")
        return df_enriched
    
    def _categorize_wine_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize wine types based on wine name or varietal"""
        def determine_wine_type(row):
            wine_text = ''
            
            # Combine relevant text columns
            for col in ['wine', 'varietal', 'designation']:
                if col in df.columns and pd.notna(row.get(col)):
                    wine_text += str(row[col]).lower() + ' '
            
            # Red wine indicators
            red_indicators = [
                'cabernet', 'merlot', 'pinot noir', 'syrah', 'shiraz', 'malbec',
                'sangiovese', 'tempranillo', 'grenache', 'zinfandel', 'petite sirah',
                'mourvedre', 'carignan', 'barbera', 'nebbiolo', 'chianti', 'bordeaux',
                'red wine', 'rouge', 'tinto', 'rosso'
            ]
            
            # White wine indicators
            white_indicators = [
                'chardonnay', 'sauvignon blanc', 'pinot grigio', 'pinot gris',
                'riesling', 'gewurztraminer', 'chenin blanc', 'viognier',
                'albarino', 'verdejo', 'muscadet', 'chablis', 'sancerre',
                'white wine', 'blanc', 'blanco', 'bianco'
            ]
            
            # Sparkling wine indicators
            sparkling_indicators = [
                'champagne', 'prosecco', 'cava', 'sparkling', 'brut', 'sec',
                'crémant', 'franciacorta', 'spumante', 'sekt'
            ]
            
            # Rosé wine indicators
            rose_indicators = [
                'rosé', 'rose', 'rosado', 'rosato', 'pink', 'blush'
            ]
            
            # Dessert wine indicators
            dessert_indicators = [
                'port', 'sherry', 'sauternes', 'ice wine', 'dessert',
                'late harvest', 'tokaji', 'madeira', 'moscato'
            ]
            
            # Check categories in order of specificity
            if any(indicator in wine_text for indicator in sparkling_indicators):
                return 'Sparkling'
            elif any(indicator in wine_text for indicator in dessert_indicators):
                return 'Dessert'
            elif any(indicator in wine_text for indicator in rose_indicators):
                return 'Rosé'
            elif any(indicator in wine_text for indicator in red_indicators):
                return 'Red'
            elif any(indicator in wine_text for indicator in white_indicators):
                return 'White'
            else:
                return 'Other'
        
        df['wine_type'] = df.apply(determine_wine_type, axis=1)
        
        type_counts = df['wine_type'].value_counts()
        logger.info(f"🍷 Wine type distribution: {dict(type_counts)}")
        
        return df
    
    def _extract_vintage_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract vintage year from wine names"""
        def extract_year(wine_name):
            if pd.isna(wine_name):
                return np.nan
            
            # Look for 4-digit years between 1800 and current year
            import datetime
            current_year = datetime.datetime.now().year
            
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', str(wine_name))
            
            if years:
                year = int(years[0])
                if 1800 <= year <= current_year:
                    return year
            
            return np.nan
        
        if 'wine' in df.columns:
            df['vintage_year'] = df['wine'].apply(extract_year)
            
            # Calculate wine age
            current_year = pd.Timestamp.now().year
            df['wine_age'] = current_year - df['vintage_year']
            
            vintage_count = df['vintage_year'].notna().sum()
            logger.info(f"📅 Extracted vintage year for {vintage_count} wines")
        
        return df
    
    def _categorize_price_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize wines into price ranges"""
        if 'price' in df.columns:
            # Calculate price percentiles
            price_data = df['price'].dropna()
            if len(price_data) > 0:
                price_percentiles = price_data.quantile([0.25, 0.5, 0.75])
                
                def categorize_price(price):
                    if pd.isna(price):
                        return 'Unknown'
                    elif price <= price_percentiles[0.25]:
                        return 'Budget'
                    elif price <= price_percentiles[0.5]:
                        return 'Mid-range'
                    elif price <= price_percentiles[0.75]:
                        return 'Premium'
                    else:
                        return 'Luxury'
                
                df['price_category'] = df['price'].apply(categorize_price)
                
                price_cat_counts = df['price_category'].value_counts()
                logger.info(f"💰 Price category distribution: {dict(price_cat_counts)}")
        
        return df
    
    def _add_text_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-based statistics"""
        if 'review' in df.columns:
            # Review length statistics
            df['review_length'] = df['review'].str.len()
            df['review_word_count'] = df['review'].str.split().str.len()
            
            # Average word length
            df['avg_word_length'] = df['review'].apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) else 0
            )
            
            logger.info(f"📝 Added text statistics (avg length: {df['review_length'].mean():.0f} chars)")
        
        return df
    
    def _normalize_country_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize country names"""
        if 'country' in df.columns:
            # Country name mappings
            country_mapping = {
                'US': 'United States',
                'USA': 'United States',
                'UK': 'United Kingdom',
                'Deutschland': 'Germany',
                'España': 'Spain',
                'Italia': 'Italy',
                'França': 'France'
            }
            
            # Apply mapping
            df['country'] = df['country'].replace(country_mapping)
            
            # Clean country names
            df['country'] = df['country'].str.title().str.strip()
            
            country_counts = df['country'].value_counts().head(10)
            logger.info(f"🌍 Top countries: {dict(country_counts)}")
        
        return df