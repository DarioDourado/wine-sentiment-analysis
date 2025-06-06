"""
Data cleaning utilities for Wine Sentiment Analysis
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class WineDataCleaner:
    """Wine data cleaning and preprocessing"""
    
    def __init__(self, settings):
        self.settings = settings
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess wine data"""
        logger.info(f"🧹 Starting data cleaning for {len(df)} records...")
        
        df_clean = df.copy()
        
        # Clean text columns
        df_clean = self._clean_text_columns(df_clean)
        
        # Clean numeric columns
        df_clean = self._clean_numeric_columns(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        logger.info(f"✅ Data cleaning completed. {len(df_clean)} records remaining")
        return df_clean
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        text_columns = ['review', 'wine', 'winery', 'varietal', 'country', 'designation']
        
        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Remove special characters for review column
                if col == 'review':
                    df[col] = df[col].str.replace(r'[^\w\s\.,!?;:()\-\'"]', ' ', regex=True)
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                
                # Replace 'nan' strings with NaN
                df[col] = df[col].replace(['nan', 'NaN', 'null', 'NULL', ''], np.nan)
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns"""
        # Clean price column
        if 'price' in df.columns:
            df['price'] = self._clean_price_column(df['price'])
        
        # Clean rating column
        if 'rating' in df.columns:
            df['rating'] = self._clean_rating_column(df['rating'])
        
        # Clean alcohol column
        if 'alcohol' in df.columns:
            df['alcohol'] = self._clean_alcohol_column(df['alcohol'])
        
        return df
    
    def _clean_price_column(self, price_series: pd.Series) -> pd.Series:
        """Clean price column"""
        # Convert to string first
        price_clean = price_series.astype(str)
        
        # Remove currency symbols and extra characters
        price_clean = price_clean.str.replace(r'[$€£¥,]', '', regex=True)
        price_clean = price_clean.str.replace(r'[^\d.]', '', regex=True)
        
        # Convert to numeric
        price_clean = pd.to_numeric(price_clean, errors='coerce')
        
        # Remove unrealistic values
        price_clean = price_clean.where(
            (price_clean >= 1) & (price_clean <= 10000), np.nan
        )
        
        return price_clean
    
    def _clean_rating_column(self, rating_series: pd.Series) -> pd.Series:
        """Clean rating column"""
        rating_clean = pd.to_numeric(rating_series, errors='coerce')
        
        # Assume ratings are on 100-point scale, but accept other scales
        rating_clean = rating_clean.where(
            (rating_clean >= 50) & (rating_clean <= 100), np.nan
        )
        
        return rating_clean
    
    def _clean_alcohol_column(self, alcohol_series: pd.Series) -> pd.Series:
        """Clean alcohol column"""
        alcohol_clean = alcohol_series.astype(str)
        
        # Remove % symbol
        alcohol_clean = alcohol_clean.str.replace('%', '')
        alcohol_clean = pd.to_numeric(alcohol_clean, errors='coerce')
        
        # Remove unrealistic values
        alcohol_clean = alcohol_clean.where(
            (alcohol_clean >= 5) & (alcohol_clean <= 20), np.nan
        )
        
        return alcohol_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        
        # Remove exact duplicates
        df_no_dups = df.drop_duplicates()
        
        # Remove duplicates based on review text if available
        if 'review' in df.columns:
            df_no_dups = df_no_dups.drop_duplicates(subset=['review'], keep='first')
        
        removed_count = initial_count - len(df_no_dups)
        if removed_count > 0:
            logger.info(f"🗑️ Removed {removed_count} duplicate records")
        
        return df_no_dups
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Drop rows with missing reviews (critical column)
        if 'review' in df.columns:
            initial_count = len(df)
            df = df.dropna(subset=['review'])
            dropped_count = initial_count - len(df)
            if dropped_count > 0:
                logger.info(f"🗑️ Dropped {dropped_count} rows with missing reviews")
        
        return df