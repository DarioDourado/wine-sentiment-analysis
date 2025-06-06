"""
Data loading utilities for Wine Sentiment Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class WineDataLoader:
    """Wine data loader with validation and error handling"""
    
    def __init__(self, settings):
        self.settings = settings
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with multiple encoding attempts"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return pd.DataFrame()
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"✅ Data loaded successfully with {encoding} encoding")
                logger.info(f"📊 Dataset shape: {df.shape}")
                logger.info(f"📋 Columns: {list(df.columns)}")
                return df
                
            except UnicodeDecodeError:
                logger.warning(f"⚠️ Failed to load with {encoding} encoding")
                continue
            except Exception as e:
                logger.error(f"❌ Error loading file: {e}")
                return pd.DataFrame()
        
        logger.error("❌ Failed to load file with any encoding")
        return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data and return validation report"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("Dataset is empty")
            return validation
        
        # Check for required columns
        required_columns = ['review']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for recommended columns
        recommended_columns = ['rating', 'price', 'wine', 'country', 'varietal']
        missing_recommended = [col for col in recommended_columns if col not in df.columns]
        
        if missing_recommended:
            validation['warnings'].append(f"Missing recommended columns: {missing_recommended}")
        
        # Calculate basic stats
        validation['stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return validation