"""
Wine-specific lexicon for sentiment analysis enhancement
"""

import json
from pathlib import Path
from typing import Dict, List

class WineLexicon:
    """Wine terminology lexicon with sentiment scores"""
    
    def __init__(self):
        self.lexicon = self._create_wine_lexicon()
    
    def get_lexicon(self) -> Dict[str, float]:
        """Get the complete wine lexicon dictionary"""
        return self.lexicon
    
    def _create_wine_lexicon(self) -> Dict[str, float]:
        """Create comprehensive wine lexicon with sentiment scores"""
        
        lexicon = {
            # Extremely positive terms (0.8 - 1.0)
            'exceptional': 0.9,
            'outstanding': 0.9,
            'magnificent': 0.9,
            'superb': 0.8,
            'excellent': 0.8,
            'brilliant': 0.8,
            'extraordinary': 0.9,
            'spectacular': 0.8,
            'sublime': 0.9,
            'divine': 0.9,
            
            # Very positive terms (0.6 - 0.8)
            'elegant': 0.7,
            'refined': 0.7,
            'sophisticated': 0.6,
            'complex': 0.6,
            'harmonious': 0.7,
            'balanced': 0.6,
            'smooth': 0.6,
            'silky': 0.7,
            'velvety': 0.7,
            'lush': 0.6,
            'rich': 0.6,
            'full-bodied': 0.5,
            'well-structured': 0.6,
            'concentrated': 0.5,
            
            # Positive terms (0.3 - 0.6)
            'fresh': 0.4,
            'crisp': 0.4,
            'clean': 0.4,
            'bright': 0.4,
            'vibrant': 0.5,
            'lively': 0.5,
            'juicy': 0.4,
            'fruity': 0.3,
            'aromatic': 0.3,
            'fragrant': 0.4,
            'pleasant': 0.4,
            'appealing': 0.4,
            'enjoyable': 0.5,
            'satisfying': 0.4,
            'delicious': 0.6,
            'tasty': 0.5,
            
            # Neutral descriptive terms (0.0 - 0.3)
            'dry': 0.0,
            'medium-bodied': 0.0,
            'light-bodied': 0.0,
            'oaked': 0.0,
            'unoaked': 0.0,
            'tannic': 0.0,
            'acidic': 0.0,
            'sweet': 0.0,
            'mineral': 0.1,
            'earthy': 0.1,
            'woody': 0.0,
            'spicy': 0.1,
            'herbal': 0.0,
            'floral': 0.2,
            
            # Slightly negative terms (-0.1 - -0.3)
            'simple': -0.2,
            'basic': -0.2,
            'ordinary': -0.3,
            'thin': -0.3,
            'light': -0.1,
            'weak': -0.4,
            'watery': -0.5,
            'bland': -0.4,
            'flat': -0.4,
            'dull': -0.4,
            'unremarkable': -0.3,
            
            # Negative terms (-0.4 - -0.6)
            'harsh': -0.5,
            'rough': -0.4,
            'coarse': -0.4,
            'aggressive': -0.4,
            'overpowering': -0.4,
            'unbalanced': -0.5,
            'clumsy': -0.5,
            'awkward': -0.4,
            'disjointed': -0.5,
            'hot': -0.4,  # referring to alcohol
            'burning': -0.5,
            
            # Very negative terms (-0.6 - -0.8)
            'faulty': -0.7,
            'flawed': -0.7,
            'off': -0.6,
            'tainted': -0.7,
            'spoiled': -0.8,
            'sour': -0.6,
            'bitter': -0.4,  # can be positive in some contexts
            'astringent': -0.3,
            'medicinal': -0.6,
            'chemical': -0.7,
            
            # Extremely negative terms (-0.8 - -1.0)
            'corked': -0.9,
            'oxidized': -0.8,
            'maderized': -0.8,
            'vinegary': -0.9,
            'rotten': -1.0,
            'putrid': -1.0,
            'disgusting': -0.9,
            'undrinkable': -0.9,
            'awful': -0.8,
            'terrible': -0.8,
            
            # Texture terms
            'creamy': 0.4,
            'buttery': 0.3,
            'oily': -0.2,
            'chalky': -0.3,
            'gritty': -0.4,
            'powdery': -0.3,
            
            # Finish terms
            'long finish': 0.6,
            'lingering': 0.5,
            'persistent': 0.4,
            'short finish': -0.3,
            'abrupt': -0.4,
            'cut off': -0.4,
            
            # Age-related terms
            'mature': 0.3,
            'developed': 0.2,
            'evolved': 0.2,
            'youthful': 0.1,
            'young': 0.0,
            'over the hill': -0.6,
            'past its prime': -0.5,
            'tired': -0.4,
            
            # Integration terms
            'integrated': 0.5,
            'seamless': 0.6,
            'cohesive': 0.5,
            'unified': 0.4,
            'disjointed': -0.5,
            'fragmented': -0.4,
        }
        
        return lexicon
    
    def get_positive_terms(self) -> List[str]:
        """Get list of positive wine terms"""
        return [term for term, score in self.lexicon.items() if score > 0]
    
    def get_negative_terms(self) -> List[str]:
        """Get list of negative wine terms"""
        return [term for term, score in self.lexicon.items() if score < 0]
    
    def get_neutral_terms(self) -> List[str]:
        """Get list of neutral wine terms"""
        return [term for term, score in self.lexicon.items() if score == 0]
    
    def save_to_json(self, file_path: str):
        """Save lexicon to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, file_path: str):
        """Load lexicon from JSON file"""
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)