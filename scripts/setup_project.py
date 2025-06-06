#!/usr/bin/env python3
"""
Setup script for Wine Sentiment Analysis project
Creates directories, downloads dependencies, and sets up initial configuration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure"""
    base_dir = Path.cwd()
    
    directories = [
        "src/analysis",
        "src/visualization", 
        "src/data_processing",
        "src/utils",
        "src/config",
        "data/raw",
        "data/processed",
        "data/exports",
        "assets/images",
        "assets/translations",
        "logs",
        "tests",
        "docs",
        "scripts"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directory structure created successfully!")

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False
    
    return True

def setup_nltk_data():
    """Download required NLTK data"""
    print("📥 Setting up NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except ImportError:
        print("⚠️ NLTK not installed, skipping NLTK setup")
    except Exception as e:
        print(f"⚠️ Error setting up NLTK: {e}")

def create_sample_data():
    """Create sample wine data if no data exists"""
    data_dir = Path("data/raw")
    sample_file = data_dir / "sample_wine_reviews.csv"
    
    if not sample_file.exists():
        print("📋 Creating sample data file...")
        
        sample_data = """wine,winery,category,designation,varietal,appellation,reviewer,review,rating,price,alcohol
"Château Margaux 2015","Château Margaux","Red Wine","","Cabernet Sauvignon","Margaux, France","Robert Parker","This exceptional wine shows brilliant complexity with notes of cassis, graphite, and violets. The tannins are silky and well-integrated, leading to a long, elegant finish.",98,800,13.5
"Kendall-Jackson Vintner's Reserve 2019","Kendall-Jackson","White Wine","Vintner's Reserve","Chardonnay","California","Wine Spectator","A well-balanced Chardonnay with tropical fruit flavors, vanilla oak, and crisp acidity. Clean finish with good length.",87,25,14.1
"Dom Pérignon 2012","Moët & Chandon","Sparkling","","Chardonnay Blend","Champagne, France","James Suckling","Extraordinary champagne with fine bubbles, mineral complexity, and notes of white flowers. Exceptional vintage showing great aging potential.",96,200,12.5
"""
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        print(f"✅ Sample data created: {sample_file}")

def main():
    """Main setup function"""
    print("🍷 Wine Sentiment Analysis - Project Setup")
    print("=" * 50)
    
    # Create directories
    create_directory_structure()
    
    # Create sample data
    create_sample_data()
    
    # Install dependencies
    if Path("requirements.txt").exists():
        install_dependencies()
        setup_nltk_data()
    else:
        print("⚠️ requirements.txt not found. Run manually: pip install -r requirements.txt")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your wine review CSV file in data/raw/")
    print("2. Run: python main.py")
    print("3. Choose analysis mode (dashboard, charts, or terminal)")
    print("\n🚀 Happy analyzing!")

if __name__ == "__main__":
    main()