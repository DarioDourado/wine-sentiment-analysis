#!/usr/bin/env python3
"""
🍷 Wine Sentiment Analysis System
Sistema avançado de análise de sentimento para avaliações de vinhos
Author: Dário Dourado
Version: 2.0.0
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config.settings import Settings
from utils.logger_config import setup_logger
from data_processing.data_loader import WineDataLoader
from data_processing.data_cleaner import WineDataCleaner
from data_processing.data_enricher import WineDataEnricher
from analysis.sentiment_analyzer import WineSentimentAnalyzer
from visualization.terminal_display import TerminalDisplay
from visualization.dashboard import launch_dashboard
from visualization.chart_generator import ChartGenerator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Wine Sentiment Analysis System')
    parser.add_argument('--mode', choices=['dashboard', 'charts', 'terminal', 'all'], 
                       default='interactive', help='Analysis mode')
    parser.add_argument('--input', type=str, default='data/raw/avaliacao_vinhos.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/',
                       help='Output directory')
    parser.add_argument('--language', choices=['en', 'pt'], default='en',
                       help='Interface language')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    return parser.parse_args()

def show_interactive_menu():
    """Show interactive menu for user choice"""
    print("\n" + "="*70)
    print("🎨 WINE SENTIMENT ANALYSIS - HOW TO VIEW RESULTS?")
    print("="*70)
    print("1️⃣  🌐 Interactive Dashboard")
    print("    └─ Web interface with filters and interactive charts")
    print()
    print("2️⃣  📊 Advanced Visualizations (With charts)")
    print("    └─ 11 charts including word clouds and NLP analysis")
    print()
    print("3️⃣  📋 Complete Terminal Analysis (No charts)")
    print("    └─ Detailed summary with NLP insights and correlations")
    print()
    print("4️⃣  🔄 Process All Modes")
    print("    └─ Generate charts + show terminal summary + prepare dashboard")
    print()
    print("5️⃣  🚪 Exit")
    print("    └─ Process data only and exit")
    print("="*70)
    
    while True:
        try:
            choice = input("\n👉 Enter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ Invalid input. Enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\n👋 Program interrupted by user.")
            return "5"

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(verbose=args.verbose)
    logger.info("🍷 Starting Wine Sentiment Analysis System")
    
    try:
        # Initialize settings
        settings = Settings()
        
        # Load and process data
        logger.info("📁 Loading data...")
        loader = WineDataLoader(settings)
        df = loader.load_csv(args.input)
        
        if df.empty:
            logger.error("❌ No data loaded. Exiting.")
            return False
        
        # Clean data
        logger.info("🧹 Cleaning data...")
        cleaner = WineDataCleaner(settings)
        df_clean = cleaner.clean_data(df)
        
        # Enrich data
        logger.info("✨ Enriching data...")
        enricher = WineDataEnricher(settings)
        df_enriched = enricher.enrich_data(df_clean)
        
        # Sentiment analysis
        logger.info("🔬 Performing sentiment analysis...")
        analyzer = WineSentimentAnalyzer(settings)
        df_final = analyzer.analyze_sentiment(df_enriched)
        
        # Save processed data
        output_file = Path(args.output) / 'wine_sentiment_data_processed.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_file, index=False)
        logger.info(f"💾 Data saved to: {output_file}")
        
        # Determine mode
        if args.mode == 'interactive':
            mode = show_interactive_menu()
        else:
            mode_map = {'dashboard': '1', 'charts': '2', 'terminal': '3', 'all': '4'}
            mode = mode_map.get(args.mode, '3')
        
        # Execute chosen mode
        if mode == '1':
            logger.info("🌐 Launching dashboard...")
            launch_dashboard()
            
        elif mode == '2':
            logger.info("📊 Generating visualizations...")
            chart_gen = ChartGenerator(settings)
            chart_gen.generate_all_charts(df_final)
            
        elif mode == '3':
            logger.info("📋 Showing terminal analysis...")
            terminal = TerminalDisplay(settings)
            terminal.show_complete_analysis(df_final)
            
        elif mode == '4':
            logger.info("🔄 Processing all modes...")
            # Generate charts
            chart_gen = ChartGenerator(settings)
            chart_gen.generate_all_charts(df_final)
            # Show terminal analysis
            terminal = TerminalDisplay(settings)
            terminal.show_complete_analysis(df_final)
            # Prepare dashboard
            logger.info("🌐 Dashboard prepared. Run 'streamlit run src/visualization/dashboard.py'")
            
        elif mode == '5':
            logger.info("👋 Processing complete. Exiting.")
        
        logger.info("✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)