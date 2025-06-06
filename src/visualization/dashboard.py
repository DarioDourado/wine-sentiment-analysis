"""
Dashboard launch utilities
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def launch_dashboard():
    """Launch Streamlit dashboard"""
    # Procurar o dashboard na raiz do projeto
    project_root = Path(__file__).parent.parent.parent
    dashboard_file = project_root / "dashboard.py"
    
    if not dashboard_file.exists():
        logger.error(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        logger.info("🌐 Launching Streamlit dashboard...")
        print("🌐 Starting Streamlit dashboard...")
        print(f"📁 Dashboard file: {dashboard_file}")
        print("🌍 Dashboard will open at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        
        # Mudar para o diretório raiz do projeto (onde estão as traduções)
        os.chdir(project_root)
        
        # Launch streamlit
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file), 
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error launching dashboard: {e}")
        print(f"❌ Error launching dashboard: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Dashboard stopped by user")
        print("\n👋 Dashboard stopped by user")
        return True