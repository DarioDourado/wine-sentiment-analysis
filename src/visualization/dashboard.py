"""
Dashboard launch utilities
"""

import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def launch_dashboard():
    """Launch Streamlit dashboard"""
    dashboard_file = Path(__file__).parent / "dashboard_app.py"
    
    if not dashboard_file.exists():
        logger.error(f"❌ Dashboard file not found: {dashboard_file}")
        print("❌ Dashboard file not implemented yet.")
        print("💡 Please run the original dashboard.py file directly:")
        print("   streamlit run dashboard.py")
        return
    
    try:
        logger.info("🌐 Launching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_file)])
    except Exception as e:
        logger.error(f"❌ Error launching dashboard: {e}")
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running manually: streamlit run dashboard.py")