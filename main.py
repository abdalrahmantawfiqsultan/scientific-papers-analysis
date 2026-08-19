import sys
import os
from streamlit.web import cli as stcli
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == '__main__':
    print("Starting the HSLDE Knowledge Graph Dashboard...")
    # This programmatically runs: `streamlit run src/ui/app.py`
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
