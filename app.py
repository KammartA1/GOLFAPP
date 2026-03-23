"""
Golf Quant Engine — Secondary Streamlit Entry Point
=====================================================
Delegates to dashboard.py (the canonical entry point).

Usage:
    streamlit run app.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and execute the refactored dashboard
# This file exists for backwards compatibility — dashboard.py is canonical.
exec(open(str(Path(__file__).resolve().parent / "dashboard.py")).read())
