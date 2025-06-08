#!/usr/bin/env python3
"""
MCTX SAP HANA Management CLI

A command-line tool for managing the SAP HANA database used with MCTX.
"""

import argparse
import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional

# Add the parent directory to the path so we can import mctx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the HANA modules
try:
    from mctx.enterprise.hana_management import main as hana_main
    HAS_HANA_MODULES = True
except ImportError:
    HAS_HANA_MODULES = False

def main():
    """Main entry point for the CLI tool."""
    if not HAS_HANA_MODULES:
        print("Error: Required modules not found")
        print("Make sure MCTX is installed with enterprise components")
        print("Install requirements: pip install hdbcli==2.19.21")
        return 1
    
    # Delegate to the main function from hana_management
    return hana_main()

if __name__ == "__main__":
    sys.exit(main())