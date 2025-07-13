#!/usr/bin/env python3
"""
Simple script to run the Retail Pipeline Dashboard locally.
"""

import os
from app import app, refresh_all_data

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting Retail Pipeline Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
    
    # Ensure fresh data on app startup
    print("🔄 Loading fresh data from warehouse...")
    refresh_all_data()
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )