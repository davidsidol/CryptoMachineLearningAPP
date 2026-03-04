"""
Standalone database setup script.
Run this first to create the CryptoML database and schema.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import database as db

if __name__ == "__main__":
    print("Setting up CryptoML database on .\\IDOLML ...")
    db.setup_database()
    print("Done! Database and tables are ready.")
