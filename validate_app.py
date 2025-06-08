#!/usr/bin/env python3

"""
This script validates that your Flask application loads correctly
before deployment to Python Anywhere.
"""

import os
import sys

try:
    print("Testing application loading...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Try to import Flask app
    print("Importing Flask application...")
    from app import app
    print("✓ Flask application imported successfully")
    
    # Check if required data files exist
    data_path = 'agriculture_dataset.csv'
    if os.path.exists(data_path):
        print(f"✓ Dataset found: {data_path}")
    else:
        print(f"✗ Missing dataset: {data_path}")
    
    # Check if model files exist
    model_path = 'app/models/random_forest_model.joblib'
    scaler_path = 'app/models/scaler.joblib'
    
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
    else:
        print(f"✗ Missing model file: {model_path}")
    
    if os.path.exists(scaler_path):
        print(f"✓ Scaler file found: {scaler_path}")
    else:
        print(f"✗ Missing scaler file: {scaler_path}")
    
    # Check template directory
    template_path = 'app/templates'
    if os.path.exists(template_path) and os.path.isdir(template_path):
        template_files = os.listdir(template_path)
        print(f"✓ Template directory found with {len(template_files)} files")
    else:
        print(f"✗ Missing template directory: {template_path}")
    
    # Check static directory
    static_path = 'app/static'
    if os.path.exists(static_path) and os.path.isdir(static_path):
        print(f"✓ Static directory found")
    else:
        print(f"✗ Missing static directory: {static_path}")
    
    print("\nApplication validation complete.")
    print("If all checks passed, your application is ready for deployment.")
    print("If any checks failed, address the issues before deploying.")
    
except Exception as e:
    print(f"Error during validation: {e}")
    sys.exit(1)
