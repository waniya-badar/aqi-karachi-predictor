#!/bin/bash

# AQI Predictor Karachi - Complete Setup Script
# Run this to set up the entire project structure

echo "========================================="
echo "AQI Predictor Karachi - Project Setup"
echo "========================================="
echo ""

# Create main directories
echo "Creating directory structure..."
mkdir -p .github/workflows
mkdir -p data/raw
mkdir -p data/alerts
mkdir -p models/saved_models
mkdir -p models/explanations
mkdir -p notebooks/plots
mkdir -p pipelines
mkdir -p src
mkdir -p streamlit_app

echo "✓ Directories created"
echo ""

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
ENV/

# Environment variables
.env
.env.local

# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Distribution / packaging
build/
dist/
*.egg-info/

# Models
models/saved_models/*.pkl
models/saved_models/*.h5

# Data
data/raw/*
!data/raw/.gitkeep

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Logs
*.log

# Plots (optional - comment out if you want to commit plots)
notebooks/plots/*.png
notebooks/plots/*.jpg
EOF

echo "✓ .gitignore created"
echo ""

# Create .env.example
echo "Creating .env.example..."
cat > .env.example << 'EOF'
# AQICN API Configuration
AQICN_API_KEY=your_api_key_here

# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=aqi_karachi

# Location Configuration
KARACHI_STATION_ID=@8762
EOF

echo "✓ .env.example created"
echo ""

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core Data Processing
requests==2.31.0
pymongo==4.6.1
pandas==2.1.4
numpy==1.24.3

# Machine Learning
scikit-learn==1.3.2

# Configuration
python-dotenv==1.0.0

# Web Dashboard
streamlit==1.29.0
plotly==5.18.0

# Model Explainability
shap==0.44.0

# Utilities
schedule==1.2.0

# Data Analysis & Visualization
jupyter==1.0.0
matplotlib==3.8.2
seaborn==0.13.0
EOF

echo "✓ requirements.txt created"
echo ""

# Create README.md
echo "Creating README.md..."
cat > README.md << 'EOF'
# 🌫️ Karachi AQI Predictor

Serverless Air Quality Index Prediction System for Karachi, Pakistan.

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Secrets**
   - Copy `.env.example` to `.env`
   - Add your AQICN API key
   - Add your MongoDB connection string

3. **Test Setup**
   ```bash
   python test_setup.py
   ```

4. **Collect Data**
   ```bash
   python pipelines/feature_pipeline.py
   ```

5. **Train Models** (after 100+ records)
   ```bash
   python pipelines/training_pipeline.py
   ```

6. **Launch Dashboard**
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Documentation

- [Complete Setup Guide](SETUP_GUIDE.md)
- [Workflow Explanation](COMPLETE_WORKFLOW_EXPLANATION.md)
- [Quick Commands](QUICK_COMMANDS.md)
- [Requirements Checklist](PROJECT_REQUIREMENTS_CHECKLIST.md)

## Features

- ✅ Hourly automated data collection
- ✅ 3 ML models (Random Forest, Gradient Boosting, Ridge)
- ✅ SHAP explainability
- ✅ Hazardous AQI alerts
- ✅ Interactive Streamlit dashboard
- ✅ 3-day AQI predictions
- ✅ CI/CD with GitHub Actions

Made with ❤️ for cleaner air in Karachi
EOF

echo "✓ README.md created"
echo ""

# Create placeholder files
echo "Creating placeholder files..."
touch data/raw/.gitkeep
touch data/alerts/.gitkeep
touch models/saved_models/.gitkeep
touch notebooks/plots/.gitkeep

echo "✓ Placeholder files created"
echo ""

# Summary
echo "========================================="
echo "✓ Project structure created successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy all Python files into their respective directories:"
echo "   - src/*.py"
echo "   - pipelines/*.py"
echo "   - streamlit_app/app.py"
echo "   - .github/workflows/*.yml"
echo ""
echo "2. Create virtual environment:"
echo "   python -m venv venv"
echo ""
echo "3. Activate virtual environment:"
echo "   # Windows:"
echo "   venv\\Scripts\\activate"
echo "   # Mac/Linux:"
echo "   source venv/bin/activate"
echo ""
echo "4. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "5. Create .env file from .env.example and add your keys"
echo ""
echo "6. Run test_setup.py to verify everything"
echo ""
echo "========================================="