# Complete CI/CD Pipeline Setup & Execution Guide

## System Overview
Your AQI Predictor has a fully automated CI/CD pipeline with 4 scheduled workflows that run 24/7:
- **Hourly Feature Pipeline** - Collects real-time AQI data
- **Daily Training Pipeline** - Trains ML models
- **Weekly EDA & Explainability** - Analyzes data patterns
- **Inference Pipeline** - Generates predictions

---

## PHASE 1: LOCAL ENVIRONMENT SETUP

### Step 1.1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/aqi-predictor-karachi.git
cd aqi-predictor-karachi
```

### Step 1.2: Create Python Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 1.3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1.4: Create .env File
Create a file named `.env` in the project root with your credentials:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/aqi_karachi?retryWrites=true&w=majority
MONGODB_DB_NAME=aqi_karachi
AQICN_API_KEY=your_api_key_here
```

**How to get credentials:**
- MongoDB URI: MongoDB Atlas → Cluster → Connect → Connection String
- AQICN API Key: https://aqicn.org/api/ → Sign up → Copy token

### Step 1.5: Test Local Environment
```bash
# Run standalone tests
python test_standalone.py

# Expected output:
# ✓ MongoDB connection successful
# ✓ API connection successful
# ✓ Data pipeline working
# ✓ Model training working
```

---

## PHASE 2: GITHUB REPOSITORY SETUP

### Step 2.1: Push Code to GitHub
```bash
git add .
git commit -m "Initial commit: AQI Predictor CI/CD System"
git push origin main
```

### Step 2.2: Verify Workflows Are Present
1. Go to your GitHub repository
2. Click **Code** tab
3. Verify `.github/workflows/` folder contains 4 YAML files:
   - `hourly-feature-pipeline.yml`
   - `daily-training-pipeline.yml`
   - `weekly-eda-explainability.yml`
   - `inference-pipeline.yml`

### Step 2.3: Enable GitHub Actions
1. Go to **Settings** → **Actions** → **General**
2. Under "Actions permissions" select **Allow all actions and reusable workflows**
3. Click **Save**

---

## PHASE 3: CONFIGURE GITHUB SECRETS (CRITICAL)

### Step 3.1: Navigate to Secrets Settings
1. Go to GitHub Repository
2. Click **Settings** (top menu)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret** button

### Step 3.2: Add Secret 1 - MongoDB URI
- **Name**: `MONGODB_URI`
- **Value**: `mongodb+srv://username:password@cluster.mongodb.net/aqi_karachi?retryWrites=true&w=majority`
- Click **Add secret**

### Step 3.3: Add Secret 2 - MongoDB Database Name
- **Name**: `MONGODB_DB_NAME`
- **Value**: `aqi_karachi`
- Click **Add secret**

### Step 3.4: Add Secret 3 - AQICN API Key
- **Name**: `AQICN_API_KEY`
- **Value**: Your API token (get from https://aqicn.org/api/)
- Click **Add secret**

### Step 3.5: Verify Secrets Configured
1. Go to Settings → Secrets and variables → Actions
2. You should see 3 secrets listed:
   ```
   MONGODB_URI ✓
   MONGODB_DB_NAME ✓
   AQICN_API_KEY ✓
   ```

---

## PHASE 4: VERIFY CI/CD PIPELINE SETUP

### Step 4.1: Check Workflow Files
View each workflow file to understand what it does:

**File 1: hourly-feature-pipeline.yml**
- **Schedule**: Runs every hour at XX:00
- **What it does**: 
  - Fetches real-time AQI data from AQICN API
  - Calculates 46 engineered features
  - Stores data in MongoDB
  - Duration: 5-10 minutes

**File 2: daily-training-pipeline.yml**
- **Schedule**: Runs daily at 2:00 AM UTC
- **What it does**:
  - Fetches all historical data from MongoDB
  - Trains 3 ML models (Ridge, Gradient Boosting, Random Forest)
  - Evaluates model performance
  - Saves best model (Ridge with 99.47% accuracy)
  - Duration: 20-30 minutes

**File 3: weekly-eda-explainability.yml**
- **Schedule**: Runs weekly on Sunday at 1:00 AM UTC
- **What it does**:
  - Performs exploratory data analysis
  - Generates 9 visualization plots
  - Runs SHAP & LIME explainability analysis
  - Uploads plots to artifacts
  - Duration: 30-45 minutes

**File 4: inference-pipeline.yml**
- **Schedule**: Runs every 6 hours (0:00, 6:00, 12:00, 18:00 UTC)
- **What it does**:
  - Loads trained model
  - Generates next 24-hour AQI predictions
  - Stores predictions in MongoDB
  - Duration: 10-15 minutes

### Step 4.2: Understand Workflow Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    HOURLY (Every Hour)                      │
│  Feature Pipeline: API → Features → MongoDB                 │
│  └─ Run Duration: 5-10 min                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   DAILY (2:00 AM UTC)                        │
│  Training Pipeline: MongoDB → Train → Save Models           │
│  └─ Run Duration: 20-30 min                                  │
│  └─ Outputs: ridge_latest.pkl, gradient_boost_latest.pkl    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                WEEKLY (Sunday 1:00 AM UTC)                   │
│  EDA: Analyze → Visualize → Explain                         │
│  └─ Run Duration: 30-45 min                                  │
│  └─ Outputs: 9 plots + SHAP/LIME analysis                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            EVERY 6 HOURS (0, 6, 12, 18 UTC)                 │
│  Inference Pipeline: Model → Predictions                    │
│  └─ Run Duration: 10-15 min                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## PHASE 5: TRIGGER & TEST WORKFLOWS

### Step 5.1: Manually Trigger Feature Pipeline (First Test)
1. Go to GitHub Repository
2. Click **Actions** tab (top menu)
3. Click **Hourly Feature Engineering Pipeline** (left sidebar)
4. Click **Run workflow** button (top right)
5. Select **Branch**: main
6. Click **Run workflow** (green button)

**Expected execution time**: 5-10 minutes

### Step 5.2: Monitor Workflow Execution
1. Click the running workflow name
2. Click **build** job
3. Expand each step to see logs:
   - "Checkout code"
   - "Set up Python"
   - "Install dependencies"
   - "Run feature pipeline"
   - "Upload artifact (logs)"

### Step 5.3: Verify Success
Look for green checkmark ✓ and logs showing:
```
✓ Connected to MongoDB
✓ Fetched AQI data
✓ Calculated 46 features
✓ Stored in MongoDB successfully
✓ Log saved: logs/feature_pipeline_log.json
```

### Step 5.4: Check MongoDB Data
1. Go to MongoDB Atlas (https://cloud.mongodb.com)
2. Click your cluster
3. Click **Collections**
4. Select **features** collection
5. Click **Find** → View recent records (should have new data!)

### Step 5.5: Manually Trigger Training Pipeline (Second Test)
1. Go to Actions → **Daily Model Training Pipeline**
2. Click **Run workflow** → **Run workflow**

**Expected execution time**: 20-30 minutes

### Step 5.6: Monitor Training Pipeline
In the workflow logs, you should see:
```
✓ Loaded feature data: X records
✓ Training Ridge Regression...
✓ Training Gradient Boosting...
✓ Training Random Forest...
✓ Best model: Ridge (R² = 0.9947)
✓ Saved models to saved_models/
```

### Step 5.7: Verify Models Saved
1. Go to GitHub → **Code** tab
2. Click **models/saved_models/**
3. Verify these files exist:
   - `ridge_latest.pkl`
   - `gradient_boost_latest.pkl`
   - `feature_names.json`
   - `model_registry.json`

---

## PHASE 6: CONFIGURE AUTOMATIC SCHEDULES

### Step 6.1: Understand Your Schedules

**Hourly Feature Pipeline**
- Runs at: Every hour (XX:00)
- Timezone: UTC
- Collection: Every hour = 24 runs/day = 8,760 runs/year

**Daily Training Pipeline**
- Runs at: 2:00 AM UTC (adjust if needed)
- Timezone: UTC = Eastern -5:00 / UK 0:00 / Pakistan +5:00
- Frequency: Once per day

**Weekly EDA Pipeline**
- Runs at: Sunday 1:00 AM UTC
- Timezone: UTC
- Frequency: Once per week

**Inference Pipeline**
- Runs at: 0:00, 6:00, 12:00, 18:00 UTC
- Frequency: 4 times per day

### Step 6.2: Modify Schedules (Optional)
If you want to change timing:

1. Go to GitHub → **Code** tab
2. Click `.github/workflows/hourly-feature-pipeline.yml`
3. Click ✎ (edit icon)
4. Find this section:
```yaml
schedule:
  - cron: '0 * * * *'    # Every hour at XX:00
```

**Cron time examples:**
```
'0 * * * *'      = Every hour at XX:00
'0 0 * * *'      = Daily at 00:00 (midnight UTC)
'0 2 * * *'      = Daily at 02:00 (2 AM UTC)
'0 0 * * 0'      = Weekly on Sunday at 00:00
'*/30 * * * *'   = Every 30 minutes
'0 */6 * * *'    = Every 6 hours (0, 6, 12, 18 UTC)
```

5. Modify the cron expression
6. Scroll down → Click **Commit changes**

---

## PHASE 7: SET UP LOCAL MONITORING

### Step 7.1: Run Streamlit Dashboard Locally
```bash
# Activate virtual environment (if not already)
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux

# Start dashboard
streamlit run streamlit_app/app.py
```

**Dashboard opens at**: http://localhost:8501

**Dashboard shows:**
- Real-time AQI value
- 7-day historical trend
- 24-hour predictions
- Model information
- Alert system status
- Data export option

### Step 7.2: View Pipeline Logs Locally
```bash
# Check feature pipeline logs (last 1000 runs)
cat logs/feature_pipeline_log.json

# Check training pipeline logs (last 500 runs)
cat logs/training_pipeline_log.json

# Count successful runs
Get-Content logs/feature_pipeline_log.json | ConvertFrom-Json | Where-Object {$_.status -eq "success"} | Measure-Object
```

### Step 7.3: Create Local Monitoring Script
Create `monitor_pipelines.py`:
```python
import json
from datetime import datetime
from pathlib import Path

def check_logs(log_file):
    with open(log_file) as f:
        logs = json.load(f)
    
    if logs:
        last_run = logs[-1]
        status = "✓ SUCCESS" if last_run['status'] == 'success' else "✗ FAILED"
        timestamp = last_run['timestamp']
        print(f"{log_file}: {status} - {timestamp}")
    else:
        print(f"{log_file}: No runs yet")

print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PIPELINE STATUS\n")
check_logs('logs/feature_pipeline_log.json')
check_logs('logs/training_pipeline_log.json')
```

Run with:
```bash
python monitor_pipelines.py
```

---

## PHASE 8: PRODUCTION DEPLOYMENT

### Step 8.1: Deploy Dashboard to Streamlit Cloud (Free)

1. Go to https://share.streamlit.io
2. Click **Deploy an app** button
3. Connect your GitHub account
4. Select repository: `aqi-predictor-karachi`
5. Select main branch
6. Select app script: `streamlit_app/app.py`
7. Click **Deploy**

**Your dashboard will be live at**: `https://share.streamlit.io/YOUR_USERNAME/aqi-predictor-karachi/main/streamlit_app/app.py`

### Step 8.2: Monitor GitHub Actions Dashboard
1. Go to GitHub → **Actions** tab
2. View all workflow runs
3. Set up email notifications:
   - GitHub Settings → Notifications → Check "Workflows"
   - Select email address
   - Save

### Step 8.3: Set Up GitHub Releases (Optional)
Create automatic releases with model artifacts:

1. Create a release manually:
   - Go to GitHub → **Releases** → **Create a new release**
   - Tag: `v1.0.0`
   - Title: "AQI Predictor v1.0.0"
   - Upload model files as attachments
   - Publish release

---

## PHASE 9: TROUBLESHOOTING GUIDE

### Issue: Workflow Not Running on Schedule

**Solution:**
1. Go to Settings → Actions → Permissions
2. Select "Allow all actions and reusable workflows"
3. Enable "Allow GitHub Actions to create and approve pull requests"
4. Check if main branch protection is enabled (disable if not needed)

### Issue: API Connection Failed

**Check:**
1. AQICN_API_KEY is correct → https://aqicn.org/api
2. API key hasn't expired (generate new if needed)
3. API request rate limit (free tier: 500 req/day)

**Log location**: Actions → Workflow run → Build → Logs

### Issue: MongoDB Connection Failed

**Check:**
1. MONGODB_URI is correct format
2. MongoDB cluster IP whitelist includes GitHub Actions IPs (allow 0.0.0.0/0)
3. Database user password doesn't contain special characters requiring URL encoding
4. MongoDB cluster is running (not paused)

**Fix IP whitelist:**
1. MongoDB Atlas → Cluster → Network Access
2. Add IP Address → `0.0.0.0/0` (allows all)
3. Confirm

### Issue: Models Not Saving

**Check:**
1. Verify `models/saved_models/` directory exists
2. Check MongoDB has enough data (need at least 100 records)
3. View training pipeline logs for errors

**Manual fix:**
```bash
mkdir -p models/saved_models
python -c "from pipelines.training_pipeline import train_all_models; train_all_models()"
```

### Issue: Dashboard Shows Old Data

**Solution:**
1. Wait for hourly pipeline to complete (5-10 min after the hour)
2. Manually refresh dashboard (Ctrl+R)
3. Clear Streamlit cache: `streamlit cache clear`
4. Restart dashboard

### Issue: "Permission Denied" Errors

**Solution:**
1. Ensure your GitHub user has admin access to repo
2. Check PAT (Personal Access Token) has correct scopes
3. Regenerate GitHub secrets if needed

---

## PHASE 10: VERIFICATION CHECKLIST

### Before Going Live ✓

- [ ] Local environment tests pass (`python test_standalone.py`)
- [ ] `.env` file created with all credentials
- [ ] GitHub repository created and code pushed
- [ ] GitHub Actions enabled in repository settings
- [ ] 3 secrets configured (MONGODB_URI, MONGODB_DB_NAME, AQICN_API_KEY)
- [ ] Feature pipeline triggered manually and completed successfully
- [ ] MongoDB has new records from feature pipeline
- [ ] Training pipeline triggered and models saved
- [ ] Model files exist in `models/saved_models/`
- [ ] Streamlit dashboard runs locally without errors
- [ ] Dashboard displays current data from MongoDB
- [ ] All 4 workflow schedules configured correctly
- [ ] GitHub Actions logs show no errors
- [ ] Email notifications configured for failed workflows

### Post-Deployment ✓

- [ ] Workflows run on schedule (check Actions tab daily)
- [ ] MongoDB records increasing (1 new record per hour)
- [ ] Models retrain daily (check model_registry.json timestamp)
- [ ] Streamlit dashboard shows current data
- [ ] EDA plots generate weekly
- [ ] No failed workflow runs
- [ ] API rate limits not exceeded

---

## PHASE 11: MONITORING & MAINTENANCE

### Daily Checks
```bash
# Check workflow status
# GitHub → Actions → View all workflow runs

# Expected:
# - Hourly pipeline: Latest run within last hour
# - All recent runs: GREEN checkmark
```

### Weekly Checks
```bash
# Verify data volume
# MongoDB Atlas → Collections → features → Count: should grow by ~168 (7 days × 24 hours)

# Check model performance
# logs/training_pipeline_log.json → Last entry → R² score should be > 0.99
```

### Monthly Tasks
```bash
# Review model accuracy trends
# GitHub → Actions → Daily-training-pipeline → See all runs

# Archive old logs (keep last 3 months)
# MongoDB → features collection → Delete records older than 90 days

# Update documentation if needed
# README.md, CI_CD_QUICK_REFERENCE.md
```

---

## PHASE 12: SCALING & OPTIMIZATION

### If You Hit API Rate Limits (500 requests/day)

**Current usage**: 24 requests/day (1 per hour)
**Headroom**: 95.2% unused

**To increase capacity:**
1. Upgrade AQICN API plan: https://aqicn.org/api/
2. Or reduce frequency: Change `'0 * * * *'` to `'0 */2 * * *'` (every 2 hours)

### If MongoDB Reaches Storage Limit

**Current MongoDB free tier**: 512 MB
**Data growth rate**: ~100 KB per day (can store ~5+ years of data)

**To add storage:**
1. MongoDB Atlas → Cluster → Tier → Upgrade M0 to M2
2. Or archive old data: Keep last 12 months in main collection, archive older data

### To Add More Cities/Stations

1. Edit `src/data_fetcher.py`
2. Add more city codes from https://aqicn.org/json-api/
3. Update MongoDB collections
4. Retrain models with combined data

---

## QUICK REFERENCE: Command Cheatsheet

```bash
# Local Testing
python test_standalone.py
streamlit run streamlit_app/app.py

# Manual Pipeline Runs
python pipelines/feature_pipeline.py
python pipelines/training_pipeline.py
python notebooks/eda_analysis.py

# Check Logs
cat logs/feature_pipeline_log.json
tail -f logs/feature_pipeline_log.json

# Deploy Changes
git add .
git commit -m "Description"
git push origin main

# Activate Virtual Environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux

# Install/Update Dependencies
pip install -r requirements.txt
```

---

## SUCCESS INDICATORS

### ✓ System is Running Properly When:
1. GitHub Actions tab shows all workflows with green checkmarks
2. Latest feature pipeline run is within the last hour
3. MongoDB collections have recent records (within last hour)
4. Streamlit dashboard shows current AQI value
5. Latest training pipeline run timestamp is today
6. Model files in `models/saved_models/` updated today

### ✓ Data Quality Is Good When:
1. Feature pipeline logs show no errors
2. MongoDB has 24+ new records per day
3. Model R² score is > 0.99
4. Predictions show realistic values (typically 0-500 AQI)

### ✓ Ready for Production When:
1. All checks above ✓ pass
2. 7+ days of successful automated runs
3. Streamlit dashboard is deployed to Streamlit Cloud
4. Email notifications configured for failures
5. Documented maintenance schedule in place

---

## SUPPORT & RESOURCES

- **MongoDB Atlas**: https://cloud.mongodb.com/
- **AQICN API**: https://aqicn.org/api/
- **GitHub Actions Docs**: https://docs.github.com/actions
- **Streamlit Docs**: https://docs.streamlit.io/
- **Project Repository**: https://github.com/YOUR_USERNAME/aqi-predictor-karachi

---

**Last Updated**: January 2026
**Status**: Production Ready ✓
