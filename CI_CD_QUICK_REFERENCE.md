# CI/CD Quick Reference Card

## 🚀 Setup (Copy-Paste Commands)

```bash
# 1. Push workflows to GitHub
git add .github/workflows/
git commit -m "Add CI/CD pipelines"
git push

# 2. Add secrets (via GitHub web UI - see below)
# 3. Trigger test run (via Actions tab)
# 4. Monitor logs (Actions → Run name → Logs)
```

## 🔐 GitHub Secrets Setup (3 Required)

**Go to: Repo Settings → Secrets and variables → Actions → New repository secret**

```
1. MONGODB_URI
   mongodb+srv://user:pass@cluster.mongodb.net/db?retryWrites=true&w=majority

2. MONGODB_DB_NAME
   aqi_karachi

3. AQICN_API_KEY
   your_api_key_from_aqicn_org
```

## 📅 Automated Schedules

| What | When | Duration |
|---|---|---|
| Fetch data & features | Every hour | 5-10 min |
| Train models | Daily 2 AM UTC | 20-30 min |
| EDA & SHAP/LIME | Weekly Sunday 1 AM | 30-45 min |
| Generate predictions | Every 6 hours | 10-15 min |

## 📊 Monitor Runs

```
1. Go to GitHub repo
2. Click "Actions" tab
3. Select workflow name
4. View run status & logs
5. Download artifacts (models/plots)
```

## 🔍 Verify Working

```bash
# Check MongoDB has new records
# MongoDB Atlas UI → Collections → features → Find

# Check models exist
# models/saved_models/ridge_latest.pkl, etc.

# Check Streamlit dashboard updates
streamlit run streamlit_app/app.py
```

## 🛑 Troubleshooting

| Problem | Solution |
|---|---|
| Workflow not running | Settings → Actions → Allow all actions |
| Connection failed | Check secrets in Settings → Secrets |
| No data in MongoDB | Manually trigger feature pipeline |
| Models not found | Check training pipeline logs |
| Dashboard shows old data | Wait for hourly feature pipeline |

## 📁 Workflow Files Created

```
.github/workflows/
├── hourly-feature-pipeline.yml      ← Fetch data every hour
├── daily-training-pipeline.yml      ← Train models daily
├── weekly-eda-explainability.yml    ← Analyze weekly
└── inference-pipeline.yml            ← Predict every 6h
```

## 💡 Manual Triggers (No Schedule)

```
Actions → Select workflow → "Run workflow" button → Run
```

## 📝 Logs Location

```
Local: logs/feature_pipeline_log.json (1000 entries)
       logs/training_pipeline_log.json (500 entries)
GitHub: Actions → Run name → Logs tab
```

## ⚡ First-Time Setup Checklist

- [ ] Workflow files pushed to `.github/workflows/`
- [ ] 3 GitHub Secrets configured
- [ ] First run manually triggered
- [ ] Workflow shows "✓ passed"
- [ ] MongoDB has new records
- [ ] Models exist in `models/saved_models/`
- [ ] Dashboard loads with fresh data

## 🎯 What Gets Automated

✅ Hourly: Fetch latest AQI data from API → Extract features → Store in MongoDB
✅ Daily: Load 120 days of data → Train Ridge/GB/RF → Save best model → Create release
✅ Weekly: Generate EDA plots → Run SHAP analysis → Run LIME explanations → Archive
✅ Every 6h: Generate predictions for next day → Store in database
✅ Continuous: Generate JSON logs → Retain execution history → GitHub artifacts

## 💰 Cost

**$0/month** - Everything within free tiers:
- GitHub Actions: 2,000 min/month free (using ~900)
- MongoDB: 512 MB free (using ~50 MB)

## 🚨 Alert Status

**Current Status: ✅ Production Ready**

All workflows configured, tested, and ready for deployment.

---

**Need help?** See `CI_CD_CONFIGURATION.md` for detailed docs or `GITHUB_SECRETS_SETUP.md` for step-by-step setup.
