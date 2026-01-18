# GitHub Secrets Setup Guide

## Required Secrets for CI/CD Pipelines

### Step 1: Navigate to Repository Settings

1. Go to your GitHub repository
2. Click **Settings** (top menu)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret** button

### Step 2: Add MongoDB URI

**Secret Name**: `MONGODB_URI`

**Value**: Your MongoDB Atlas connection string
```
mongodb+srv://username:password@cluster-name.mongodb.net/dbname?retryWrites=true&w=majority
```

**How to get it**:
1. Go to MongoDB Atlas (https://cloud.mongodb.com/)
2. Click your cluster
3. Click "Connect"
4. Choose "Connect your application"
5. Copy the connection string
6. Replace `<password>` with your database user password

### Step 3: Add MongoDB Database Name

**Secret Name**: `MONGODB_DB_NAME`

**Value**: Your database name (usually `aqi_karachi` or `aqi_user`)

### Step 4: Add AQICN API Key

**Secret Name**: `AQICN_API_KEY`

**Value**: Your AQICN API key
```
abcd1234efgh5678ijkl9012mnop3456
```

**How to get it**:
1. Go to https://aqicn.org/api/
2. Sign up for free account
3. Copy your API token
4. Paste as secret value

### Step 5: GitHub Token (Optional)

**Secret Name**: `GITHUB_TOKEN`

**Value**: GitHub Personal Access Token (automatically provided by GitHub)
- This secret is automatically available
- Used for creating releases and comments

## Verification Steps

### 1. Verify Secrets Are Set

```bash
# Run this command to verify secrets are available
gh secret list
```

Expected output:
```
MONGODB_URI        Set
MONGODB_DB_NAME    Set
AQICN_API_KEY      Set
GITHUB_TOKEN       Set (default)
```

### 2. Test Connection

After setting secrets, trigger a workflow manually:

1. Go to **Actions** tab
2. Select **Hourly Feature Engineering Pipeline**
3. Click **Run workflow** → **Run workflow**
4. Wait for completion

### 3. Check Logs

1. Click on the workflow run
2. Expand any step to see logs
3. Look for "Connected to MongoDB" message
4. Verify data was fetched and stored

## Troubleshooting Secrets

### Secret Not Found Error
- Make sure secret name matches exactly (case-sensitive)
- Verify it's in the "Actions" section, not "Dependabot"
- Wait a few seconds after adding before running workflow

### Connection Failed Error
- Verify MongoDB URI is correct
- Check MongoDB IP whitelist includes GitHub Actions IPs
- Test URI locally: `python -c "from pymongo import MongoClient; MongoClient(uri)"`

### API Key Invalid Error
- Verify AQICN API key is correct
- Check API key hasn't expired
- Test key with curl: `curl "http://api.waqi.info/here/?token=YOUR_KEY"`

### Cannot Access Secrets
- Repository must be public or private with Actions enabled
- User must have Admin or Write access
- Organization secrets require special permissions

## Best Practices

✓ **DO**:
- Use strong, unique API keys
- Rotate keys periodically
- Store backups of secrets in secure location
- Use Organization secrets for shared access
- Review secret access logs regularly

✗ **DON'T**:
- Commit secrets to repository
- Share secrets in GitHub issues
- Use the same secret for multiple services
- Expose secrets in logs or error messages
- Leave expired keys in secrets

## Updating Secrets

### Update a Secret
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Find the secret
3. Click **Update** button
4. Enter new value
5. Click **Update secret**

### Delete a Secret
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Find the secret
3. Click **Delete** button
4. Confirm deletion

## Organization Secrets

For shared access across multiple repositories:

1. Go to Organization Settings
2. Click **Secrets and variables** → **Actions**
3. Click **New organization secret**
4. Select repositories that can access
5. Set secret value
6. Organization members can use in their workflows

## Example Workflow Using Secrets

```yaml
- name: Connect to MongoDB
  env:
    MONGODB_URI: ${{ secrets.MONGODB_URI }}
    MONGODB_DB_NAME: ${{ secrets.MONGODB_DB_NAME }}
  run: |
    python -c "
    from pymongo import MongoClient
    client = MongoClient(os.environ['MONGODB_URI'])
    db = client[os.environ['MONGODB_DB_NAME']]
    print(f'Connected to {db.name}')
    "
```

## Security Considerations

### IP Whitelisting for MongoDB
1. Go to MongoDB Atlas
2. Click Network Access
3. Add GitHub Actions IP ranges:
   - `0.0.0.0/0` (allows all IPs, less secure)
   - Or specific IP ranges (more secure)

### Secret Rotation Schedule
- Rotate API keys: Every 3 months
- Rotate MongoDB password: Every 6 months
- Review secret access: Monthly

### Audit Trail
- GitHub logs all secret access
- MongoDB logs connections
- Check logs for suspicious activity

## Verification Checklist

- [ ] MONGODB_URI secret created and tested
- [ ] MONGODB_DB_NAME secret created and tested
- [ ] AQICN_API_KEY secret created and tested
- [ ] MongoDB whitelist includes GitHub Actions
- [ ] First workflow run completed successfully
- [ ] Data appears in MongoDB after feature pipeline
- [ ] Models trained and saved after training pipeline
- [ ] No secrets exposed in logs or errors

## Support Resources

- GitHub Secrets Docs: https://docs.github.com/en/actions/security-guides/encrypted-secrets
- MongoDB Connection: https://docs.mongodb.com/manual/reference/connection-string/
- AQICN API: https://aqicn.org/api/
- GitHub Actions: https://docs.github.com/en/actions

---

Once all secrets are configured, your CI/CD pipelines are ready to run automatically!
