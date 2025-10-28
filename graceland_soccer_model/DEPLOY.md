# 🚀 How to Deploy to Streamlit Cloud

## Quick Start

### Method 1: Deploy via Streamlit Cloud Web Interface

1. **Go to Streamlit Cloud**

   - Visit: [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "Sign up" or "Log in"

2. **Authorize GitHub**

   - Click "Sign in with GitHub"
   - Authorize Streamlit Cloud to access your repositories

3. **Create New App**

   - Click "New app" button
   - Select repository: `Alvaromp3/ml-models`
   - Branch: `main`
   - **Main file path**: `graceland_soccer_model/app.py`
   - App URL: Choose your subdomain (e.g., `graceland-soccer-demo`)
   - Click "Deploy!"

4. **Wait for Deployment**
   - Streamlit will install dependencies from `requirements.txt`
   - First deploy takes ~3-5 minutes
   - You'll get a live URL like: `https://graceland-soccer-demo.streamlit.app`

### Method 2: Deploy via Streamlit CLI

```bash
# Install Streamlit Cloud CLI (if not already installed)
pip install streamlit

# Login to Streamlit Cloud
streamlit login

# Deploy from the project directory
cd graceland_soccer_model
streamlit deploy
```

## Configuration Files

### `.streamlit/config.toml`

Already created with theme customization.

### `packages.txt`

Optional file for system-level dependencies.

## Troubleshooting

### Common Issues

**1. App won't deploy**

- Make sure `app.py` is in the root or specify correct path
- Check that `requirements.txt` exists and is valid

**2. Import errors**

- Ensure all dependencies are in `requirements.txt`
- Check Python version compatibility (3.10+)

**3. Large file size**

- `sample_catapult_data.csv` might be too large (>100MB)
- Consider using GitHub LFS or reduce sample data size

**4. Port conflicts**

- Streamlit Cloud handles port automatically
- No need to specify port in code

## After Deployment

✅ **Your app is live!**

- Share the URL with others
- Make changes to code
- Push to GitHub
- Streamlit Cloud auto-updates

## Updating the App

1. Make changes to `app.py` or other files
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```
3. Streamlit Cloud automatically redeploys (takes 1-2 minutes)

## Advanced Configuration

### Secrets Management

If you need API keys or secrets:

1. Go to Streamlit Cloud dashboard
2. Click "Settings"
3. Click "Secrets"
4. Add secrets in TOML format:
   ```toml
   my_secret = "value"
   ```
5. Access in code:
   ```python
   import streamlit as st
   secret = st.secrets["my_secret"]
   ```

### Custom Domain

1. Go to Settings → General
2. Add custom domain
3. Update DNS records

## Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Cloud Community](https://discuss.streamlit.io/)
- [Streamlit Cloud Pricing](https://streamlit.io/pricing)

## Support

Need help? Contact:

- 📧 Email: support@streamlit.io
- 💬 Discord: [Streamlit Discord](https://discord.gg/streamlit)
