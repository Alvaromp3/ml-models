FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY graceland_soccer_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire graceland_soccer_model directory
COPY graceland_soccer_model/ .

EXPOSE 8501

# Healthcheck (optional, Railway handles this)
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Use PORT env var if available (Railway provides this), otherwise default to 8501
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0

