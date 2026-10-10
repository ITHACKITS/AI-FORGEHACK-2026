FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# rasterio's wheel needs libexpat, which the slim image doesn't ship
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# scikit-learn 1.6.1 is the version the fire-scorer pickle was trained with
RUN pip install --no-cache-dir -r requirements.txt "scikit-learn==1.6.1"
COPY . .
EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
