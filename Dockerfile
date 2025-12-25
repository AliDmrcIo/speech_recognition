FROM python:3.13.3

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Requirements kopyala ve yükle
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyaları
COPY . . 

# Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit çalıştır
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]