FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema (OpenCV + ffmpeg)
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do app
COPY app_licensed.py .
COPY web_service.py .

# Copiar módulos necessários
COPY engine/ ./engine/
COPY kivy/data/images/ ./kivy/data/images/

# Criar diretórios persistentes
RUN mkdir -p /app/data /app/saved_videos

# Porta do Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Variáveis de ambiente padrão
ENV PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

# Rodar app via web_service.py
CMD ["python", "web_service.py"]
