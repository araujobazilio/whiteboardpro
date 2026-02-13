📋 ESTRUTURA COMPLETA: DOCKER + COOLIFY + REDIS + WORKER
Vou consolidar tudo em um documento visual e detalhado:

🏗️ ARQUITETURA GERAL
┌─────────────────────────────────────────────────────────────────┐
│                     GITHUB REPOSITORY                           │
│              (seu-app-whiteboard)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    (git push main)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              GITHUB ACTIONS (CI/CD)                             │
│  - Build Docker images                                          │
│  - Push para Coolify                                            │
│  - Deploy automático                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         COOLIFY (VPS Hostinger - srv1341155)                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Nginx (Reverse Proxy + SSL)                              │  │
│  │ - whiteboardpro.com (seu domínio)                        │  │
│  │ - HTTPS automático (Let's Encrypt)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   WEB API    │   WORKER 1   │   WORKER 2   │    REDIS     │ │
│  │  (Gradio)    │  (OpenCV)    │  (OpenCV)    │   (Cache)    │ │
│  │  Port: 7860  │  Port: -     │  Port: -     │  Port: 6379  │ │
│  │              │              │              │              │ │
│  │ - Login      │ - Processa   │ - Processa   │ - Fila de    │ │
│  │ - Upload     │   vídeo 1    │   vídeo 2    │   jobs       │ │
│  │ - Status     │ - Salva em   │ - Salva em   │ - Cache de   │ │
│  │ - Download   │   cache      │   cache      │   resultados │ │
│  │ - Health     │              │              │              │ │
│  │ - Metrics    │              │              │              │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ OpenClaw (Monitoramento)                                 │  │
│  │ - Health checks da fila                                  │  │
│  │ - Alertas Telegram                                       │  │
│  │ - Métricas de processamento                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Volumes Persistentes                                     │  │
│  │ - /data/videos (vídeos processados)                      │  │
│  │ - /data/uploads (imagens enviadas)                       │  │
│  │ - /data/logs (logs da aplicação)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
📁 ESTRUTURA DE PASTAS (GitHub)
seu-app-whiteboard/
│
├── .github/
│   └── workflows/
│       └── deploy.yml                    # GitHub Actions (auto-deploy)
│
├── docker/
│   ├── web/
│   │   ├── Dockerfile                    # Build image Gradio
│   │   └── .dockerignore
│   │
│   ├── worker/
│   │   ├── Dockerfile                    # Build image Worker
│   │   └── .dockerignore
│   │
│   └── nginx/
│       ├── nginx.conf                    # Config Nginx (reverse proxy)
│       └── ssl/                          # Certificados SSL (Coolify gera)
│
├── src/
│   ├── app.py                            # API Gradio + endpoints
│   ├── worker.py                         # Worker background
│   ├── config.py                         # Configurações globais
│   ├── requirements.txt                  # Dependências Python
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video_processor.py            # generate_sketch_video()
│       ├── stripe_auth.py                # LicenseManager + send_otp_email()
│       ├── queue_manager.py              # Redis queue
│       ├── health_check.py               # Health endpoints
│       └── metrics.py                    # Métricas
│
├── docker-compose.yml                    # Local (desenvolvimento)
├── docker-compose.prod.yml               # Produção (Coolify)
├── coolify.json                          # Config Coolify
├── .env.example                          # Variáveis de exemplo
├── .gitignore
├── README.md
└── DEPLOYMENT.md                         # Guia de deploy
🔧 ARQUIVOS ESSENCIAIS
1. docker-compose.yml (LOCAL - Desenvolvimento)
yaml
version: '3.8'
 
services:
  # ========== WEB API (Gradio) ==========
  web:
    build:
      context: .
      dockerfile: docker/web/Dockerfile
    container_name: whiteboard-web
    ports:
      - "7860:7860"      # Gradio
      - "8000:8000"      # Metrics API
    environment:
      - REDIS_URL=redis://redis:6379
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
      - STRIPE_PRICE_ID=${STRIPE_PRICE_ID}
      - STRIPE_PAYMENT_LINK=${STRIPE_PAYMENT_LINK}
      - SMTP_SERVER=${SMTP_SERVER}
      - SMTP_PORT=${SMTP_PORT}
      - SMTP_EMAIL=${SMTP_EMAIL}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
      - ENVIRONMENT=development
    volumes:
      - ./src:/app/src
      - whiteboard_videos:/data/videos
      - whiteboard_uploads:/data/uploads
      - whiteboard_logs:/data/logs
    depends_on:
      - redis
    networks:
      - whiteboard-network
    restart: unless-stopped
 
  # ========== WORKER (Processamento de vídeo) ==========
  worker:
    build:
      context: .
      dockerfile: docker/worker/Dockerfile
    container_name: whiteboard-worker
    environment:
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    volumes:
      - ./src:/app/src
      - whiteboard_videos:/data/videos
      - whiteboard_uploads:/data/uploads
      - whiteboard_logs:/data/logs
    depends_on:
      - redis
    networks:
      - whiteboard-network
    restart: unless-stopped
 
  # ========== REDIS (Fila + Cache) ==========
  redis:
    image: redis:7-alpine
    container_name: whiteboard-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - whiteboard-network
    restart: unless-stopped
 
volumes:
  whiteboard_videos:
  whiteboard_uploads:
  whiteboard_logs:
  redis_data:
 
networks:
  whiteboard-network:
    driver: bridge
2. docker-compose.prod.yml (PRODUÇÃO - Coolify)
yaml
version: '3.8'
 
services:
  # ========== WEB API ==========
  web:
    image: ${DOCKER_REGISTRY}/whiteboard-web:latest
    container_name: whiteboard-web
    expose:
      - "7860"
      - "8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
      - STRIPE_PRICE_ID=${STRIPE_PRICE_ID}
      - STRIPE_PAYMENT_LINK=${STRIPE_PAYMENT_LINK}
      - SMTP_SERVER=${SMTP_SERVER}
      - SMTP_PORT=${SMTP_PORT}
      - SMTP_EMAIL=${SMTP_EMAIL}
      - SMTP_PASSWORD=${SMTP_PASSWORD}
      - ENVIRONMENT=production
    volumes:
      - whiteboard_videos:/data/videos
      - whiteboard_uploads:/data/uploads
      - whiteboard_logs:/data/logs
    depends_on:
      - redis
    networks:
      - whiteboard-network
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
 
  # ========== WORKER 1 ==========
  worker-1:
    image: ${DOCKER_REGISTRY}/whiteboard-worker:latest
    container_name: whiteboard-worker-1
    environment:
      - REDIS_URL=redis://redis:6379
      - WORKER_ID=worker-1
      - ENVIRONMENT=production
    volumes:
      - whiteboard_videos:/data/videos
      - whiteboard_uploads:/data/uploads
      - whiteboard_logs:/data/logs
    depends_on:
      - redis
    networks:
      - whiteboard-network
    restart: always
 
  # ========== WORKER 2 ==========
  worker-2:
    image: ${DOCKER_REGISTRY}/whiteboard-worker:latest
    container_name: whiteboard-worker-2
    environment:
      - REDIS_URL=redis://redis:6379
      - WORKER_ID=worker-2
      - ENVIRONMENT=production
    volumes:
      - whiteboard_videos:/data/videos
      - whiteboard_uploads:/data/uploads
      - whiteboard_logs:/data/logs
    depends_on:
      - redis
    networks:
      - whiteboard-network
    restart: always
 
  # ========== REDIS ==========
  redis:
    image: redis:7-alpine
    container_name: whiteboard-redis
    expose:
      - "6379"
    volumes:
      - redis_data:/data
    networks:
      - whiteboard-network
    restart: always
    command: redis-server --appendonly yes
 
volumes:
  whiteboard_videos:
    driver: local
  whiteboard_uploads:
    driver: local
  whiteboard_logs:
    driver: local
  redis_data:
    driver: local
 
networks:
  whiteboard-network:
    driver: bridge
3. Dockerfile (WEB - Gradio)
dockerfile
FROM python:3.11-slim
 
WORKDIR /app
 
# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Copiar requirements
COPY src/requirements.txt .
 
# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt
 
# Copiar código
COPY src/ .
 
# Criar diretórios
RUN mkdir -p /data/videos /data/uploads /data/logs
 
# Expor portas
EXPOSE 7860 8000
 
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
 
# Rodar app
CMD ["python", "app.py"]
4. Dockerfile (WORKER - Processamento)
dockerfile
FROM python:3.11-slim
 
WORKDIR /app
 
# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
# Copiar requirements
COPY src/requirements.txt .
 
# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt
 
# Copiar código
COPY src/ .
 
# Criar diretórios
RUN mkdir -p /data/videos /data/uploads /data/logs
 
# Rodar worker
CMD ["python", "worker.py"]
5. requirements.txt
gradio>=4.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
av>=11.0.0
Pillow>=10.0.0
stripe>=7.0.0
redis>=5.0.0
rq>=1.14.0
python-dotenv>=1.0.0
requests>=2.31.0
flask>=3.0.0
6. .env.example
bash
# ========== STRIPE ==========
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PRICE_ID=price_...
STRIPE_PAYMENT_LINK=https://buy.stripe.com/...
 
# ========== SMTP (Email) ==========
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=seu-email@gmail.com
SMTP_PASSWORD=sua-senha-ou-app-password
 
# ========== REDIS ==========
REDIS_URL=redis://redis:6379
 
# ========== AMBIENTE ==========
ENVIRONMENT=development
7. .github/workflows/deploy.yml (GitHub Actions)
yaml
name: Deploy to Coolify
 
on:
  push:
    branches:
      - main
 
jobs:
  deploy:
    runs-on: ubuntu-latest
 
    steps:
      - uses: actions/checkout@v3
 
      - name: Build and push Docker images
        run: |
          docker build -f docker/web/Dockerfile -t whiteboard-web:latest .
          docker build -f docker/worker/Dockerfile -t whiteboard-worker:latest .
 
      - name: Deploy to Coolify
        env:
          COOLIFY_API_KEY: ${{ secrets.COOLIFY_API_KEY }}
          COOLIFY_SERVER_ID: ${{ secrets.COOLIFY_SERVER_ID }}
          COOLIFY_APP_ID: ${{ secrets.COOLIFY_APP_ID }}
        run: |
          curl -X POST \
            -H "Authorization: Bearer $COOLIFY_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{"deployment_id": "'$COOLIFY_APP_ID'"}' \
            https://coolify.io/api/v1/applications/$COOLIFY_APP_ID/deploy
🔄 FLUXO DE PROCESSAMENTO
1. CLIENTE ACESSA APP
   ↓
   https://whiteboardpro.com (Nginx redireciona para web:7860)
   ↓
 
2. LOGIN
   - Email + Stripe payment link
   - Recebe OTP por email (SMTP)
   - Valida OTP
   - Cria sessão (30 dias)
   ↓
 
3. UPLOAD DE IMAGEM
   - POST /api/process
   - Valida autenticação (sessão)
   - Salva imagem em /data/uploads
   - Enfileira job no Redis
   - Retorna: {"job_id": "xyz123", "status": "queued"}
   ↓
 
4. WORKER PROCESSA
   - Worker 1 ou Worker 2 pega job da fila
   - Chama generate_sketch_video()
   - Processa por 1-5 minutos
   - Salva vídeo em /data/videos
   - Atualiza status no Redis: "completed"
   ↓
 
5. CLIENTE CONSULTA STATUS
   - GET /api/job/xyz123/status
   - Retorna: {"status": "processing", "progress": 45}
   - Quando pronto: {"status": "completed", "download_url": "..."}
   ↓
 
6. CLIENTE BAIXA VÍDEO
   - GET /api/job/xyz123/download
   - Retorna arquivo MP4
   ↓
 
7. MONITORAMENTO (OpenClaw)
   - Verifica saúde da fila
   - Conta jobs em processamento
   - Alerta se worker cair
   - Envia relatório via Telegram
📊 ENDPOINTS DA API
Método	Endpoint	Descrição
GET	/health	Health check
GET	/metrics	Métricas (fila, workers, CPU)
POST	/api/auth/send-otp	Envia OTP por email
POST	/api/auth/verify-otp	Valida OTP
POST	/api/process	Enfileira vídeo
GET	/api/job/{id}/status	Status do job
GET	/api/job/{id}/download	Download do vídeo
POST	/api/logout	Logout
🚀 PASSO A PASSO PARA DEPLOY
Passo 1: Preparar VPS
bash
# SSH na VPS
ssh root@srv1341155.hstgr.cloud
 
# Instalar Docker + Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
 
# Instalar Coolify
curl -fsSL https://get.coolify.io/install.sh | bash
Passo 2: Configurar Coolify
Acessar: https://seu-vps-ip:3000
Criar projeto "whiteboard-pro"
Adicionar aplicação Docker Compose
Conectar repositório GitHub
Passo 3: Configurar Variáveis de Ambiente
No Coolify, adicionar .env com:
STRIPE_SECRET_KEY
STRIPE_PRICE_ID
SMTP_*
etc
Passo 4: Deploy
bash
# No GitHub, fazer push para main
git push origin main
 
# GitHub Actions executa automaticamente
# Coolify faz deploy
📋 CHECKLIST PRÉ-DEPLOY
Repositório GitHub criado
Dockerfile web testado localmente
Dockerfile worker testado localmente
docker-compose.yml funciona localmente
Variáveis de ambiente configuradas
SMTP testado (envio de email)
Stripe em modo teste
GitHub Actions configurado
Coolify instalado na VPS
Domínio apontado para VPS
SSL configurado (Coolify gera automaticamente)
OpenClaw configurado para monitoramento
💾 VOLUMES PERSISTENTES
Volume	Função	Tamanho
whiteboard_videos	Vídeos processados	~500MB-1GB
whiteboard_uploads	Imagens enviadas	~100MB
whiteboard_logs	Logs da aplicação	~50MB
redis_data	Cache + fila	~100MB
🔐 SEGURANÇA
Variáveis sensíveis em .env (não commitar)
HTTPS obrigatório (Coolify com Let's Encrypt)
Redis sem senha (apenas rede interna)
Stripe em modo teste durante desenvolvimento
OTP válido por 10 minutos
Sessão válida por 30 dias
Rate limiting em endpoints de autenticação
✅ PRÓXIMOS PASSOS
Depois de consolidar essa estrutura, vamos:

Refatorar app.py (Gradio + endpoints)
Criar worker.py (processamento background)
Criar queue_manager.py (Redis)
Criar utils/ (funções auxiliares)
Testar localmente com docker-compose
Deploy na VPS via Coolify
Integrar OpenClaw para monitoramento