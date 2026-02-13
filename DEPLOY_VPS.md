# Deploy WhiteboardMaker na VPS Hostinger

## Arquitetura Final

```
VPS Hostinger (srv1341155.hstgr.cloud)
│
├── Container 1: OpenClaw #1 (assistente pessoal - JÁ EXISTE)
│   └── Porta: 50145
│
├── Container 2: WhiteboardMaker (Gradio + SQLite)
│   ├── Porta: 7860
│   └── Volumes: /app/data (SQLite) + /app/saved_videos (vídeos)
│
└── Container 3: OpenClaw #2 (monitoramento do WhiteboardMaker)
    └── Porta: 50146
```

---

## PASSO 1: Conectar via SSH

```bash
ssh root@srv1341155.hstgr.cloud
```

---

## PASSO 2: Clonar repositório e buildar imagem

```bash
# Criar pasta do projeto
mkdir -p /opt/whiteboardmaker
cd /opt/whiteboardmaker

# Clonar repositório
git clone https://github.com/araujobazilio/whiteboardpro.git .

# Buildar imagem Docker
docker build -t whiteboardmaker:latest .
```

Isso vai demorar alguns minutos (instala Python, OpenCV, ffmpeg, etc.).

Para verificar se a imagem foi criada:
```bash
docker images | grep whiteboardmaker
```

---

## PASSO 3: Criar arquivo .env

```bash
cd /opt/whiteboardmaker

cat > .env << 'EOF'
# STRIPE (substitua pelos seus valores reais)
STRIPE_SECRET_KEY=sk_test_SUA_CHAVE_AQUI
STRIPE_PRICE_ID=price_SEU_PRICE_ID_AQUI
STRIPE_PAYMENT_LINK=https://buy.stripe.com/SEU_LINK_AQUI

# OPENCLAW #2 (preencher depois)
OPENCLAW_TELEGRAM_BOT_TOKEN=seu-bot-token-aqui
OPENCLAW_TELEGRAM_CHAT_ID=seu-chat-id-aqui
EOF
```

---

## PASSO 4: Implantar via Painel Hostinger

### Opção A: Pelo Painel Visual (recomendado)

1. Acesse: `hpanel.hostinger.com` → VPS → Gerenciador Docker
2. Clique em **"Criar"** (novo projeto Compose)
3. Nome do projeto: `whiteboardmaker`
4. Clique na aba **"Editor .yaml"**
5. Cole o conteúdo do `docker-compose.yml` (abaixo)
6. Configure as variáveis de ambiente
7. Clique **"Implantar"**

### YAML para colar no painel:

```yaml
version: '3.8'

services:
  whiteboardmaker:
    image: whiteboardmaker:latest
    container_name: whiteboardmaker
    ports:
      - "7860:7860"
    environment:
      - PORT=7860
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
      - STRIPE_PRICE_ID=${STRIPE_PRICE_ID}
      - STRIPE_PAYMENT_LINK=${STRIPE_PAYMENT_LINK}
    volumes:
      - whiteboardmaker_data:/app/data
      - whiteboardmaker_videos:/app/saved_videos
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/"]
      interval: 30s
      timeout: 10s
      start_period: 60s
      retries: 3

volumes:
  whiteboardmaker_data:
    driver: local
  whiteboardmaker_videos:
    driver: local
```

### Opção B: Via SSH (alternativa)

```bash
cd /opt/whiteboardmaker
docker-compose up -d
```

---

## PASSO 5: Instalar OpenClaw #2 pelo Catálogo Hostinger

1. Acesse: `hpanel.hostinger.com` → VPS → Gerenciador Docker
2. Vá no **Catálogo** de imagens
3. Procure **OpenClaw** e instale
4. Configure:
   - Nome: `whiteboardmaker-openclaw`
   - Porta: `50146:50145` (diferente do #1 que usa 50145)
   - Variáveis de ambiente:
     - `TELEGRAM_BOT_TOKEN` = token do bot dedicado ao WhiteboardMaker
     - `TELEGRAM_CHAT_ID` = seu chat ID
5. Implantar

---

## PASSO 6: Verificar se está tudo rodando

```bash
# Ver containers rodando
docker ps

# Ver logs do WhiteboardMaker
docker logs whiteboardmaker -f

# Testar acesso
curl http://localhost:7860/
```

O app estará acessível em: `http://srv1341155.hstgr.cloud:7860`

---

## COMO ATUALIZAR O APP

Quando fizer mudanças no código e push no GitHub:

```bash
ssh root@srv1341155.hstgr.cloud
cd /opt/whiteboardmaker
git pull origin main
docker build -t whiteboardmaker:latest .
docker-compose down
docker-compose up -d
```

Ou em uma linha:
```bash
cd /opt/whiteboardmaker && git pull && docker build -t whiteboardmaker:latest . && docker-compose down && docker-compose up -d
```

---

## TROUBLESHOOTING

### App não inicia
```bash
docker logs whiteboardmaker
```

### Porta ocupada
```bash
docker ps -a | grep 7860
docker stop <container_id>
```

### Rebuild completo
```bash
docker-compose down
docker rmi whiteboardmaker:latest
docker build -t whiteboardmaker:latest .
docker-compose up -d
```

### Ver banco SQLite
```bash
docker exec -it whiteboardmaker ls -la /app/data/
```

### Ver vídeos salvos
```bash
docker exec -it whiteboardmaker ls -la /app/saved_videos/
```
