# 🎨 Image to Sketch Animation

Sistema completo para transformar imagens em vídeos de animação whiteboard estilo "mão desenhando".

Baseado em: https://github.com/daslearning-org/image-to-animation-offline

## 🌟 Características

- ✅ Interface web intuitiva com Gradio
- ✅ Gera vídeos MP4 de alta qualidade
- ✅ Animação progressiva estilo whiteboard
- ✅ Configurações personalizáveis (velocidade, qualidade, FPS)
- ✅ **🆕 Processamento em lote (batch processing)**
- ✅ **🆕 Upload em massa de múltiplas imagens**
- ✅ **🆕 Download automático em arquivo ZIP**
- ✅ **🆕 Processamento paralelo (até 4 imagens simultâneas)**
- ✅ **🆕 Lógica de Desenho Inteligente (desenha objeto por objeto)**
- ✅ **🆕 Movimento Fluido da Mão (sem "pulos" bruscos)**
- ✅ **🆕 Compressão de Vídeo Otimizada (arquivos leves)**
- ✅ Conversão automática para H264
- ✅ Funciona 100% offline (sem internet necessária)
- ✅ Perfeito para criar vídeos educacionais e apresentações

## 📋 Requisitos

- Python 3.8 ou superior
- Windows, Linux ou Mac

## 🚀 Instalação

### 1. Ativar ambiente virtual (se já criado):
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Instalar dependências:
```powershell
pip install -r requirements.txt
```

## 🎬 Como Usar

### Iniciar o aplicativo:
```powershell
python app.py
```

O aplicativo abrirá automaticamente no navegador em: `http://localhost:7860`

### 📋 Opções de Processamento

O sistema agora oferece **duas formas de processamento**:

---

## 🖼️ Processamento Individual

### Passos:

1. **Upload da Imagem**
   - Vá para a aba **"🖼️ Processamento Individual"**
   - Clique em "Upload da Imagem"
   - Selecione uma imagem (PNG, JPG, JPEG)
   - O sistema mostrará automaticamente as informações da imagem

2. **Configurar Parâmetros**
   - **Split Length**: Tamanho da divisão em grid
     - Menor (5-10) = Mais lento e detalhado
     - Maior (15-30) = Mais rápido
   
   - **Frame Rate (FPS)**: Qualidade do vídeo
     - 15-24: Vídeo leve
     - 30: Padrão (recomendado)
     - 60: Alta qualidade
   
   - **Skip Rate**: Velocidade do desenho
     - 1-5: Lento e suave
     - 5-10: Equilibrado
     - 10-20: Rápido
   
   - **Duração Final**: Tempo que a imagem completa aparece no final (1-10 segundos)

3. **Gerar Vídeo**
   - Clique em "🚀 Gerar Vídeo"
   - Aguarde o processamento (pode levar alguns minutos)
   - O vídeo será exibido automaticamente quando pronto

4. **Download**
   - Clique no botão de download no player de vídeo
   - Ou acesse a pasta `saved_videos/`

---

## 📦 Processamento em Lote (NOVO!)

### ⚡ Vantagens:
- Processa múltiplas imagens simultaneamente
- Download automático em arquivo ZIP organizado
- Processamento paralelo (até 4 imagens ao mesmo tempo)
- Relatório detalhado de processamento

### Passos:

1. **Upload em Massa**
   - Vá para a aba **"📦 Processamento em Lote"**
   - Clique em "Upload de Múltiplas Imagens"
   - Selecione várias imagens (arraste ou clique)
   - O sistema mostrará quantidade e tamanho total

2. **Configurar Parâmetros do Lote**
   - Use os mesmos parâmetros do processamento individual
   - As configurações se aplicam a todas as imagens do lote

3. **Processar Lote**
   - Clique em "🚀 Processar Lote"
   - Acompanhe o progresso em tempo real
   - Veja estatísticas de processamento

4. **Download do ZIP**
   - Ao final, baixe o arquivo ZIP gerado
   - O ZIP contém todos os vídeos + relatório de processamento

### 📊 Estrutura do Arquivo ZIP:
```
batch_videos_YYYYMMDD_HHMMSS.zip
├── sketch_imagem1_h264.mp4
├── sketch_imagem2_h264.mp4
├── sketch_imagem3_h264.mp4
└── relatorio.txt
```

📖 **Guia Completo**: Veja `BATCH_PROCESSING_GUIDE.md` para detalhes avançados

---

## 📊 Exemplos de Configuração

### Vídeo Rápido (para imagens complexas)
```
Split Length: 20
Frame Rate: 30
Skip Rate: 15
Duração Final: 3
```

### Vídeo Detalhado (para imagens simples)
```
Split Length: 8
Frame Rate: 60
Skip Rate: 3
Duração Final: 5
```

### Vídeo Equilibrado (recomendado)
```
Split Length: 10
Frame Rate: 30
Skip Rate: 5
Duração Final: 3
```

## 📁 Estrutura do Projeto

```
automated-whiteboard/
├── app.py                    # Script principal com interface Gradio
├── requirements.txt          # Dependências Python
├── README_PT.md             # Esta documentação
├── BATCH_PROCESSING_GUIDE.md # Guia completo de processamento em lote
├── kivy/                    # Arquivos do projeto original
│   └── data/
│       └── images/
│           ├── drawing-hand.png      # Imagem da mão
│           └── hand-mask.png         # Máscara da mão
└── saved_videos/            # Vídeos gerados (criado automaticamente)
```

## 🎯 Como Funciona

1. **Processamento da Imagem**
   - Redimensiona para resolução padrão
   - Converte para escala de cinza
   - Aplica threshold adaptativo para detectar bordas

2. **Divisão em Grid**
   - Divide a imagem em pequenos quadrados (grids)
   - Identifica quais grids contêm desenho

3. **Animação Progressiva**
   - Desenha cada grid sequencialmente
   - Usa algoritmo de distância euclidiana para ordem natural
   - Adiciona imagem da mão em cada frame

4. **Geração do Vídeo**
   - Cria vídeo MP4 com os frames gerados
   - Converte para H264 (melhor compatibilidade)
   - Adiciona imagem final colorida

## 💡 Dicas

1. **Imagens Simples** (logos, desenhos, texto):
   - Use Split Length menor (5-10)
   - Skip Rate menor (3-5)
   - Resultado: Vídeo mais longo e detalhado

2. **Imagens Complexas** (fotos, ilustrações detalhadas):
   - Use Split Length maior (15-30)
   - Skip Rate maior (10-15)
   - Resultado: Vídeo mais curto e rápido

3. **Melhor Qualidade**:
   - Frame Rate: 60 FPS
   - Skip Rate: 3-5
   - Resultado: Vídeo mais suave

4. **Vídeo Mais Curto**:
   - Split Length: 20-30
   - Skip Rate: 15-20
   - Resultado: Animação mais rápida

## 🐛 Solução de Problemas

### Erro: "Nenhuma imagem carregada"
- Verifique se o arquivo é uma imagem válida (PNG, JPG, JPEG)
- Tente fazer upload novamente

### Vídeo muito longo
- Aumente o Split Length (20-30)
- Aumente o Skip Rate (10-20)

### Vídeo muito rápido
- Diminua o Split Length (5-10)
- Diminua o Skip Rate (3-5)

### Erro ao converter para H264
- O vídeo será salvo em MP4 original
- Ainda é compatível com a maioria dos players
- Para instalar suporte H264: `pip install av --upgrade`

## 🎥 Para Usar no HeyGen

1. Gere seu vídeo whiteboard
2. Faça download do vídeo
3. Acesse [HeyGen](https://heygen.com)
4. Faça upload do vídeo gerado
5. Escolha um avatar
6. O avatar irá "apresentar" seu desenho animado

## 📝 Notas Técnicas

- **Resolução**: Automaticamente ajustada para resolução padrão mais próxima
- **Formato**: MP4 (H264 quando possível)
- **Codec**: mp4v ou h264
- **Processamento**: 100% local, sem envio de dados para internet
- **Performance**: Depende da complexidade da imagem e configurações

## 🔄 Atualizações Futuras

- [x] ✅ **Suporte para múltiplas imagens (batch processing)** - IMPLEMENTADO!
- [x] ✅ **Upload em massa com download ZIP** - IMPLEMENTADO!
- [x] ✅ **Processamento paralelo otimizado** - IMPLEMENTADO!
- [ ] Opção de escolher cor de fundo (branco/preto)
- [ ] Adicionar música de fundo
- [ ] Exportar em diferentes resoluções
- [ ] Pré-visualização antes de gerar
- [ ] Suporte para mais formatos de imagem

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique se todas as dependências estão instaladas
2. Verifique se o ambiente virtual está ativado
3. Verifique se as imagens da mão existem em `kivy/data/images/`

## 📄 Licença

MIT License - Baseado no projeto original de daslearning-org

---

**Desenvolvido para criar vídeos whiteboard animados de forma simples e eficiente!** 🚀
