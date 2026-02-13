Plano técnico e de produto para refatorar seu app e alcançar paridade com Speedpaint
Resumo executivo
Seu app atual (pelo material enviado) gera vídeos de “mão desenhando” a partir de uma imagem, usando uma lógica raster baseada em threshold + divisão em grid + ordenação por componentes conectados, com exportação MP4 e compressão H.264, e uma UI em Gradio com processamento individual e em lote.

Para ficar visualmente e funcionalmente equivalente ao Speedpaint, o salto principal não é “só UI”: é trocar o núcleo de animação de revelação em blocos (grid) por um pipeline de traços vetoriais (paths) (ou pseudo‑vetoriais) com timeline determinística, controle fino de velocidade/easing, ordenação de sequência e estilos de sketch/hand. Isso é consistente com as opções expostas no próprio Speedpaint (FPS 30/60/120; duração do sketch e do fill; “fade out”; qualidade HD/SD; cor de fundo; múltiplos estilos de mão; cores e estilos de sketch; e modos de sequência como “vertical top‑to‑bottom”, “text first”, “text at the end”). 

A proposta abaixo é um plano incremental: primeiro igualar as opções e a UX (paridade “externa”), depois substituir o motor por um renderer orientado a paths e máscara (paridade “visual”), e por fim consolidar exportação/provas automatizadas (paridade “operacional”). A arquitetura recomendada é “core de renderização portátil + pipeline de exportação”, com duas trilhas possíveis (porque seu stack alvo e requisitos de performance são “não especificados”):

Trilha A (menor risco imediato): manter Python/Gradio como produto atual e refatorar o motor para paths + streaming para FFmpeg.
Trilha B (melhor para longo prazo web): criar “core” em TS que renderiza em Canvas (browser e Node via node‑canvas) e exporta com FFmpeg (server) ou WebCodecs/ffmpeg.wasm (cliente). A B facilita replicar a UX do Speedpaint e evoluir para plugins/integrações.
Nota legal/prática: os termos do Speedpaint explicitamente proíbem “reverse‑engineer ou replicar” o serviço; o plano abaixo foca em reproduzir comportamento e resultado a partir de técnicas conhecidas (CV + vetorização + animação), sem copiar implementação/propriedade intelectual e sem depender de engenharia reversa do backend. 

Diagnóstico do seu app atual e lacunas para paridade
Pelo código e documentação fornecidos, o comportamento atual é (em alto nível):

Entrada: imagem (JPG/PNG) → conversão para tons de cinza → adaptive threshold → “desenho” progressivo.
Ordem de desenho: divisão em grades (split_len) e ordenação por componentes conectados (com heurística top‑down/left‑right), mais um passo de “vizinho mais próximo” e interpolação quando há salto grande.
Efeito da mão: sobreposição de 1 sprite de mão + máscara (PNG + mask), com offset opcional (no modo colorização você já começa a considerar a ponta da caneta).
Colorização: modo “Contornos + Colorização” que busca regiões (connected components) e “pinta” pixels em lotes, renderizando frames com a mão avançando por scanline/batches.
Exportação: MP4 via OpenCV VideoWriter e tentativa de compressão H.264 via PyAV (com CRF e preset).
Produto/UI: app web via Gradio, com individual + lote e ZIP; inclui licenciamento com Stripe (OTP/sessão).
Lacunas “visuais” principais vs Speedpaint
A diferença visual mais provável (e mais perceptível) vem de dois pontos:

Granularidade do traço: grid revela blocos; Speedpaint se apresenta como “stroke‑by‑stroke” suave (com controle de “Sketch Details Level”, “Sketch Type” e cores), sugerindo um pipeline que trabalha em contornos/caminhos (paths) e não em blocos. 

Motor de timing/UX: Speedpaint expõe controles diretos de FPS (30/60/120), duração do sketch, duração do fill, fade out, qualidade, fundo, mãos múltiplas e ordens de sequência. Seu app expõe “split_len/skip_rate” (mais técnico) em vez de controles de “duração” e “sequência” orientados a usuário final. 

Lacunas “funcionais/UX” principais vs Speedpaint
O Speedpaint (no fluxo web) é essencialmente: Upload → Edit (opções) → Animate → Download, com presets de resolução e abas de opções (“Basic Options / Hand Styles / Sketching Options / Custom Sequence”). 

Seu app tem fluxo similar, mas com diferença de “linguagem do produto”: parâmetros técnicos (split/skip) vs parâmetros do Speedpaint (duração, estilo, fundo, sequência, FPS).

A seguir, a recomendação é redesenhar o produto em torno das mesmas dimensões do Speedpaint, e “traduzir” internamente para parâmetros técnicos do motor.

Decomposição do Speedpaint e implicações técnicas
Controles e limites observáveis no Speedpaint
O Speedpaint expõe (de forma direta no site) os seguintes elementos que precisam existir no seu app se a meta é paridade de produto:

Presets de resolução para formatos sociais (story, post, 16:9, A4 etc.). 
FPS: 30/60/120. 
Sketching Duration e Color Fill Duration. 
Opções de Fade Out e Background Color. 
Quality: SD/HD (e o produto vende HD 1920×1080). 
Hand Styles: 8 estilos no plano “Standard”. 
Sketch “Color”: black and white / color / shades of grey / neon blue. 
Custom Sequence: Auto; Vertical top‑to‑bottom; Text First; Text at the end. 
Limites de duração por plano (ex.: Basic 15s; Standard 50s, para sketch e color). 
Tudo acima impacta o design técnico do motor: se você aceitar 120 FPS e 50s em 1080p, o worst case é 6000 frames. Isso vira requisito de: render incremental eficiente + export streaming (sem explodir disco/memória).

Como reproduzir os elementos “de vídeo” do Speedpaint
Abaixo estão os elementos que você pediu para decompor (traço/velocidade/easing/camadas/transições/sincronização/efeitos), com abordagem recomendada e bibliotecas candidatas.

Estilo de traço e “Sketch Details Level”
Objetivo visual: linhas contínuas, sem “quadriculado”, com controle de detalhe.

Abordagem técnica (recomendada):

(Raster → paths) Vetorização do contorno com ferramenta tipo Potrace (bitmap → curvas suaves) e/ou contornos OpenCV + simplificação. Potrace é explicitamente feito para transformar bitmap em vetores suaves e escaláveis. 
Alternativa JS/TS pronta: ts-potrace, que documenta etapas (threshold → edge detection → path creation/optimization) e expõe parâmetros que mapeiam bem para “details level” (threshold, turdSize, optTolerance etc.). 
Alternativa CV: findContours + approxPolyDP (Douglas‑Peucker) para reduzir vértices conforme um “epsilon” ligado ao slider de detalhe. 
Bibliotecas candidatas (render):

Canvas 2D puro (browser + Node) e/ou Paper.js (vetor sobre Canvas) para manipular curvas e path ops. 
Two.js se você quiser a mesma API renderizar em canvas/svg/webgl (útil pra experimentar). 
PixiJS/WebGL quando a carga de render ficar alta (muitos strokes + 120fps), aproveitando GPU. 
Velocidade, easing e “smoothness”
Objetivo visual: o traço acelera/desacelera de forma natural; o movimento da mão acompanha sem travamentos; não depender de skip_rate (frames pulados) como parâmetro primário.

Abordagem técnica:

Trocar “skip_rate” por uma timeline baseada em tempo contínuo: dado t (segundos desde início), calcular progresso p ∈ [0,1] do sketch usando funções de easing (ex.: cubic in/out).
Em browser, o Web Audio API é referência para timing preciso (se houver áudio no futuro) e o temporizador de áudio (AudioContext.currentTime) é adequado para sincronização fina. 
Para easing/timeline no front, você pode usar:
Web Animations API (spec e MDN) como modelo de timeline/clock. 
Timeline própria (recomendado para export determinístico, porque você quer “mesmo frame = mesmo pixel” em qualquer execução).
Camadas, composição e efeitos (hand/shadow/overlays)
Objetivo visual: stroke em uma camada; fill em outra; mão acima; opcional sombra suave.

Abordagem técnica (compositing):

Pipeline de render em camadas:
background (cor sólida)
fill layer (revelação de cor)
stroke layer (desenho)
hand layer (sprite + sombra)
post (fade out, leve vinheta se existir)
No Canvas2D você consegue sombra via shadowBlur/shadowOffset, mas o controle fino e performance pode exigir WebGL em casos pesados. (Canvas 2D como base: APIs MDN). 
Transições (fade out) e finalização
Objetivo visual: opção de Fade Out no final.

Abordagem técnica:

Implementar ao final um “clip” de transição com curva de opacidade sobre camadas, por fadeDuration. Esse controle existe como opção no Speedpaint. 
Sincronização com áudio
Paridade: não há evidência de controles de áudio no painel de opções do Speedpaint (o que aparece são opções de sketch/fill/fps/estilo/ordem). 

Recomendação: tratar áudio como extensão (você pediu, então a arquitetura precisa suportar), usando:

Web Audio API para preview e agendamento (timing de alta precisão). 
Export: mux de áudio e vídeo via FFmpeg (server). FFmpeg é o caminho padrão para transcodificar e multiplexar. 
Efeito de mão/brush e variação de espessura
Objetivo visual: cursor “mão” crível; opcional variação sutil na espessura do traço/pressão.

Abordagens técnicas:

Hand sprite follow path: amostrar pontos do path conforme avanço e posicionar/rotacionar sprite.
Para amostrar path em SVG, getTotalLength() e getPointAtLength() são APIs padrão (úteis para prototipar sampling). 
Espessura variável: usar algoritmo tipo perfect-freehand para gerar um polígono “outline” do traço, com pressão simulada por distância/velocidade. Isso é útil para dar aparência mais orgânica. 
Preenchimentos (color fill) e estilos de cor
Objetivo visual: o fill entra por tempo controlado (“Color Fill Duration”) e suporta paletas (B/W, Color, Grey, Neon Blue). 

Abordagens possíveis para fill (em ordem de custo/realismo):

Crossfade/Reveal simples: desenha o sketch; depois faz blend do original por máscara global (rápido, mas menos “mão pintando”).
Regiões por contorno: segmenta regiões limitadas por linhas (connected components) e “pinta” região a região (seu app já faz algo desse tipo).
Pincel por path: cria trajetórias de pintura dentro das regiões (scanline/zigzag) e move a mão/cursor (mais realista).
Morphing e transformações de shape
O Speedpaint não expõe morphing como opção; ainda assim, se você quer evoluir (ou caso seu app aceite SVGs), morphing pode ser útil para transições. Uma opção prática em SVG é flubber para interpolar paths arbitrários. 

Arquitetura proposta com diagrama e comparativos por componente
A arquitetura abaixo foi desenhada para: (i) reproduzir as opções do Speedpaint, (ii) ter render determinístico para export, e (iii) permitir evolução (áudio, cenas, templates) sem reescrever tudo.

mermaid
Copiar
flowchart TB
  subgraph UI[Frontend / UI]
    U1[Uploader + Presets de resolução]
    U2[Editor de opções\n(fps, durações, estilo, fundo, sequência)]
    U3[Preview Sketch (rápido)]
    U4[Player Preview\n(timeline determinística)]
  end

  subgraph CORE[Core de Renderização]
    C1[Asset Pipeline\n(raster->paths, cache)]
    C2[Timeline/Scheduler\n(clock + easing)]
    C3[Renderer\n(strokes, fill, hand, post)]
    C4[Audio Sync\n(opcional)]
  end

  subgraph BACK[Backend/Worker]
    B1[Job Queue + Status]
    B2[Frame Renderer\n(headless ou node-canvas)]
    B3[Encoder\nFFmpeg]
    B4[Storage/Cache\n(projetos, frames, vídeo)]
  end

  U1 --> C1
  U2 --> C2
  U3 --> C1
  U4 --> C3
  C1 --> C3
  C2 --> C3
  C4 --> C3

  U4 -->|Exportar| B1
  B1 --> B2
  B2 --> B3
  B3 --> B4
  B4 -->|Download| UI
Fontes que embasam decisões-chave: (a) Speedpaint expõe opções/funil de UX e múltiplos estilos/durações/ordem. 
 (b) Para render web: Canvas/WebGL/OffscreenCanvas e otimizações. 
 (c) Export: FFmpeg/H.264. 

Renderer (preview + export)
Opção	Prós	Contras	Complexidade	Performance	Custo
Canvas 2D puro (browser)	Simples; fácil de debugar; suficiente para MVP	Pode sofrer com muitos paths/120fps	Baixa	Média	Baixo
Canvas 2D “portável” (browser + Node via node-canvas)	Mesmo código preview/export; determinístico	node-canvas tem deps nativas (Cairo)	Média	Média/Alta	Médio
SVG (DOM) + stroke-dashoffset	Animação de traço “nativa” e elegante	Export frame a frame vira difícil sem headless	Média	Média	Médio
WebGL via PixiJS	GPU, aguenta muita coisa e 120fps	Pipeline mais complexo; texturas/antialias	Alta	Alta	Médio

Fontes: PixiJS (engine WebGL), WebGL como API acelerada por hardware, e node-canvas como implementação Canvas API em Node. 

Asset pipeline (raster→paths, simplificação, cache)
Opção	Prós	Contras	Complexidade	Performance	Custo
Potrace CLI (server)	Qualidade alta de vetorização; maduro	Dep de binário; pipeline de deploy	Média	Alta	Baixo
ts-potrace (Node)	Parametrizável; integra bem com web	Qualidade depende da entrada/params	Média	Média	Baixo
OpenCV contours + approxPolyDP	Controle fino; bom para “detalhe level”	Requer mais engenharia pra suavizar	Alta	Média	Baixo
ML (modelo próprio)	Potencial para “humanização” semelhante ao claim do Speedpaint	Alto esforço; dataset/treino	Muito alta	Variável	Alto

Fontes: Potrace (bitmap→vetor suave) e ts-potrace (raster→SVG com opções), OpenCV findContours e approxPolyDP. 

Timeline/scheduler (determinismo e easing)
Opção	Prós	Contras	Complexidade	Performance	Custo
Timeline própria (recomendado)	Export determinístico; fácil “seek”; previsível	Você implementa clocks/easing	Média	Alta	Baixo
Web Animations API	Modelo padrão; clocks/timelines definidos	Export server-side mais difícil	Média	Alta	Baixo
Biblioteca de timeline (GSAP etc.)	Produtividade alta	Dependência e menos controle determinístico	Média	Alta	Médio

Fontes: especificação Web Animations define “timelines” e “currentTime”; MDN descreve Web Animations API. 

Audio sync (opcional, mas arquitetado)
Opção	Prós	Contras	Complexidade	Performance	Custo
Web Audio API (preview)	Timing preciso; agendamento por currentTime	Export em tempo real não é ideal	Média	Alta	Baixo
FFmpeg (mux no export)	Padrão para juntar áudio+vídeo	Precisa pipeline server	Média	Alta	Médio
Sem áudio (paridade estrita)	Simplicidade	Usuários vão editar fora	Baixa	Alta	Baixo

Fontes: Web Audio API (timing e currentTime) e FFmpeg como conversor/transcoder universal. 

Exportador e serving/caching
Opção	Prós	Contras	Complexidade	Performance	Custo
Server FFmpeg (recomendado)	Melhor qualidade/controle; MP4/H.264 confiável	Infra/worker	Média	Alta	Médio
ffmpeg.wasm (browser)	Sem backend; privacidade	Peso alto; lento em 1080p/120fps	Média	Baixa/Média	Baixo
WebCodecs (browser)	Acesso low-level e eficiente a frames/encoders	Montar MP4 container não é trivial	Alta	Alta	Baixo/Médio
MediaRecorder (browser)	Simples para WebM; bom para preview capture	Controle limitado (codec/container)	Baixa	Alta	Baixo

Fontes: FFmpeg H.264 encode wiki; ffmpeg.wasm (FFmpeg em WebAssembly); WebCodecs API; MediaRecorder; OffscreenCanvas como estratégia de performance. 

Especificações de implementação acionáveis
A meta aqui é você conseguir “dar para o Windsurf” uma estrutura concreta para gerar/alterar código.

Modelo de projeto (formato interno)
Recomendação: JSON versionado com separação entre entrada, asset derivado, timeline e export.

json
Copiar
{
  "version": "1.0.0",
  "source": {
    "type": "raster",
    "originalFileName": "input.png",
    "width": 1920,
    "height": 1080,
    "sha256": "..."
  },
  "settings": {
    "preset": "presentation-16-9",
    "fps": 60,
    "sketchDurationSec": 12.0,
    "fillDurationSec": 6.0,
    "fadeOutSec": 0.8,
    "quality": "HD",
    "backgroundColor": "#FFFFFF",
    "handStyleId": "hand-03",
    "sketchColorMode": "bw",
    "sketchDetailLevel": 0.65,
    "sequenceMode": "auto",
    "seed": 12345
  },
  "derived": {
    "vector": {
      "engine": "potrace",
      "paths": [
        { "id": "p1", "d": "M ... Z", "length": 523.2, "bbox": [x,y,w,h], "tags": ["text?"] }
      ],
      "pathOrder": ["p7","p2","p9"]
    },
    "fill": {
      "engine": "regions",
      "regions": [
        { "id": "r1", "maskRef": "mask/r1.png", "bbox": [x,y,w,h] }
      ],
      "regionOrder": ["r3","r1","r2"]
    }
  },
  "timeline": {
    "clips": [
      { "type": "sketch", "t0": 0.0, "t1": 12.0 },
      { "type": "fill", "t0": 12.0, "t1": 18.0 },
      { "type": "fade", "t0": 18.0, "t1": 18.8 }
    ]
  },
  "export": {
    "container": "mp4",
    "videoCodec": "h264",
    "audioCodec": "aac",
    "bitrateKbps": 8000
  }
}
Por que isso é importante para paridade: o Speedpaint expõe exatamente esse tipo de configurações (fps, durações, fade, fundo, estilos, sequência, presets). 

API interna (contratos) — interfaces que o Windsurf deve implementar
Mesmo que “backend sim/não” esteja não especificado, a sua base fica mais limpa se os contratos existirem:

1) Vetorização

ts
Copiar
type VectorizeInput = {
  imageRGBA: Uint8ClampedArray; width: number; height: number;
  detailLevel: number; // 0..1
  mode: "bw" | "grey" | "color" | "neon";
};

type PathItem = { id: string; d: string; length: number; bbox: [number, number, number, number]; };

type VectorizeOutput = {
  paths: PathItem[];
  suggestedOrder: string[];
  previewSketchRGBA: Uint8ClampedArray;
};

interface Vectorizer {
  vectorize(input: VectorizeInput): Promise<VectorizeOutput>;
}
Base técnica: ts-potrace fornece geração de SVG e parâmetros (threshold, optTolerance etc.) que mapeiam bem para “detail”. 

2) Timeline

ts
Copiar
type Clip = { type: "sketch" | "fill" | "fade"; t0: number; t1: number; easing: string; };
type Timeline = { fps: number; duration: number; clips: Clip[] };

function timeToFrame(t: number, fps: number): number {
  return Math.floor(t * fps + 1e-6);
}
Referência conceitual (não implementação): Web Animations API define “timelines” e currentTime como base de clock. 

3) Renderer (frame determinístico)

ts
Copiar
type RenderFrameInput = {
  timeline: Timeline;
  vector: VectorizeOutput;
  settings: any;
  t: number; // seconds
};

interface Renderer {
  renderFrame(canvas: CanvasLike, input: RenderFrameInput): void;
}
Canvas 2D como base (API). 

Algoritmo de animação de traço (substituir grid por paths)
A ideia é: em cada frame, você desenha:

todos os paths já “completos”
o path corrente “parcial”
desenha a mão na ponta do path corrente
Há duas técnicas principais:

Técnica A: “partial polyline draw” (Canvas)

Converta o d do SVG path para uma polyline amostrada (pontos ao longo do comprimento).
Para um progresso p, desenhe apenas os primeiros k pontos.
Técnica B: “dashoffset” (SVG mental model) No SVG, stroke-dasharray e stroke-dashoffset são usados para revelar linhas. Esse conceito é útil para pensar e parametrizar. 

Para sampling/posição da ponta, getTotalLength() e getPointAtLength() ajudam a pegar “ponto em distância”. 

Pseudocódigo (Canvas, técnica A):

ts
Copiar
function renderSketch(ctx, paths, order, t, settings) {
  const { sketchDurationSec, lineWidth, strokeStyle, easing } = settings;

  // progresso global 0..1
  const pGlobal = clamp01(t / sketchDurationSec);
  const eased = ease(easing, pGlobal);

  // total de "comprimento" para distribuir tempo
  const totalLen = sum(order.map(id => paths[id].length));

  // quanto do comprimento já deve ter sido revelado
  const revealedLen = eased * totalLen;

  let acc = 0;
  for (const id of order) {
    const path = paths[id];
    const next = acc + path.length;

    if (revealedLen >= next) {
      drawFullPath(ctx, path); // completo
    } else if (revealedLen > acc) {
      const partialLen = revealedLen - acc;
      drawPartialPathByLength(ctx, path, partialLen); // parcial
      const tip = getPointOnPath(path, partialLen);   // para a mão
      drawHand(ctx, tip, path, settings);
      break;
    } else {
      break;
    }

    acc = next;
  }
}
Onde getPointOnPath pode ser implementado via:

(SVG) getPointAtLength em um <path> temporário (browser), ou
(Core) amostragem pré-computada de polyline.
Simular mão desenhando
O Speedpaint vende “Hand Styles” (8 no plano Standard). 

Para paridade:

Padronize o asset: handStyleId aponta para um pacote (sprite + mask + metadados).
Metadados críticos por mão:
tipOffsetPx: {x,y} (ponta da caneta)
baseRotationDeg (se o sprite já está inclinado)
shadow: {blur, dx, dy, alpha}
Pseudocódigo:

ts
Copiar
function drawHand(ctx, tip, path, settings) {
  const hand = handRegistry[settings.handStyleId];
  const angle = estimateTangentAngle(path, tip.s); // s = comprimento atual
  ctx.save();
  ctx.translate(tip.x, tip.y);
  ctx.rotate(angle + hand.baseRotationRad);

  // sombra (opcional)
  ctx.globalAlpha = hand.shadow.alpha;
  ctx.filter = `blur(${hand.shadow.blur}px)`;
  ctx.drawImage(hand.shadowImage, -hand.tipOffset.x + hand.shadow.dx, -hand.tipOffset.y + hand.shadow.dy);

  // mão
  ctx.globalAlpha = 1;
  ctx.filter = "none";
  ctx.drawImage(hand.image, -hand.tipOffset.x, -hand.tipOffset.y);
  ctx.restore();
}
Anti-aliasing, subpixel e performance
Para chegar no “acabamento” do Speedpaint em HD e 60/120fps, priorize:

Render em DPR: canvas com width = w*dpr, height = h*dpr, e escala ctx.scale(dpr, dpr) (cresce nitidez).
Pré-render de elementos estáticos em canvas “offscreen” (background + fills completos), e só compor por frame (MDN recomenda pre-render em offscreen). 
OffscreenCanvas em worker para evitar travar a UI em preview (quando estiver em web). 
Cache de paths amostrados: precompute polylinePoints[] para cada path e não recompute por frame.
Pipeline de exportação e recomendações de codecs
O pipeline recomendado (alinhado ao seu pedido) é:

Render frame (Canvas/WebGL) →
Gerar frames (RGBA) em stream →
FFmpeg codifica em MP4/H.264 (ou WebM) →
Salvar e servir arquivo.
MP4/H.264 (recomendação padrão)
Por compatibilidade e equivalência com ferramentas “HD export”, H.264 é o “workhorse” comum e amplamente suportado por hardware, e o guia do FFmpeg para H.264 (x264) recomenda uso de CRF + preset como fluxo típico. 

Exemplo de comando (server/worker):

bash
Copiar
ffmpeg -y \
  -f rawvideo -pix_fmt rgba -s 1920x1080 -r 60 -i pipe:0 \
  -c:v libx264 -pix_fmt yuv420p -preset medium -crf 20 \
  -movflags +faststart \
  output.mp4
Pontos-chave:

-pix_fmt yuv420p maximiza compatibilidade.
-crf controla qualidade (menor = melhor/maior arquivo).
-preset troca tempo de encode por eficiência. 
WebM (VP9/AV1) como opção
Se você quiser “export direto do browser”, WebM pode ser mais simples com MediaRecorder, mas o controle de qualidade/codec varia por navegador. MediaRecorder é oficialmente a API web para “record” de streams. 

Export dentro do browser
ffmpeg.wasm: é literalmente FFmpeg em WebAssembly e permite converter/encodar no browser, mas com custo alto em CPU e tempo em 1080p/120fps. 
WebCodecs: dá acesso low-level a frames/encoders (ex.: VideoEncoder), excelente para performance, mas você terá que tratar container/mux (MP4) corretamente. 
Minha recomendação para paridade com Speedpaint: server export com FFmpeg (mais previsível), e preview no cliente.

Qualidade, métricas, testes e plano incremental de refatoração
Checklist visual de paridade (manual)
Use o próprio painel do Speedpaint como “spec”:

Presets de resolução e enquadramento. 
FPS 30/60/120 realmente muda fluidez e duração (não só “duplicar frames”). 
Duração de sketch e fill respeita tempo total. 
Fundo configurável e fade out. 
Estilos de sketch (BW/Color/Grey/Neon Blue) batem com expectativa. 
Mãos: pelo menos 8 estilos (se você quer equivalência ao plano Standard). 
Sequência: Auto / vertical / text first / text last produz comportamento diferente de fato. 
Métricas objetivas de “igualdade visual”
Para reduzir subjetividade e permitir testes automatizados:

SSIM (Structural Similarity Index): é uma métrica perceptual usada para comparar semelhança estrutural; é mais alinhada à percepção humana do que MSE e está disponível em bibliotecas como scikit-image. 
Perceptual hash (pHash): útil para detectar “muito parecido” com tolerância a pequenas diferenças; pHash e bibliotecas como ImageHash documentam esse uso. 
PSNR/SSIM em vídeo com OpenCV: existe tutorial mostrando comparação frame‑a‑frame para vídeos. 
Recomendação prática: em testes A/B, exporte N frames da sua animação e N frames do Speedpaint (mesma imagem/config) e compare:

SSIM médio e mínimo por frame
distribuição de erro ao longo do tempo (se o começo “parece” mas o fim diverge)
Roteiro de testes automatizados
1) Golden tests de frame

Fixar input e seed.
Gerar frames em timestamps específicos (ex.: 0%, 25%, 50%, 75%, 100%).
Comparar contra “golden images” (baseline) com SSIM mínimo aceitável.
2) Visual regression (UI web) Se você for pela trilha web, Playwright oferece toHaveScreenshot() para comparação visual automatizada. 

3) Performance

Budget: tempo de export por segundo de vídeo (ex.: 1s vídeo em ≤ X s CPU) e memória máxima.
No browser, use OffscreenCanvas + pre-render para evitar jank (MDN). 
Plano incremental em milestones com tarefas “para o Windsurf”
Abaixo está um plano pensado para entregar primeiro a paridade “visível”, depois a paridade “real” do motor.

2026-02-15
2026-02-22
2026-03-01
2026-03-08
2026-03-15
2026-03-22
2026-03-29
2026-04-05
2026-04-12
2026-04-19
Normalizar opções (fps/durações/fade/fundo/presets)
Preview Sketch + presets de resolução
Raster->paths (ts-potrace/Potrace/OpenCV)
Renderer de stroke por comprimento + easing
Hand follow path + 8 hand styles
Sequência (auto/vertical/text first/text last)
Fill simples (crossfade/mask global)
Export streaming FFmpeg (H.264) + presets
Fill por regiões (componentes / scanline “pintando”)
Métricas SSIM/pHash + golden tests
Paridade de UX (rápido)
Motor de paths (paridade visual)
Color fill (paridade funcional)
Export e QA
Roadmap incremental para paridade com Speedpaint


Exibir código
Milestone “Paridade de UX” (MVP de produto)

Criar camada UISettings com campos idênticos aos do Speedpaint (fps, sketchDuration, fillDuration, fadeOut, quality, backgroundColor, handStyle, sketchColorMode, detailLevel, sequenceMode, preset). 
Implementar presets de resolução listados no Speedpaint (pelo menos: 1080×1920, 1920×1080, 500×500). 
Traduzir settings → motor atual (grid) apenas para liberar UI, mesmo que o visual ainda não bata (isso ajuda validação rápida com usuários).
Milestone “Motor de paths” (onde a paridade visual nasce)

Implementar Vectorizer:
Opção TS: ts-potrace para gerar SVG e extrair paths; controle de detalhe via options (threshold, optTolerance etc.). 
Opção Python: Potrace CLI como etapa (bitmap→SVG); ou OpenCV findContours + approxPolyDP. 
Implementar PathSampler (amostragem por distância).
Implementar StrokeRenderer (desenho parcial por comprimento + easing).
Implementar HandRenderer com metadados e pacote de 8 mãos (paridade com plano Standard). 
Milestone “Sequência”

sequenceMode="vertical": ordenar por bbox.y.
sequenceMode="auto": TSP heurístico (vizinho mais próximo) sobre centroid/bbox, com penalidade para saltos.
text first/text last: duas opções:
heurística (bounding boxes finas, alta densidade de contornos → provável texto)
OCR (mais pesado).
O Speedpaint expõe esses modos; a equivalência precisa existir pelo menos em comportamento. 
Milestone “Fill”

Começar com mask global (rápido) para respeitar “Color Fill Duration”. 
Evoluir para fill por regiões (componentes conectados dentro de contornos), aproximando o efeito “pintando”. (Você já tem base em Python com connected components.)
Milestone “Export e QA”

Export streaming para FFmpeg (H.264) seguindo recomendações de CRF/preset/pix_fmt. 
Testes:
SSIM por frame e média. 
pHash para detectar regressões grosseiras. 
Se web: Playwright screenshot regression. 
Recursos prioritários e entregáveis para o Windsurf
Referências prioritárias
Speedpaint (opções e limites): painel de conversão e página de preços/FAQ. 
Canvas API e otimização: MDN pt‑BR (Canvas 2D e dicas de performance). 
SVG stroke reveal: stroke-dasharray / stroke-dashoffset. 
Sampling de path: getTotalLength e getPointAtLength. 
Vetorização:
Potrace (conceito e docs) 
ts-potrace (uso e parâmetros) 
OpenCV contours + approxPolyDP 
Render performance:
WebGL e PixiJS (quando precisar GPU) 
OffscreenCanvas 
Export:
FFmpeg H.264 encode guia 
ffmpeg.wasm (se quiser export client-side) 
WebCodecs (VideoEncoder/WebCodecs overview) 
Métricas:
SSIM paper + docs scikit-image 
ImageHash/pHash 
Entregáveis esperados (arquivos e funcionalidades) — lista para o Windsurf gerar
A lista abaixo é propositalmente “concreta” (nomes de arquivos e responsabilidades) para você colar no Windsurf como checklist de geração/refatoração.

Estrutura mínima (independente da trilha A/B):

core/settings/schema.ts
schema de settings + validação + defaults (espelhando Speedpaint). 
core/vectorize/vectorizer.ts
interface Vectorizer + implementação TsPotraceVectorizer (trilha B) ou “wrapper” (trilha A). 
core/vectorize/pathSampler.ts
amostragem por distância, cache de polylines; suporte a “detailLevel”. 
core/timeline/timeline.ts
clips (sketch/fill/fade), easing, seek, timeToFrame. 
core/render/canvasTypes.ts
abstração CanvasLike (browser + node-canvas). 
core/render/strokeRenderer.ts
render parcial por comprimento + estilos (bw/grey/color/neon). 
core/render/handRenderer.ts
pacote de assets de mãos + offsets + sombra. 
core/render/fillRenderer.ts
fase 1: mask global; fase 2: fill por regiões/pincel. 
core/export/ffmpegPipe.ts (server)
pipe de frames RGBA → FFmpeg (libx264, yuv420p, CRF/preset). 
tests/visual/goldenFrames.spec.ts
gera frames fixos e compara SSIM/pHash. 
Se você ficar na trilha A (Python/Gradio):

Refatorar generate_sketch_video em módulos:
engine/vectorize.py (Potrace/OpenCV)
engine/timeline.py
engine/render_paths.py
engine/export_ffmpeg.py (idealmente migrar de PyAV para FFmpeg direto, ou manter PyAV mas com stream)
Adaptar UI para settings “Speedpaint-like”.
Manter batch e ZIP (já existe) e só trocar o “motor”.
Se você for de trilha B (web + worker):

apps/web/
componentes: uploader, presets, painéis (Basic/Hand/Sketch/Sequence), preview player
apps/api/ + apps/worker/
endpoints: /projects, /projects/:id/preview, /export, /jobs/:id, /download/:id
worker: render frames e FFmpeg encode.
Esses entregáveis cobrem as dimensões que o Speedpaint expõe publicamente (opções, estilos, sequência, resoluções e HD export), e substituem a limitação estrutural do motor baseado em grid por um motor baseado em paths, que é o passo técnico necessário para “paridade visual” real. 