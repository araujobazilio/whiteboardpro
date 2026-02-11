import gradio as gr
import cv2
import numpy as np
import os
import time
import math
import tempfile
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurações
SAVE_PATH = "generated_videos"
HAND_PATH = "kivy/data/images/drawing-hand.png"
HAND_MASK_PATH = "kivy/data/images/hand-mask.png"

# Garantir diretórios
os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# FUNÇÕES AUXILIARES - HUMAN SPEEDPAINT
# ============================================================

def easeInOutQuad(t):
    """Curva de easing natural: início lento, pico rápido, parada suave."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def resize_smart(img, max_w=1920, max_h=1080):
    """Resize mantendo proporção, limitando ao máximo especificado"""
    h, w = img.shape[:2]
    
    if w <= max_w and h <= max_h:
        return img
    
    aspect_ratio = w / h
    
    if w / max_w > h / max_h:
        new_w = max_w
        new_h = int(max_w / aspect_ratio)
    else:
        new_h = max_h
        new_w = int(max_h * aspect_ratio)
    
    return cv2.resize(img, (new_w, new_h))

def get_neighbors(pixel, all_pixels, max_dist=5):
    """Encontra pixels vizinhos dentro da distância máxima"""
    x, y = pixel
    neighbors = []
    
    for px, py in all_pixels:
        if (px, py) == (x, y):
            continue
            
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if 2 <= dist <= max_dist:
            neighbors.append((px, py))
    
    return neighbors

def group_into_strokes(pixels, max_distance=5):
    """Agrupa pixels em strokes conectados usando DFS."""
    if not pixels:
        return []
    
    strokes = []
    visited = set()
    pixel_list = [(int(px), int(py)) for px, py in pixels]
    
    for pixel in pixel_list:
        if pixel in visited:
            continue
        
        stroke = [pixel]
        visited.add(pixel)
        stack = [pixel]
        
        while stack:
            current = stack.pop()
            neighbors = get_neighbors(current, pixel_list, max_distance)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    stroke.append(neighbor)
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        if len(stroke) > 3:
            stroke = order_stroke_pixels(stroke)
            strokes.append(stroke)
    
    return strokes

def order_stroke_pixels(pixels):
    """Ordena pixels de um stroke para criar um caminho contínuo"""
    if len(pixels) <= 2:
        return pixels
    
    ordered = [pixels[0]]
    remaining = pixels[1:]
    
    while remaining:
        current = ordered[-1]
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pixel in enumerate(remaining):
            dist = ((current[0] - pixel[0]) ** 2 + (current[1] - pixel[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        ordered.append(remaining.pop(closest_idx))
    
    return ordered

def preprocess_hand_image(hand_path, hand_mask_path):
    """Processa imagem da mão e máscara"""
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, 0)
    
    if hand is None or hand_mask is None:
        raise FileNotFoundError("Imagem da mão ou máscara não encontrada")
    
    hand_mask_inv = cv2.bitwise_not(hand_mask)
    hand_ht, hand_wd = hand.shape[:2]
    
    return hand, hand_mask, hand_mask_inv, hand_ht, hand_wd

def draw_hand_on_canvas(canvas, hand_img, x, y, hand_mask_inv, hand_ht, hand_wd, canvas_ht, canvas_wd):
    """Versão simplificada para posicionar a mão no canvas"""
    start_x = max(0, x - hand_wd // 2)
    start_y = max(0, y - hand_ht // 2)
    end_x = min(canvas_wd, start_x + hand_wd)
    end_y = min(canvas_ht, start_y + hand_ht)
    
    hand_start_x = max(0, hand_wd // 2 - x)
    hand_start_y = max(0, hand_ht // 2 - y)
    hand_end_x = hand_start_x + (end_x - start_x)
    hand_end_y = hand_start_y + (end_y - start_y)
    
    if hand_end_x <= hand_start_x or hand_end_y <= hand_start_y:
        return canvas
    
    hand_roi = hand_img[hand_start_y:hand_end_y, hand_start_x:hand_end_x]
    mask_roi = hand_mask_inv[hand_start_y:hand_end_y, hand_start_x:hand_end_x]
    canvas_roi = canvas[start_y:end_y, start_x:end_x]
    
    if canvas_roi.size == 0 or hand_roi.size == 0:
        return canvas
    
    canvas_masked = cv2.bitwise_and(canvas_roi, canvas_roi, mask=255 - mask_roi)
    hand_masked = cv2.bitwise_and(hand_roi, hand_roi, mask=mask_roi)
    combined = cv2.add(canvas_masked, hand_masked)
    
    canvas[start_y:end_y, start_x:end_x] = combined
    
    return canvas

# ============================================================
# FUNÇÕES PRINCIPAIS - HUMAN SPEEDPAINT
# ============================================================

def generate_sketch_video(
    image_path,
    split_len,
    frame_rate,
    skip_rate,
    end_duration,
    draw_mode="Apenas Contornos",
    progress=gr.Progress()
):
    """
    HUMAN SPEEDPAINT: Objetos individuais + easing natural + stroke grouping
    """
    try:
        progress(0, desc="🧠 Analisando estrutura da imagem...")
        
        # 1. Carregar e pré-processar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Erro ao carregar imagem"
        
        img = resize_smart(img, max_w=1920, max_h=1080)
        img_ht, img_wd = img.shape[:2]
        
        progress(0.05, desc=f"🔧 Imagem redimensionada para {img_wd}x{img_ht}")
        
        # Processar com Canny (melhor para contornos)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Carregar mão
        progress(0.1, desc="✋ Carregando imagem da mão...")
        hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
            HAND_PATH, HAND_MASK_PATH
        )
        
        # 2. Connected Components → OBJETOS individuais
        progress(0.15, desc="🎯 Identificando objetos individuais...")
        
        num_labels, labels = cv2.connectedComponents(edges)
        
        object_list = []
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            size = np.sum(component_mask)
            if size > 50:
                ys, xs = np.where(component_mask)
                center = (int(np.mean(xs)), int(np.mean(ys)))
                object_list.append({
                    'id': label_id,
                    'mask': component_mask,
                    'center': center,
                    'size': size,
                    'pixels': list(zip(xs, ys))
                })
        
        object_list.sort(key=lambda obj: obj['size'] * (1 + obj['center'][1]/img_ht), reverse=True)
        
        total_objects = len(object_list)
        print(f"🎯 {total_objects} objetos identificados")
        
        # 3. Preparar vídeo
        now = datetime.now()
        video_name = f"human_sketch_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(SAVE_PATH, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_object = cv2.VideoWriter(video_path, fourcc, frame_rate, (img_wd, img_ht))
        
        # Canvas branco
        canvas = np.ones((img_ht, img_wd, 3), dtype=np.uint8) * 255
        
        # 4. ANIMAÇÃO com EASING
        progress(0.3, desc=f"🎨 Animando {total_objects} objetos...")
        
        for obj_idx, obj in enumerate(object_list):
            progress(0.3 + (0.5 * obj_idx / total_objects), 
                    desc=f"🎨 Desenhando objeto {obj_idx+1}/{total_objects}")
            
            obj_pixels = obj['pixels']
            strokes = group_into_strokes(obj_pixels, max_distance=5)
            
            for stroke in strokes:
                stroke_len = len(stroke)
                if stroke_len < 3:
                    continue
                
                frames_for_stroke = max(5, stroke_len // 10)
                
                for frame_in_stroke in range(frames_for_stroke):
                    t = frame_in_stroke / frames_for_stroke
                    eased_t = easeInOutQuad(t)
                    drawn_up_to = int(stroke_len * eased_t)
                    
                    # Desenhar pixels até posição atual
                    stroke_canvas = canvas.copy()
                    for i in range(drawn_up_to):
                        x, y = stroke[i]
                        cv2.circle(stroke_canvas, (x, y), 1, (0, 0, 0), -1)
                    
                    # Posicionar mão
                    if drawn_up_to > 0:
                        current_pos = stroke[min(drawn_up_to-1, stroke_len-1)]
                        stroke_canvas = draw_hand_on_canvas(
                            stroke_canvas, hand, current_pos[0], current_pos[1],
                            hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd
                        )
                    
                    video_object.write(stroke_canvas)
            
            # Pequena pausa entre objetos
            for _ in range(3):
                video_object.write(canvas)
        
        # 5. Imagem final
        progress(0.9, desc="🌈 Adicionando imagem final...")
        for _ in range(int(frame_rate * end_duration)):
            video_object.write(img)
        
        video_object.release()
        
        progress(1.0, desc="✅ Vídeo humano pronto!")
        return video_path, f"Vídeo com movimento NATURAL humano gerado com sucesso!"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erro no processamento: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg

def generate_sketch_video_single(
    image_path,
    split_len,
    frame_rate,
    skip_rate,
    end_duration,
    draw_mode="Apenas Contornos"
):
    """
    VERSÃO HUMAN SPEEDPAINT para batch processing
    """
    try:
        # 1. Carregar e pré-processar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Erro ao carregar imagem: {image_path}"
        
        img = resize_smart(img, max_w=1920, max_h=1080)
        img_ht, img_wd = img.shape[:2]
        
        # Processar com Canny
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Carregar mão
        hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
            HAND_PATH, HAND_MASK_PATH
        )
        
        # 2. Connected Components
        num_labels, labels = cv2.connectedComponents(edges)
        
        object_list = []
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            size = np.sum(component_mask)
            if size > 50:
                ys, xs = np.where(component_mask)
                center = (int(np.mean(xs)), int(np.mean(ys)))
                object_list.append({
                    'id': label_id,
                    'mask': component_mask,
                    'center': center,
                    'size': size,
                    'pixels': list(zip(xs, ys))
                })
        
        object_list.sort(key=lambda obj: obj['size'] * (1 + obj['center'][1]/img_ht), reverse=True)
        
        # 3. Preparar vídeo
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        video_name = f"human_sketch_{base_name}_{int(time.time())}.mp4"
        video_path = os.path.join(tempfile.gettempdir(), video_name)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_object = cv2.VideoWriter(video_path, fourcc, frame_rate, (img_wd, img_ht))
        
        # Canvas branco
        canvas = np.ones((img_ht, img_wd, 3), dtype=np.uint8) * 255
        
        # 4. ANIMAÇÃO com EASING
        for obj in object_list:
            obj_pixels = obj['pixels']
            strokes = group_into_strokes(obj_pixels, max_distance=5)
            
            for stroke in strokes:
                stroke_len = len(stroke)
                if stroke_len < 3:
                    continue
                
                frames_for_stroke = max(4, stroke_len // 12)
                
                for frame_in_stroke in range(frames_for_stroke):
                    t = frame_in_stroke / frames_for_stroke
                    eased_t = easeInOutQuad(t)
                    drawn_up_to = int(stroke_len * eased_t)
                    
                    stroke_canvas = canvas.copy()
                    for i in range(drawn_up_to):
                        x, y = stroke[i]
                        cv2.circle(stroke_canvas, (x, y), 1, (0, 0, 0), -1)
                    
                    if drawn_up_to > 0:
                        current_pos = stroke[min(drawn_up_to-1, stroke_len-1)]
                        stroke_canvas = draw_hand_on_canvas(
                            stroke_canvas, hand, current_pos[0], current_pos[1],
                            hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd
                        )
                    
                    video_object.write(stroke_canvas)
            
            for _ in range(2):
                video_object.write(canvas)
        
        # 5. Imagem final
        for _ in range(int(frame_rate * end_duration)):
            video_object.write(img)
        
        video_object.release()
        return video_path, None
        
    except Exception as e:
        return None, f"Erro: {str(e)}"

# ============================================================
# INTERFACE GRADIO
# ============================================================

def create_commercial_interface():
    """Interface principal com Human Speedpaint"""
    with gr.Blocks(title="Whiteboard Animation Pro - Human Speedpaint", theme=gr.themes.Soft()) as app:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🎨 Whiteboard Animation Pro</h1>
            <p><strong>HUMAN SPEEDPAINT</strong> - Animações naturais com objetos individuais e easing</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="📸 Carregue sua imagem",
                    type="filepath"
                )
                
                with gr.Row():
                    frame_rate = gr.Slider(
                        minimum=15, maximum=60, value=30, step=5,
                        label="🎬 FPS"
                    )
                    end_duration = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="⏱️ Duração final (segundos)"
                    )
                
                draw_mode = gr.Radio(
                    choices=["Apenas Contornos", "Contornos + Colorização"],
                    value="Apenas Contornos",
                    label="🎨 Modo de desenho"
                )
                
                generate_btn = gr.Button(
                    "🚀 Gerar Animação Human Speedpaint",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
                    <h3>🎯 Novidades do Human Speedpaint</h3>
                    <ul>
                        <li>🧠 Objetos individuais (Connected Components)</li>
                        <li>🌊 Easing natural (curvas sigmoide)</li>
                        <li>✏️ Stroke grouping (traços conectados)</li>
                        <li>🖐️ Movimento humano realista</li>
                    </ul>
                </div>
                """)
                
                video_output = gr.Video(
                    label="📹 Vídeo gerado",
                    visible=False
                )
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    lines=2
                )
        
        # Eventos
        def process_image(image_path, frame_rate, end_duration, draw_mode, progress=gr.Progress()):
            if image_path is None:
                return None, "❌ Por favor, carregue uma imagem"
            
            try:
                video_path, message = generate_sketch_video(
                    image_path, 15, frame_rate, 5, end_duration, draw_mode, progress
                )
                
                if video_path:
                    return video_path, f"✅ {message}"
                else:
                    return None, f"❌ {message}"
            except Exception as e:
                return None, f"❌ Erro: {str(e)}"
        
        generate_btn.click(
            fn=process_image,
            inputs=[image_input, frame_rate, end_duration, draw_mode],
            outputs=[video_output, status_output],
            show_progress=True
        )
    
    return app

if __name__ == "__main__":
    app = create_commercial_interface()
    app.launch()
