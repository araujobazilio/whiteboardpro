"""
Image to Sketch Animation - Versão COMERCIAL
Sistema completo com licenciamento integrado para venda na Lemon Squeezy
"""

import os
import cv2
import numpy as np
import gradio as gr
import time
import datetime
import math
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import shutil
import json
import requests
from datetime import datetime, timedelta

# Sistema de Licenciamento Integrado (Lemon Squeezy API)

class LicenseManager:
    # Cache de licenças validadas (em memória, por sessão do servidor)
    _validated_licenses = {}
    
    def __init__(self):
        # Credenciais via variáveis de ambiente (segurança)
        self.api_key = os.environ.get("LEMONSQUEEZY_API_KEY", "")
        self.store_id = os.environ.get("LEMONSQUEEZY_STORE_ID", "")
        self.product_id = os.environ.get("LEMONSQUEEZY_PRODUCT_ID", "")
        self._current_license = None
        self._demo_mode = not self.api_key  # Modo demo se não houver API key
    
    def validate_license_key(self, license_key):
        """Valida chave de licença via API Lemon Squeezy ou modo demo"""
        # Modo demo: aceita qualquer chave se não houver API key configurada
        if self._demo_mode:
            return {
                "valid": True,
                "email": "demo@whiteboardpro.com",
                "plan": "pro",
                "activated_at": datetime.now().isoformat(),
                "status": "active",
                "key": license_key,
                "demo": True
            }
        
        try:
            url = "https://api.lemonsqueezy.com/v1/licenses/validate"
            payload = {
                "license_key": license_key,
                "instance_name": "web_app"
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            data = response.json()
            
            if data.get("valid") or data.get("license_key", {}).get("status") == "active":
                license_info = data.get("license_key", {})
                meta = data.get("meta", {})
                return {
                    "valid": True,
                    "email": license_info.get("user_email", ""),
                    "plan": meta.get("variant_name", "pro"),
                    "activated_at": license_info.get("created_at", datetime.now().isoformat()),
                    "status": "active",
                    "key": license_key
                }
            
            error_msg = data.get("error", "Chave inválida ou expirada")
            return {"valid": False, "error": error_msg}
            
        except requests.exceptions.ConnectionError:
            # Modo offline: verificar cache local
            cached = LicenseManager._validated_licenses.get(license_key)
            if cached:
                return cached
            return {"valid": False, "error": "Sem conexão com o servidor de licenças"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def activate_license(self, license_key, email):
        """Ativa licença via API Lemon Squeezy"""
        if not license_key or not email or len(license_key) < 5:
            return False, "❌ Preencha a chave de licença e o email corretamente."
        
        result = self.validate_license_key(license_key.strip())
        
        if result.get("valid"):
            # Salvar no cache em memória
            result["email"] = email
            LicenseManager._validated_licenses[license_key.strip()] = result
            self._current_license = result
            return True, "✅ Licença ativada com sucesso!"
        else:
            error = result.get("error", "Chave inválida")
            return False, f"❌ Erro: {error}"
    
    def is_licensed(self):
        """Verifica se há licença ativa na sessão"""
        if self._current_license and self._current_license.get("valid"):
            return True
        # Verificar se há alguma licença no cache
        for key, data in LicenseManager._validated_licenses.items():
            if data.get("valid"):
                self._current_license = data
                return True
        return False
    
    def get_license_info(self):
        """Obtém informações da licença ativa"""
        if self._current_license:
            return {
                "email": self._current_license.get("email", ""),
                "plan": self._current_license.get("plan", "pro"),
                "activated_at": self._current_license.get("activated_at", ""),
                "status": self._current_license.get("status", "active")
            }
        return None

# Configurações globais
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HAND_PATH = os.path.join(BASE_PATH, 'kivy', 'data', 'images', 'drawing-hand.png')
HAND_MASK_PATH = os.path.join(BASE_PATH, 'kivy', 'data', 'images', 'hand-mask.png')
SAVE_PATH = os.path.join(BASE_PATH, "saved_videos")
os.makedirs(SAVE_PATH, exist_ok=True)

# Inicializa gerenciador de licença
license_manager = LicenseManager()

# Funções originais do processamento de imagem
def euc_dist(arr1, point):
    """Calcula distância euclidiana entre array de pontos e um ponto"""
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))

def find_nearest_res(given):
    """Encontra a resolução padrão mais próxima"""
    arr = np.array([640, 360, 480, 1280, 720, 1920, 1080, 2560, 1440, 3840, 2160, 7680, 4320])
    idx = (np.abs(arr - given)).argmin()
    return arr[idx]

def get_extreme_coordinates(mask):
    """Encontra coordenadas extremas de uma máscara"""
    indices = np.where(mask == 255)
    x = indices[1]
    y = indices[0]
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))
    return topleft, bottomright

def preprocess_hand_image(hand_path, hand_mask_path):
    """Processa a imagem da mão para desenho"""
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    
    top_left, bottom_right = get_extreme_coordinates(hand_mask)
    hand = hand[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask = hand_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask_inv = 255 - hand_mask
    
    hand_mask = hand_mask / 255
    hand_mask_inv = hand_mask_inv / 255
    
    hand_bg_ind = np.where(hand_mask == 0)
    hand[hand_bg_ind] = [0, 0, 0]
    
    hand_ht, hand_wd = hand.shape[0], hand.shape[1]
    
    return hand, hand_mask, hand_mask_inv, hand_ht, hand_wd

def draw_hand_on_img(drawing, hand, x, y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    """Desenha a mão na posição especificada"""
    remaining_ht = img_ht - y
    remaining_wd = img_wd - x
    
    crop_hand_ht = min(hand_ht, remaining_ht)
    crop_hand_wd = min(hand_wd, remaining_wd)
    
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]
    
    for c in range(3):
        drawing[y:y+crop_hand_ht, x:x+crop_hand_wd, c] = (
            drawing[y:y+crop_hand_ht, x:x+crop_hand_wd, c] * hand_mask_inv_cropped
        )
    
    drawing[y:y+crop_hand_ht, x:x+crop_hand_wd] += hand_cropped
    return drawing

def common_divisors(num1, num2):
    """Encontra divisores comuns de dois números"""
    divisors1 = []
    divisors2 = []
    common_divs = []
    
    for i in range(1, num1 + 1):
        if num1 % i == 0:
            divisors1.append(i)
    
    for i in range(1, num2 + 1):
        if num2 % i == 0:
            divisors2.append(i)
    
    for divisor in divisors1:
        if divisor in divisors2:
            common_divs.append(divisor)
    
    common_divs.sort()
    return common_divs

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
    Gera vídeo de sketch animation
    
    Args:
        image_path: Caminho da imagem
        split_len: Tamanho da divisão em grid
        frame_rate: FPS do vídeo
        skip_rate: Taxa de pulo (velocidade)
        end_duration: Duração da imagem final
        draw_mode: Modo de desenho - 'Apenas Contornos' ou 'Contornos + Colorização'
        progress: Objeto de progresso
    """
    try:
        start_time = time.time()
        
        progress(0, desc="📸 Carregando imagem...")
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Erro ao carregar imagem"
        
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        # Ajustar resolução (limitar a 1920x1080 máximo para balancear qualidade e performance)
        aspect_ratio = img_wd / img_ht
        
        # Limitar a 1080p para qualidade HD excelente
        MAX_HEIGHT = 1080
        MAX_WIDTH = 1920
        
        if img_ht > MAX_HEIGHT or img_wd > MAX_WIDTH:
            # Calcular nova dimensão mantendo aspecto
            if img_wd / MAX_WIDTH > img_ht / MAX_HEIGHT:
                # Largura é o fator limitante
                target_wd = MAX_WIDTH
                target_ht = int(target_wd / aspect_ratio)
            else:
                # Altura é o fator limitante
                target_ht = MAX_HEIGHT
                target_wd = int(target_ht * aspect_ratio)
        else:
            target_ht = img_ht
            target_wd = img_wd
        
        # GARANTIR que dimensões sejam divisíveis pelo split_len
        # Isso evita o erro "array split does not result in an equal division"
        target_wd = (target_wd // split_len) * split_len
        target_ht = (target_ht // split_len) * split_len
        
        # Garantir dimensões mínimas (evitar 0 ou negativo)
        min_dim = split_len * 2
        target_wd = max(target_wd, min_dim)
        target_ht = max(target_ht, min_dim)
        
        # Ajustar para valores pares (necessário para codecs)
        target_ht = target_ht if target_ht % 2 == 0 else target_ht - 1
        target_wd = target_wd if target_wd % 2 == 0 else target_wd - 1
        
        progress(0.05, desc=f"🔧 Redimensionando de {img_wd}x{img_ht} para {target_wd}x{target_ht} (Full HD)...")
        img = cv2.resize(img, (target_wd, target_ht))
        
        # Processar imagem
        progress(0.1, desc="🎨 Processando imagem...")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        # Carregar mão
        progress(0.15, desc="✋ Carregando imagem da mão...")
        hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
            HAND_PATH, HAND_MASK_PATH
        )
        
        # Criar nome do vídeo
        now = datetime.now()
        video_name = f"sketch_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(SAVE_PATH, video_name)
        
        # Criar objeto de vídeo
        progress(0.2, desc="🎬 Criando vídeo...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_object = cv2.VideoWriter(video_path, fourcc, frame_rate, (target_wd, target_ht))
        
        # Canvas branco
        drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
        
        # Dividir em grids
        progress(0.25, desc="📐 Dividindo imagem em grids...")
        n_cuts_vertical = int(math.ceil(target_ht / split_len))
        n_cuts_horizontal = int(math.ceil(target_wd / split_len))
        
        grid_of_cuts = np.array(np.split(img_thresh, n_cuts_horizontal, axis=-1))
        grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
        
        # Encontrar grids com pixels pretos
        cut_having_black = (grid_of_cuts < 10) * 1
        cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
        cut_black_indices = np.array(np.where(cut_having_black > 0)).T
        
        total_cuts = len(cut_black_indices)
        selected_ind = 0
        counter = 0
        
        progress(0.3, desc=f"✏️ Desenhando ({total_cuts} grids)...")
        
        # Desenhar
        while len(cut_black_indices) > 1:
            selected_ind_val = cut_black_indices[selected_ind].copy()
            range_v_start = selected_ind_val[0] * split_len
            range_v_end = range_v_start + split_len
            range_h_start = selected_ind_val[1] * split_len
            range_h_end = range_h_start + split_len
            
            temp_drawing = np.zeros((split_len, split_len, 3))
            temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            
            drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing
            
            hand_coord_x = range_h_start + int(split_len / 2)
            hand_coord_y = range_v_start + int(split_len / 2)
            
            drawn_frame_with_hand = draw_hand_on_img(
                drawn_frame.copy(), hand.copy(), hand_coord_x, hand_coord_y,
                hand_mask_inv.copy(), hand_ht, hand_wd, target_ht, target_wd
            )
            
            cut_black_indices[selected_ind] = cut_black_indices[-1]
            cut_black_indices = cut_black_indices[:-1]
            
            del selected_ind
            
            euc_arr = euc_dist(cut_black_indices, selected_ind_val)
            selected_ind = np.argmin(euc_arr)
            
            counter += 1
            if counter % skip_rate == 0:
                video_object.write(drawn_frame_with_hand)
            
            if counter % 100 == 0:
                prog_percent = 0.3 + (0.6 * (1 - len(cut_black_indices) / total_cuts))
                progress(prog_percent, desc=f"✏️ Desenhando... {100 * (1 - len(cut_black_indices) / total_cuts):.1f}%")
        
        # === FASE 2: COLORIZAÇÃO POR REGIÕES (se modo selecionado) ===
        if draw_mode == "Contornos + Colorização":
            progress(0.7, desc="🎨 Detectando regiões para colorir...")
            
            # Inverter threshold para encontrar regiões fechadas
            img_thresh_inv = cv2.bitwise_not(img_thresh)
            kernel = np.ones((3, 3), np.uint8)
            img_thresh_dilated = cv2.dilate(img_thresh_inv, kernel, iterations=1)
            img_thresh_for_regions = cv2.bitwise_not(img_thresh_dilated)
            
            # Encontrar regiões conectadas
            num_labels, labels = cv2.connectedComponents(img_thresh_for_regions)
            
            # Calcular info de cada região
            region_info = []
            for label_id in range(1, num_labels):
                region_mask = (labels == label_id)
                region_size = np.sum(region_mask)
                
                if region_size < 50:
                    continue
                
                ys, xs = np.where(region_mask)
                if len(ys) == 0:
                    continue
                    
                # Pular regiões brancas/quase brancas
                mean_color = np.mean(img[ys, xs], axis=0)
                if np.all(mean_color > 245):
                    continue
                
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                
                region_info.append({
                    'label_id': label_id,
                    'size': region_size,
                    'cx': cx,
                    'cy': cy,
                    'ys': ys,
                    'xs': xs
                })
            
            # Ordenar por tamanho (menores primeiro)
            region_info.sort(key=lambda r: r['size'])
            
            total_regions = len(region_info)
            color_skip = max(1, skip_rate // 2)
            block_counter = 0
            
            progress(0.72, desc=f"🎨 Colorindo {total_regions} regiões...")
            
            # Processar cada região por blocos de grid (meio termo: não pixel a pixel, nem tudo de uma vez)
            for reg_idx, region in enumerate(region_info):
                ys, xs = region['ys'], region['xs']
                
                # Agrupar pixels em blocos de grid usando NumPy (vetorizado, rápido)
                grid_rows = ys // split_len
                grid_cols = xs // split_len
                grid_keys_arr = grid_rows * 10000 + grid_cols  # chave única por bloco
                unique_keys = np.unique(grid_keys_arr)
                
                # Montar lista de blocos com seus pixels
                blocks = []
                for key in unique_keys:
                    mask = grid_keys_arr == key
                    blocks.append((ys[mask], xs[mask], int(key // 10000), int(key % 10000)))
                
                if len(blocks) == 0:
                    continue
                
                # Ordenar blocos por proximidade (nearest neighbor simples)
                ordered_blocks = [blocks[0]]
                remaining = list(range(1, len(blocks)))
                current_r, current_c = blocks[0][2], blocks[0][3]
                
                while remaining:
                    best_idx = 0
                    best_dist = float('inf')
                    for i, idx in enumerate(remaining):
                        dr = blocks[idx][2] - current_r
                        dc = blocks[idx][3] - current_c
                        dist = dr * dr + dc * dc
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                    
                    chosen = remaining.pop(best_idx)
                    ordered_blocks.append(blocks[chosen])
                    current_r, current_c = blocks[chosen][2], blocks[chosen][3]
                
                # Pintar bloco por bloco com animação
                for block_ys, block_xs, gr_row, gr_col in ordered_blocks:
                    # Aplicar cor do bloco inteiro de uma vez (vetorizado)
                    drawn_frame[block_ys, block_xs] = img[block_ys, block_xs]
                    
                    block_counter += 1
                    if block_counter % color_skip == 0:
                        # Posicionar mão no centro do bloco
                        hx = min(gr_col * split_len + split_len // 2, target_wd - 1)
                        hy = min(gr_row * split_len + split_len // 2, target_ht - 1)
                        
                        drawn_frame_with_hand = draw_hand_on_img(
                            drawn_frame.copy(), hand.copy(), hx, hy,
                            hand_mask_inv.copy(), hand_ht, hand_wd, target_ht, target_wd
                        )
                        video_object.write(drawn_frame_with_hand)
                
                # Atualizar progresso
                if reg_idx % 10 == 0 and total_regions > 0:
                    prog_pct = 0.72 + (0.18 * (reg_idx + 1) / total_regions)
                    progress(prog_pct, desc=f"🎨 Colorindo... {reg_idx + 1}/{total_regions}")
        
        # Adicionar imagem final
        progress(0.9, desc="🖼️ Adicionando imagem final...")
        drawn_frame[:, :, :] = img
        
        for i in range(frame_rate * end_duration):
            video_object.write(drawn_frame)
        
        video_object.release()
        
        # Tentar converter para H264
        progress(0.95, desc="🔄 Convertendo para H264...")
        try:
            import av
            h264_path = video_path.replace('.mp4', '_h264.mp4')
            
            input_container = av.open(video_path, mode="r")
            output_container = av.open(h264_path, mode="w")
            
            in_stream = input_container.streams.video[0]
            out_stream = output_container.add_stream("h264", rate=in_stream.average_rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "20"}
            
            for frame in input_container.decode(video=0):
                packet = out_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
            
            packet = out_stream.encode(None)
            if packet:
                output_container.mux(packet)
            
            output_container.close()
            input_container.close()
            
            os.remove(video_path)
            video_path = h264_path
        except Exception as e:
            print(f"Conversão H264 falhou (usando MP4 original): {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        progress(1.0, desc="✅ Concluído!")
        
        return video_path, f"✅ Vídeo gerado com sucesso em {duration:.1f}s!\\n📁 Salvo em: {video_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erro: {str(e)}\\n\\n{traceback.format_exc()}"
        return None, error_msg

def generate_sketch_video_batch(
    image_paths,
    split_len,
    frame_rate,
    skip_rate,
    end_duration,
    draw_mode="Apenas Contornos",
    progress=gr.Progress()
):
    """
    Gera vídeos de sketch animation em lote
    """
    try:
        start_time = time.time()
        total_images = len(image_paths)
        
        if total_images == 0:
            return None, "❌ Nenhuma imagem selecionada"
        
        progress(0.05, desc=f"📸 Processando {total_images} imagens...")
        
        # Criar diretório temporário para os vídeos
        temp_dir = tempfile.mkdtemp(prefix="batch_videos_")
        generated_videos = []
        failed_images = []
        
        # Função para processar uma única imagem
        def process_single_image(idx_image_path):
            idx, image_path = idx_image_path
            try:
                # Usar a função original sem progresso para evitar conflitos
                video_path, message = generate_sketch_video_single(
                    image_path, split_len, frame_rate, skip_rate, end_duration, draw_mode
                )
                if video_path:
                    return idx, video_path, None
                else:
                    return idx, None, message
            except Exception as e:
                return idx, None, str(e)
        
        # Processar imagens em paralelo (máximo 4 threads para não sobrecarregar)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Enviar todas as tarefas
            future_to_idx = {
                executor.submit(process_single_image, (idx, path)): idx 
                for idx, path in enumerate(image_paths)
            }
            
            # Coletar resultados mantendo ordem
            results = [None] * total_images
            completed = 0
            
            for future in as_completed(future_to_idx):
                idx, video_path, error = future.result()
                results[idx] = (video_path, error)
                completed += 1
                
                progress_percent = 0.1 + (0.7 * completed / total_images)
                progress(progress_percent, desc=f"✏️ Processando... {completed}/{total_images} imagens")
        
        # Organizar resultados e gerar estatísticas
        for idx, (video_path, error) in enumerate(results):
            if video_path:
                generated_videos.append(video_path)
            else:
                failed_images.append((image_paths[idx], error))
        
        progress(0.85, desc="📦 Criando arquivo ZIP...")
        
        # Criar arquivo ZIP com todos os vídeos
        now = datetime.now()
        zip_name = f"batch_videos_{now.strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(SAVE_PATH, zip_name)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for video_path in generated_videos:
                if os.path.exists(video_path):
                    zipf.write(video_path, os.path.basename(video_path))
            
            # Adicionar arquivo de log
            log_content = f"Relatório de Processamento em Lote\\n"
            log_content += f"Data: {now.strftime('%Y-%m-%d %H:%M:%S')}\\n"
            log_content += f"Total de imagens: {total_images}\\n"
            log_content += f"Vídeos gerados: {len(generated_videos)}\\n"
            log_content += f"Falhas: {len(failed_images)}\\n\\n"
            
            if failed_images:
                log_content += "Imagens com falha:\\n"
                for img_path, error in failed_images:
                    log_content += f"- {os.path.basename(img_path)}: {error}\\n"
            
            zipf.writestr("relatorio.txt", log_content)
        
        # Limpar arquivos temporários
        for video_path in generated_videos:
            if os.path.exists(video_path):
                os.remove(video_path)
        
        os.rmdir(temp_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        progress(1.0, desc="✅ Concluído!")
        
        success_msg = f"✅ Processamento em lote concluído em {duration:.1f}s!\\n"
        success_msg += f"📊 {len(generated_videos)} vídeos gerados com sucesso\\n"
        if failed_images:
            success_msg += f"⚠️ {len(failed_images)} imagens falharam\\n"
        success_msg += f"📁 Arquivo ZIP salvo em: {zip_path}"
        
        return zip_path, success_msg
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Erro no processamento em lote: {str(e)}\\n\\n{traceback.format_exc()}"
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
    Versão simplificada da função original para uso em batch processing
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Erro ao carregar imagem: {image_path}"
        
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        # Ajustar resolução
        aspect_ratio = img_wd / img_ht
        target_ht = find_nearest_res(img_ht)
        target_wd = find_nearest_res(int(target_ht * aspect_ratio))
        img = cv2.resize(img, (target_wd, target_ht))
        
        # Processar imagem
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        # Carregar mão
        hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
            HAND_PATH, HAND_MASK_PATH
        )
        
        # Criar nome do vídeo
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        video_name = f"sketch_{base_name}_{int(time.time())}.mp4"
        video_path = os.path.join(tempfile.gettempdir(), video_name)
        
        # Criar objeto de vídeo
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_object = cv2.VideoWriter(video_path, fourcc, frame_rate, (target_wd, target_ht))
        
        # Canvas branco
        drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
        
        # Dividir em grids
        n_cuts_vertical = int(math.ceil(target_ht / split_len))
        n_cuts_horizontal = int(math.ceil(target_wd / split_len))
        
        grid_of_cuts = np.array(np.split(img_thresh, n_cuts_horizontal, axis=-1))
        grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
        
        # Encontrar grids com pixels pretos
        cut_having_black = (grid_of_cuts < 10) * 1
        cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
        cut_black_indices = np.array(np.where(cut_having_black > 0)).T
        
        total_cuts = len(cut_black_indices)
        selected_ind = 0
        counter = 0
        
        # Desenhar
        while len(cut_black_indices) > 1:
            selected_ind_val = cut_black_indices[selected_ind].copy()
            range_v_start = selected_ind_val[0] * split_len
            range_v_end = range_v_start + split_len
            range_h_start = selected_ind_val[1] * split_len
            range_h_end = range_h_start + split_len
            
            temp_drawing = np.zeros((split_len, split_len, 3))
            temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
            
            drawn_frame[range_v_start:range_v_end, range_h_start:range_h_end] = temp_drawing
            
            hand_coord_x = range_h_start + int(split_len / 2)
            hand_coord_y = range_v_start + int(split_len / 2)
            
            drawn_frame_with_hand = draw_hand_on_img(
                drawn_frame.copy(), hand.copy(), hand_coord_x, hand_coord_y,
                hand_mask_inv.copy(), hand_ht, hand_wd, target_ht, target_wd
            )
            
            cut_black_indices[selected_ind] = cut_black_indices[-1]
            cut_black_indices = cut_black_indices[:-1]
            
            del selected_ind
            
            euc_arr = euc_dist(cut_black_indices, selected_ind_val)
            selected_ind = np.argmin(euc_arr)
            
            counter += 1
            if counter % skip_rate == 0:
                video_object.write(drawn_frame_with_hand)
        
        # === FASE 2: COLORIZAÇÃO POR REGIÕES (se modo selecionado) ===
        if draw_mode == "Contornos + Colorização":
            # Inverter threshold para encontrar regiões fechadas
            img_thresh_inv = cv2.bitwise_not(img_thresh)
            
            # Dilatar levemente os contornos para fechar gaps pequenos
            kernel = np.ones((3, 3), np.uint8)
            img_thresh_dilated = cv2.dilate(img_thresh_inv, kernel, iterations=1)
            img_thresh_for_regions = cv2.bitwise_not(img_thresh_dilated)
            
            # Encontrar regiões conectadas
            num_labels, labels = cv2.connectedComponents(img_thresh_for_regions)
            
            # Calcular info de cada região
            region_info = []
            for label_id in range(1, num_labels):
                region_mask = (labels == label_id)
                region_size = np.sum(region_mask)
                
                if region_size < 20:
                    continue
                
                ys, xs = np.where(region_mask)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                
                region_info.append({
                    'label_id': label_id,
                    'size': region_size,
                    'cx': cx,
                    'cy': cy,
                    'mask': region_mask
                })
            
            # Ordenar: regiões menores primeiro (objetos), maiores por último (fundo)
            region_info.sort(key=lambda r: r['size'])
            
            color_skip = max(1, skip_rate // 2)
            color_counter = 0
            
            for region in region_info:
                region_mask = region['mask']
                ys, xs = np.where(region_mask)
                
                if len(ys) == 0:
                    continue
                
                # Pular regiões brancas/quase brancas (canvas já é branco)
                mean_color = np.mean(img[ys, xs], axis=0)
                if np.all(mean_color > 240):
                    continue
                
                # Agrupar pixels em mini-blocos baseados no grid
                pixel_grids = {}
                for py, px in zip(ys, xs):
                    grid_row = py // split_len
                    grid_col = px // split_len
                    key = (grid_row, grid_col)
                    if key not in pixel_grids:
                        pixel_grids[key] = []
                    pixel_grids[key].append((py, px))
                
                # Ordenar grids por distância euclidiana
                grid_keys = list(pixel_grids.keys())
                if len(grid_keys) == 0:
                    continue
                
                grid_coords = np.array(grid_keys)
                ordered_grids = []
                current_idx = 0
                remaining = list(range(len(grid_coords)))
                
                while remaining:
                    ordered_grids.append(remaining[current_idx])
                    current_pos = grid_coords[remaining[current_idx]].copy()
                    remaining.pop(current_idx)
                    
                    if remaining:
                        remaining_coords = grid_coords[remaining]
                        dists = euc_dist(remaining_coords, current_pos)
                        current_idx = np.argmin(dists)
                
                # Pintar cada mini-bloco desta região
                for grid_idx in ordered_grids:
                    grid_key = grid_keys[grid_idx]
                    pixels = pixel_grids[grid_key]
                    
                    for py, px in pixels:
                        drawn_frame[py, px] = img[py, px]
                    
                    color_counter += 1
                    if color_counter % color_skip == 0:
                        hx = min(grid_key[1] * split_len + split_len // 2, target_wd - 1)
                        hy = min(grid_key[0] * split_len + split_len // 2, target_ht - 1)
                        
                        drawn_frame_with_hand = draw_hand_on_img(
                            drawn_frame.copy(), hand.copy(), hx, hy,
                            hand_mask_inv.copy(), hand_ht, hand_wd, target_ht, target_wd
                        )
                        video_object.write(drawn_frame_with_hand)
        
        # Adicionar imagem final
        drawn_frame[:, :, :] = img
        
        for i in range(frame_rate * end_duration):
            video_object.write(drawn_frame)
        
        video_object.release()
        
        # Tentar converter para H264
        try:
            import av
            h264_path = video_path.replace('.mp4', '_h264.mp4')
            
            input_container = av.open(video_path, mode="r")
            output_container = av.open(h264_path, mode="w")
            
            in_stream = input_container.streams.video[0]
            out_stream = output_container.add_stream("h264", rate=in_stream.average_rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "20"}
            
            for frame in input_container.decode(video=0):
                packet = out_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
            
            packet = out_stream.encode(None)
            if packet:
                output_container.mux(packet)
            
            output_container.close()
            input_container.close()
            
            os.remove(video_path)
            video_path = h264_path
        except Exception as e:
            print(f"Conversão H264 falhou (usando MP4 original): {e}")
        
        return video_path, "Sucesso"
        
    except Exception as e:
        return None, str(e)

def get_image_info(image):
    """Obtém informações da imagem e sugere split_len"""
    if image is None:
        return "Nenhuma imagem carregada", []
    
    try:
        img = cv2.imread(image)
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        aspect_ratio = img_wd / img_ht
        target_ht = find_nearest_res(img_ht)
        target_wd = find_nearest_res(int(target_ht * aspect_ratio))
        
        divisors = common_divisors(target_ht, target_wd)
        
        # Filtrar divisores razoáveis (entre 5 e 40)
        good_divisors = [d for d in divisors if 5 <= d <= 40]
        
        info = f"📐 Resolução original: {img_wd} x {img_ht}\\n"
        info += f"🎯 Resolução do vídeo: {target_wd} x {target_ht}\\n"
        info += f"📊 Split lengths sugeridos: {good_divisors}"
        
        return info, gr.Dropdown(choices=good_divisors, value=good_divisors[len(good_divisors)//2] if good_divisors else 10)
        
    except Exception as e:
        return f"❌ Erro ao processar imagem: {e}", gr.Dropdown(choices=[10], value=10)

# Funções de licenciamento para interface
def check_license_status():
    """Verifica status da licença"""
    if license_manager.is_licensed():
        info = license_manager.get_license_info()
        return f"✅ **LICENÇA ATIVADA**\\n\\n📧 Email: {info['email']}\\n🎯 Plano: {info['plan'].upper()}\\n📅 Ativada em: {info['activated_at'][:10]}"
    else:
        return "❌ **LICENÇA NÃO ATIVADA**\\n\\nPor favor, ative sua licença para usar todas as funcionalidades."

def activate_license_action(license_key, email):
    """Ativa licença e retorna atualizações de visibilidade"""
    if not license_key or not email:
        return "❌ Por favor, preencha todos os campos.", gr.update(visible=True), gr.update(visible=False)
    
    success, message = license_manager.activate_license(license_key, email)
    
    if success:
        # Retorna mensagem de sucesso + esconde tela de ativação + mostra app
        info = license_manager.get_license_info()
        success_msg = f"✅ {message}\n\n🎉 **Licença ativada com sucesso!**\n\n📧 Email: {info['email']}\n🎯 Plano: {info['plan'].upper()}"
        return success_msg, gr.update(visible=False), gr.update(visible=True)
    else:
        return f"❌ {message}", gr.update(visible=True), gr.update(visible=False)

# Interface Gradio Comercial
def create_commercial_interface():
    """Cria interface comercial com licenciamento"""
    
    # Verifica se está licenciado
    is_licensed = license_manager.is_licensed()
    
    with gr.Blocks(title="Whiteboard Animation Pro - Commercial", theme=gr.themes.Soft()) as app:
        
        # Cabeçalho profissional
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h1 style="margin: 0; font-size: 2.5em;">🎨 Whiteboard Animation Pro</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Transforme imagens em vídeos de animação whiteboard profissionais</p>
        </div>
        """)
        
        # GRUPO 1: Tela de Ativação (inicialmente visível se não licenciado)
        with gr.Group(visible=not is_licensed) as activation_group:
            gr.HTML("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h3 style="color: #856404; margin: 0 0 10px 0;">🔑 Ativação Necessária</h3>
                <p style="color: #856404; margin: 0;">Para usar todas as funcionalidades, por favor, ative sua licença.</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 20px;">
                        <h3 style="color: #495057; margin: 0 0 15px 0;">📦 Como obter sua licença:</h3>
                        <ol style="color: #495057; line-height: 1.6;">
                            <li>Adquira sua licença em nossa página de vendas</li>
                            <li>Você receberá a chave de licença por email automaticamente</li>
                            <li>Insira a chave e o email no formulário ao lado</li>
                            <li>Clique em "Ativar Licença" e pronto!</li>
                        </ol>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #495057; margin: 0 0 15px 0;'>🔐 Ativar Licença</h3>")
                    
                    with gr.Group():
                        license_key_input = gr.Textbox(
                            label="🔑 Chave de Licença",
                            placeholder="Digite sua chave de licença",
                            type="password"
                        )
                        
                        email_input = gr.Textbox(
                            label="📧 Email Cadastrado",
                            placeholder="seu@email.com"
                        )
                        
                        activate_btn = gr.Button(
                            "🚀 Ativar Licença",
                            variant="primary",
                            size="lg"
                        )
                        
                        activation_result = gr.Markdown(
                            label="Resultado",
                            visible=True
                        )
            
            # Preview limitado
            gr.HTML("""
            <div style="background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 15px; margin-top: 20px;">
                <h4 style="color: #0c5460; margin: 0 0 10px 0;">👁️ Versão Demo</h4>
                <p style="color: #0c5460; margin: 0;">Após ativar sua licença, você terá acesso a:</p>
                <ul style="color: #0c5460; margin: 10px 0 0 20px;">
                    <li>✅ Processamento individual de imagens</li>
                    <li>✅ Processamento em lote (múltiplas imagens)</li>
                    <li>✅ Download automático em ZIP</li>
                    <li>✅ Suporte prioritário</li>
                    <li>✅ Atualizações vitalícias</li>
                </ul>
            </div>
            """)
        
        # GRUPO 2: App Completo (inicialmente visível se licenciado)
        with gr.Group(visible=is_licensed) as app_group:
            # Mostrar info da licença (se licenciado)
            if is_licensed:
                license_info = license_manager.get_license_info()
                gr.HTML(f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <h3 style="color: #155724; margin: 0 0 10px 0;">✅ Licença Ativada</h3>
                    <p style="color: #155724; margin: 0;">
                        <strong>Email:</strong> {license_info['email']} | 
                        <strong>Plano:</strong> {license_info['plan'].upper()} | 
                        <strong>Ativada em:</strong> {license_info['activated_at'][:10]}
                    </p>
                </div>
                """)
            
            with gr.Tabs():
                # Tab de Processamento Individual
                with gr.TabItem("🖼️ Processamento Individual"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", label="📤 Upload da Imagem")
                            
                            image_info = gr.Textbox(label="ℹ️ Informações da Imagem", lines=3, interactive=False)
                            
                            with gr.Accordion("⚙️ Configurações", open=True):
                                split_len = gr.Dropdown(
                                    choices=[5, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40],
                                    value=10,
                                    label="Split Length (tamanho da divisão)"
                                )
                                
                                frame_rate = gr.Slider(
                                    minimum=15, maximum=60, value=30, step=1,
                                    label="Frame Rate (FPS)"
                                )
                                
                                skip_rate = gr.Slider(
                                    minimum=1, maximum=20, value=5, step=1,
                                    label="Skip Rate (velocidade)"
                                )
                                
                                end_duration = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="Duração da Imagem Final (segundos)"
                                )
                                
                                draw_mode = gr.Radio(
                                    choices=["Apenas Contornos", "Contornos + Colorização"],
                                    value="Apenas Contornos",
                                    label="🎨 Modo de Desenho",
                                    info="'Apenas Contornos' = whiteboard clássico (preto e branco). 'Contornos + Colorização' = desenha contornos e depois preenche com as cores originais."
                                )
                            
                            generate_btn = gr.Button("🚀 Gerar Vídeo", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            video_output = gr.Video(label="🎬 Vídeo Gerado")
                            status_output = gr.Textbox(label="📊 Status", lines=3)
                
                # Tab de Processamento em Lote
                with gr.TabItem("📦 Processamento em Lote"):
                    gr.HTML("""
                    <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                        <h4 style="color: #0066cc; margin: 0 0 10px 0;">⚡ Processamento em Massa</h4>
                        <p style="color: #0066cc; margin: 0;">Processa múltiplas imagens simultaneamente e baixe tudo em um arquivo ZIP organizado.</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_images = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                label="📁 Upload de Múltiplas Imagens"
                            )
                            
                            batch_info = gr.Textbox(
                                label="📊 Informações do Lote",
                                lines=2,
                                interactive=False,
                                placeholder="Nenhuma imagem selecionada"
                            )
                            
                            with gr.Accordion("⚙️ Configurações do Lote", open=True):
                                batch_split_len = gr.Dropdown(
                                    choices=[5, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40],
                                    value=10,
                                    label="Split Length (tamanho da divisão)"
                                )
                                
                                batch_frame_rate = gr.Slider(
                                    minimum=15, maximum=60, value=30, step=1,
                                    label="Frame Rate (FPS)"
                                )
                                
                                batch_skip_rate = gr.Slider(
                                    minimum=1, maximum=20, value=5, step=1,
                                    label="Skip Rate (velocidade)"
                                )
                                
                                batch_end_duration = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="Duração da Imagem Final (segundos)"
                                )
                                
                                batch_draw_mode = gr.Radio(
                                    choices=["Apenas Contornos", "Contornos + Colorização"],
                                    value="Apenas Contornos",
                                    label="🎨 Modo de Desenho",
                                    info="'Apenas Contornos' = whiteboard clássico (preto e branco). 'Contornos + Colorização' = desenha contornos e depois preenche com as cores originais."
                                )
                            
                            batch_generate_btn = gr.Button(
                                "🚀 Processar Lote", 
                                variant="primary", 
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            batch_zip_output = gr.File(
                                label="📦 Arquivo ZIP para Download"
                            )
                            
                            batch_status_output = gr.Textbox(
                                label="📊 Status do Processamento",
                                lines=5,
                                interactive=False
                            )
                            
                            batch_stats = gr.Textbox(
                                label="📈 Estatísticas",
                                lines=3,
                                interactive=False,
                                placeholder="Processamento não iniciado"
                            )
            
            # Rodapé profissional
            gr.HTML("""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 20px; margin-top: 20px; text-align: center;">
                <h3 style="color: #495057; margin: 0 0 10px 0;">🎯 Whiteboard Animation Pro</h3>
                <p style="color: #6c757d; margin: 0;">
                    Versão Comercial - Todos os direitos reservados<br>
                    Suporte: WhatsApp (XX) XXXXX-XXXX | Email: suporte@whiteboardpro.com
                </p>
            </div>
            """)
        
        # Eventos de ativação - atualiza visibilidade dos grupos
        activate_btn.click(
            fn=activate_license_action,
            inputs=[license_key_input, email_input],
            outputs=[activation_result, activation_group, app_group]
        )
        
        # Funções auxiliares para interface
        def update_batch_info(files):
            if files is None:
                return "Nenhuma imagem selecionada"
            
            file_count = len(files)
            total_size = sum(os.path.getsize(file if isinstance(file, str) else file.name) for file in files) / (1024 * 1024)  # MB
            
            info = f"📁 {file_count} imagens selecionadas\\n"
            info += f"💾 Tamanho total: {total_size:.1f} MB"
            
            return info
        
        def process_batch_images(files, split_len, frame_rate, skip_rate, end_duration, draw_mode, progress=gr.Progress()):
            if files is None or len(files) == 0:
                return None, "❌ Nenhuma imagem selecionada", "Nenhuma imagem para processar"
            
            # Extrair caminhos dos arquivos (compatível com Gradio 5 e 6)
            image_paths = [file if isinstance(file, str) else file.name for file in files]
            
            # Processar em lote
            zip_path, message = generate_sketch_video_batch(
                image_paths, split_len, frame_rate, skip_rate, end_duration, draw_mode, progress
            )
            
            # Gerar estatísticas
            stats = f"📊 Estatísticas do Processamento:\\n"
            stats += f"🔥 Processamento paralelo com 4 threads\\n"
            stats += f"⚡ Otimização automática de recursos"
            
            if zip_path:
                return zip_path, message, stats
            else:
                return None, message, stats
        
        # Eventos - Processamento Individual
        image_input.change(
            fn=get_image_info,
            inputs=[image_input],
            outputs=[image_info, split_len]
        )
        
        generate_btn.click(
            fn=generate_sketch_video,
            inputs=[image_input, split_len, frame_rate, skip_rate, end_duration, draw_mode],
            outputs=[video_output, status_output]
        )
        
        # Eventos - Processamento em Lote
        batch_images.change(
            fn=update_batch_info,
            inputs=[batch_images],
            outputs=[batch_info]
        )
        
        batch_generate_btn.click(
            fn=process_batch_images,
            inputs=[batch_images, batch_split_len, batch_frame_rate, batch_skip_rate, batch_end_duration, batch_draw_mode],
            outputs=[batch_zip_output, batch_status_output, batch_stats],
            show_progress=True
        )
    
    return app

if __name__ == "__main__":
    print("=" * 70)
    print("🎨 WHITEBOARD ANIMATION PRO - VERSÃO COMERCIAL")
    print("=" * 70)
    print()
    print("🚀 Iniciando servidor Gradio...")
    print(f"📁 Vídeos serão salvos em: {SAVE_PATH}")
    print()
    
    # Cria e inicia a interface
    app = create_commercial_interface()
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        inbrowser=True
    )
