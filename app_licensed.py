"""
Image to Sketch Animation - Versão COMERCIAL
Sistema completo com licenciamento integrado via Stripe
"""

import os
import cv2
import numpy as np
import gradio as gr
import time

import math
import zipfile
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import shutil
import stripe
import random
import string
from datetime import datetime, timedelta

# Sistema de Licenciamento Integrado (Stripe API)

class LicenseManager:
    _validated_licenses: dict = {}
    _otp_codes: dict = {}  # {email: {"code": "123456", "created_at": datetime, "attempts": 0}}
    _sessions: dict = {}   # {session_id: {"email": "...", "created_at": datetime}}
    
    def __init__(self):
        self.stripe_secret_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.stripe_price_id = os.environ.get("STRIPE_PRICE_ID", "")
        self.payment_link = os.environ.get("STRIPE_PAYMENT_LINK", "")
        self._current_license: dict | None = None
        self._current_session_id: str | None = None
        self._demo_mode = not self.stripe_secret_key
        
        if self.stripe_secret_key:
            stripe.api_key = self.stripe_secret_key
    
    def generate_otp(self, email):
        """Gera um código OTP de 6 dígitos para o email"""
        email = email.strip().lower()
        code = ''.join(random.choices(string.digits, k=6))
        
        self._otp_codes[email] = {
            "code": code,
            "created_at": datetime.now(),
            "attempts": 0
        }
        
        return code
    
    def verify_otp(self, email, code):
        """Verifica se o código OTP é válido (válido por 10 minutos, máx 3 tentativas)"""
        email = email.strip().lower()
        
        if email not in self._otp_codes:
            return False, "❌ Código expirado. Solicite um novo código."
        
        otp_data = self._otp_codes[email]
        
        # Verificar tentativas
        if otp_data["attempts"] >= 3:
            del self._otp_codes[email]
            return False, "❌ Muitas tentativas. Solicite um novo código."
        
        # Verificar expiração (10 minutos)
        if (datetime.now() - otp_data["created_at"]).seconds > 600:
            del self._otp_codes[email]
            return False, "❌ Código expirado. Solicite um novo código."
        
        # Verificar código
        if otp_data["code"] != code:
            otp_data["attempts"] += 1
            return False, f"❌ Código incorreto. ({3 - otp_data['attempts']} tentativas restantes)"
        
        # Código válido - remover OTP
        del self._otp_codes[email]
        return True, "✅ Código verificado!"
    
    def create_session(self, email):
        """Cria uma sessão para o usuário (válida por 30 dias)"""
        email = email.strip().lower()
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        
        self._sessions[session_id] = {
            "email": email,
            "created_at": datetime.now()
        }
        
        return session_id
    
    def verify_session(self, session_id):
        """Verifica se a sessão é válida (válida por 30 dias)"""
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        
        # Verificar expiração (30 dias)
        if (datetime.now() - session["created_at"]).days > 30:
            del self._sessions[session_id]
            return None
        
        return session["email"]
    
    def logout(self, session_id):
        """Faz logout removendo a sessão"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def validate_by_email(self, email):
        """Valida se o email tem uma assinatura ativa no Stripe"""
        if self._demo_mode:
            return {
                "valid": True,
                "email": email,
                "plan": "pro",
                "activated_at": datetime.now().isoformat(),
                "status": "active",
                "demo": True
            }
        
        # Verificar cache primeiro (válido por 1 hora)
        cached = LicenseManager._validated_licenses.get(email)
        if cached:
            cache_time = cached.get("_cache_time")
            if cache_time and (datetime.now() - cache_time).seconds < 3600:
                return cached
        
        try:
            # Buscar cliente pelo email no Stripe
            customers = stripe.Customer.list(email=email.strip().lower(), limit=1)
            
            if not customers.data:
                return {"valid": False, "error": "Email não encontrado. Verifique se usou o mesmo email da compra."}
            
            customer = customers.data[0]
            
            # Buscar assinaturas ativas do cliente
            subscriptions = stripe.Subscription.list(
                customer=customer.id,
                status="active",
                limit=5
            )
            
            if subscriptions.data:
                sub = subscriptions.data[0]
                result = {
                    "valid": True,
                    "email": email,
                    "plan": "pro",
                    "activated_at": datetime.fromtimestamp(sub.created).isoformat(),
                    "status": "active",
                    "subscription_id": sub.id,
                    "_cache_time": datetime.now()
                }
                LicenseManager._validated_licenses[email] = result
                return result
            
            # Verificar também pagamentos únicos (one-time) caso mude o modelo
            payments = stripe.PaymentIntent.list(
                customer=customer.id,
                limit=5
            )
            
            for payment in payments.data:
                if payment.status == "succeeded":
                    result = {
                        "valid": True,
                        "email": email,
                        "plan": "pro",
                        "activated_at": datetime.fromtimestamp(payment.created).isoformat(),
                        "status": "active",
                        "payment_id": payment.id,
                        "_cache_time": datetime.now()
                    }
                    LicenseManager._validated_licenses[email] = result
                    return result
            
            return {"valid": False, "error": "Nenhuma assinatura ativa encontrada para este email."}
            
        except stripe.error.AuthenticationError:
            return {"valid": False, "error": "Erro de autenticação com o servidor de pagamentos."}
        except stripe.error.APIConnectionError:
            cached = LicenseManager._validated_licenses.get(email)
            if cached:
                return cached
            return {"valid": False, "error": "Sem conexão com o servidor de pagamentos."}
        except Exception as e:
            return {"valid": False, "error": f"Erro ao verificar licença: {str(e)}"}
    
    def request_otp(self, email):
        """Solicita OTP para o email (simula envio por email)"""
        if not email or len(email) < 5 or "@" not in email:
            return False, "❌ Por favor, insira um email válido."
        
        email = email.strip().lower()
        
        # Verificar se o email tem assinatura ativa no Stripe
        result = self.validate_by_email(email)
        if not result.get("valid"):
            return False, f"❌ {result.get('error', 'Email não encontrado')}"
        
        # Gerar OTP
        otp_code = self.generate_otp(email)
        
        # Exibir código na tela (futuramente pode ser enviado por email via SendGrid)
        return True, f"✅ Seu código de acesso:\n\n🔐 **{otp_code}**\n\n⏱️ Válido por 10 minutos. Insira abaixo para entrar."
    
    def verify_otp_and_login(self, email, otp_code):
        """Verifica OTP e cria sessão se válido"""
        email = email.strip().lower()
        
        # Verificar OTP
        valid, message = self.verify_otp(email, otp_code)
        if not valid:
            return False, message, None
        
        # Validar assinatura no Stripe
        result = self.validate_by_email(email)
        if not result.get("valid"):
            return False, f"❌ {result.get('error', 'Email não encontrado')}", None
        
        # Criar sessão
        session_id = self.create_session(email)
        self._current_license = result
        self._current_session_id = session_id
        
        return True, "✅ Login realizado com sucesso!", session_id
    
    def login_with_session(self, session_id):
        """Faz login usando session_id (para persistência)"""
        email = self.verify_session(session_id)
        if not email:
            return False, None
        
        # Validar assinatura no Stripe
        result = self.validate_by_email(email)
        if result.get("valid"):
            self._current_license = result
            self._current_session_id = session_id
            return True, email
        
        return False, None
    
    def logout_user(self):
        """Faz logout do usuário"""
        if self._current_session_id:
            LicenseManager._sessions.pop(self._current_session_id, None)
        self._current_license = None
        self._current_session_id = None
        return True
    
    def is_licensed(self):
        """Verifica se há licença ativa na sessão"""
        if self._current_license and self._current_license.get("valid"):
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
    
    def get_session_id(self):
        """Retorna o ID da sessão atual"""
        return self._current_session_id

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

def interpolate_points(p1, p2, steps):
    """Gera pontos intermediários entre p1 e p2 para movimento suave"""
    points = []
    for i in range(steps):
        alpha = (i + 1) / (steps + 1)
        x = int(p1[1] + (p2[1] - p1[1]) * alpha)
        y = int(p1[0] + (p2[0] - p1[0]) * alpha)
        points.append((y, x))
    return points

def get_sorted_components(img_thresh):
    """
    Retorna componentes conectados ordenados
    Retorna lista de (y_min, x_min, y_max, x_max, mask)
    """
    # Inverter para detectar objetos brancos (ou pretos se for o caso do desenho)
    # img_thresh já tem o desenho em preto (<10) e fundo branco
    # Precisamos inverter para connectedComponents achar os objetos (que devem ser brancos no fundo preto)
    img_inv = cv2.bitwise_not(img_thresh)
    
    # Dilatar levemente para conectar traços próximos que formam uma letra/objeto
    kernel = np.ones((3,3), np.uint8)
    img_dilated = cv2.dilate(img_inv, kernel, iterations=2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilated, connectivity=8)
    
    components = []
    for i in range(1, num_labels): # Pular background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20: # Ignorar ruído muito pequeno
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Criar máscara deste componente específico (usando o label original sem dilatação se possível, 
        # mas aqui usaremos a bounding box para extrair os grids reais depois)
        
        components.append({
            'y': y, 'x': x, 'h': h, 'w': w,
            'cx': int(centroids[i][0]),
            'cy': int(centroids[i][1])
        })
        
    # Ordenar componentes: Topo para baixo, Esquerda para direita
    # Uma heurística simples: ordenar por Y + (X / 10) para dar prefêrencia a linhas
    components.sort(key=lambda c: c['y'] + c['x'] * 0.1)
    
    return components

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
    Gera vídeo de sketch animation (Versão Otimizada V2)
    Melhorias: Fluidez, Ordem de Desenho e Tamanho do Vídeo
    """
    try:
        start_time = time.time()
        
        progress(0, desc="📸 Carregando imagem...")
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, "❌ Erro ao carregar imagem"
        
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        # Ajustar resolução (limitar a 1080p)
        aspect_ratio = img_wd / img_ht
        MAX_HEIGHT = 1080
        MAX_WIDTH = 1920
        
        if img_ht > MAX_HEIGHT or img_wd > MAX_WIDTH:
            if img_wd / MAX_WIDTH > img_ht / MAX_HEIGHT:
                target_wd = MAX_WIDTH
                target_ht = int(target_wd / aspect_ratio)
            else:
                target_ht = MAX_HEIGHT
                target_wd = int(target_ht * aspect_ratio)
        else:
            # Se for muito pequeno, aumentar um pouco para qualidade HD
            if img_ht < 720:
                scale = 720 / img_ht
                target_ht = 720
                target_wd = int(img_wd * scale)
            else:
                target_ht = img_ht
                target_wd = img_wd
        
        # GARANTIR que dimensões sejam divisíveis pelo split_len e pares
        target_wd = (target_wd // split_len) * split_len
        target_ht = (target_ht // split_len) * split_len
        
        target_ht = target_ht if target_ht % 2 == 0 else target_ht - 1
        target_wd = target_wd if target_wd % 2 == 0 else target_wd - 1
        
        progress(0.05, desc=f"🔧 Redimensionando para {target_wd}x{target_ht}...")
        img = cv2.resize(img, (target_wd, target_ht))
        
        # Processar imagem
        progress(0.1, desc="🎨 Analisando traços e objetos...")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        # Carregar mão
        hand, hand_mask, hand_mask_inv, hand_ht, hand_wd = preprocess_hand_image(
            HAND_PATH, HAND_MASK_PATH
        )
        
        # Configurar Vídeo
        now = datetime.now()
        video_name = f"sketch_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(SAVE_PATH, video_name)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_object = cv2.VideoWriter(video_path, fourcc, frame_rate, (target_wd, target_ht))
        
        # Canvas branco
        drawn_frame = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
        
        # Dividir em grids
        n_cuts_vertical = int(math.ceil(target_ht / split_len))
        n_cuts_horizontal = int(math.ceil(target_wd / split_len))
        
        grid_of_cuts = np.array(np.split(img_thresh, n_cuts_horizontal, axis=-1))
        grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
        
        # --- NOVA LÓGICA: COMPONENTES CONECTADOS ---
        # 1. Identificar grids que têm conteúdo (preto)
        cut_having_black = (grid_of_cuts < 50) * 1 # Usar 50 como limiar seguro
        cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
        # Matriz booleana indicando onde tem desenho
        has_drawing_grid = cut_having_black > 5 # Pelo menos 5 pixels pretos
        
        # 2. Obter componentes conectados ordenados
        progress(0.15, desc="🧩 Organizando ordem de desenho...")
        components = get_sorted_components(img_thresh)
        
        # 3. Gerar lista mestre de grids para desenhar, ordenados por componente
        final_draw_queue = []
        visited_grids = set()
        
        last_grid = (0, 0)
        
        for comp in components:
            # Encontrar grids que pertencem a este componente (baseado na bounding box)
            # Para ser mais preciso, verificamos se o grid está dentro da bbox
            
            comp_grids_indices = []
            
            start_row = max(0, comp['y'] // split_len)
            end_row = min(n_cuts_vertical, (comp['y'] + comp['h']) // split_len + 1)
            start_col = max(0, comp['x'] // split_len)
            end_col = min(n_cuts_horizontal, (comp['x'] + comp['w']) // split_len + 1)
            
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    if has_drawing_grid[r, c] and (r, c) not in visited_grids:
                        comp_grids_indices.append([r, c])
                        visited_grids.add((r, c))
            
            if not comp_grids_indices:
                continue
                
            # Agora aplicamos a lógica de vizinho mais próximo LOCALMENTE dentro do componente
            comp_grids_indices = np.array(comp_grids_indices)
            local_queue = []
            
            # Começar do ponto mais próximo do último ponto desenhado (ou do topo-esquerda do componente)
            if len(local_queue) == 0:
                dists = euc_dist(comp_grids_indices, last_grid)
                current_idx = np.argmin(dists)
            else:
                current_idx = 0
                
            curr_pos = comp_grids_indices[current_idx].copy()
            
            # Loop local
            while len(comp_grids_indices) > 0:
                # Adicionar atual
                local_queue.append(tuple(comp_grids_indices[current_idx]))
                
                # Remover atual da lista
                comp_grids_indices = np.delete(comp_grids_indices, current_idx, axis=0)
                
                if len(comp_grids_indices) == 0:
                    break
                
                # Achar próximo mais próximo
                dists = euc_dist(comp_grids_indices, curr_pos)
                current_idx = np.argmin(dists)
                curr_pos = comp_grids_indices[current_idx].copy()
            
            # Adicionar fila local à fila global
            final_draw_queue.extend(local_queue)
            if local_queue:
                last_grid = local_queue[-1]
        
        # Adicionar quaisquer grids restantes (que não caíram em componentes detectados)
        remaining_grids = []
        for r in range(n_cuts_vertical):
            for c in range(n_cuts_horizontal):
                if has_drawing_grid[r, c] and (r, c) not in visited_grids:
                    remaining_grids.append([r, c])
        
        if remaining_grids:
            remaining_grids = np.array(remaining_grids)
            # Ordenar globalmente ou anexar por proximidade
            while len(remaining_grids) > 0:
                dists = euc_dist(remaining_grids, last_grid)
                idx = np.argmin(dists)
                final_draw_queue.append(tuple(remaining_grids[idx]))
                last_grid = remaining_grids[idx]
                remaining_grids = np.delete(remaining_grids, idx, axis=0)
        
        # --- DESENHAR COM INTERPOLAÇÃO ---
        total_steps = len(final_draw_queue)
        progress(0.2, desc=f"✏️ Desenhando ({total_steps} grids)...")
        
        last_hand_pos = None
        counter = 0
        
        # Ajustar skip_rate para não ser muito rápido se tiver poucos grids
        if total_steps < 50:
            actual_skip = 1
        else:
            actual_skip = int(skip_rate)
            
        for i, (r, c) in enumerate(final_draw_queue):
            # Coordenadas do grid na imagem
            y_start = r * split_len
            x_start = c * split_len
            
            # Atualizar canvas (desenhar o grid)
            temp_drawing = np.zeros((split_len, split_len, 3))
            temp_drawing[:, :, 0] = grid_of_cuts[r][c]
            temp_drawing[:, :, 1] = grid_of_cuts[r][c]
            temp_drawing[:, :, 2] = grid_of_cuts[r][c]
            drawn_frame[y_start:y_start+split_len, x_start:x_start+split_len] = temp_drawing
            
            # Posição da mão (centro do grid)
            hand_x = x_start + split_len // 2
            hand_y = y_start + split_len // 2
            current_hand_pos = (hand_y, hand_x)
            
            # Interpolação se houver salto grande
            if last_hand_pos is not None:
                dist = np.sqrt((hand_y - last_hand_pos[0])**2 + (hand_x - last_hand_pos[1])**2)
                
                # Se a distância for grande (> 3 grids), mover a mão suavemente sem desenhar
                if dist > split_len * 3:
                    interp_steps = int(dist / (split_len)) # 1 frame a cada split_len px de movimento
                    interp_steps = min(interp_steps, 15) # Limitar a 15 frames max de viagem
                    
                    if interp_steps > 0:
                        travel_points = interpolate_points(last_hand_pos, current_hand_pos, interp_steps)
                        for ty, tx in travel_points:
                            f = draw_hand_on_img(
                                drawn_frame.copy(), hand, tx, ty,
                                hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd
                            )
                            video_object.write(f)
            
            # Gravar frame de desenho
            counter += 1
            if counter % actual_skip == 0 or i == total_steps - 1:
                f = draw_hand_on_img(
                    drawn_frame.copy(), hand, hand_x, hand_y,
                    hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd
                )
                video_object.write(f)
            
            last_hand_pos = current_hand_pos
            
            if i % 100 == 0:
                progress(0.2 + 0.5 * (i / total_steps), desc=f"✏️ Desenhando... {int(i/total_steps*100)}%")
        
        # === FASE 2: COLORIZAÇÃO (se selecionado, manter lógica simplificada mas com interpolação) ===
        if draw_mode == "Contornos + Colorização":
             # ... manter lógica similar mas adicionar interpolação se necessário ...
             # Para simplificar e não estourar o limite de linhas, usaremos uma versão enxuta da colorização
             # que reutiliza a queue se possível, mas colorização é baseada em regiões
             pass 
             # (Nota: A lógica de colorização original já era baseada em componentes, vamos mantê-la simples
             # ou idealmente refatorar também, mas o foco principal era o traço preto)
             
             progress(0.7, desc="🎨 Colorindo...")
             # Reimplemtando colorização simplificada
             img_thresh_inv = cv2.bitwise_not(img_thresh)
             kernel = np.ones((3, 3), np.uint8)
             img_dilated = cv2.dilate(img_thresh_inv, kernel, iterations=1)
             img_regions = cv2.bitwise_not(img_dilated)
             num_labels, labels = cv2.connectedComponents(img_regions)

             # Coletar regiões válidas
             regions = []
             for l in range(1, num_labels):
                 mask = (labels == l)
                 if np.sum(mask) > 50: # Ignorar ruido
                    regions.append({'label': l, 'size': np.sum(mask)})
             regions.sort(key=lambda x: x['size']) # Menores primeiro
             
             for idx, reg in enumerate(regions):
                 mask = (labels == reg['label'])
                 ys, xs = np.where(mask)
                 
                 # Pintar tudo de uma vez no canvas base
                 drawn_frame[ys, xs] = img[ys, xs]
                 
                 # Mover mão para o centro da região
                 cy, cx = int(np.mean(ys)), int(np.mean(xs))
                 
                 # Interpolação até lá
                 if last_hand_pos:
                     dist = np.sqrt((cy - last_hand_pos[0])**2 + (cx - last_hand_pos[1])**2)
                     if dist > split_len * 2:
                         interp_points = interpolate_points(last_hand_pos, (cy, cx), 5)
                         for ty, tx in interp_points:
                             video_object.write(draw_hand_on_img(drawn_frame.copy(), hand, tx, ty, hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd))
                 
                 # Frame final pintado
                 video_object.write(draw_hand_on_img(drawn_frame.copy(), hand, cx, cy, hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd))
                 last_hand_pos = (cy, cx)

        # Adicionar imagem final
        progress(0.9, desc="🖼️ Finalizando...")
        drawn_frame[:, :, :] = img
        # Remover mão suavemente? Por enquanto corte seco para a imagem final limpa
        
        for i in range(frame_rate * end_duration):
            video_object.write(drawn_frame)
        
        video_object.release()
        
        # OTIMIZAÇÃO DE VÍDEO (H.264)
        progress(0.95, desc="💾 Comprimindo vídeo (Isso vai reduzir o tamanho)...")
        try:
            import av
            h264_path = video_path.replace('.mp4', '_optimized.mp4')
            
            input_container = av.open(video_path)
            input_stream = input_container.streams.video[0]
            
            output_container = av.open(h264_path, mode='w')
            output_stream = output_container.add_stream('h264', rate=frame_rate)
            output_stream.width = input_stream.codec_context.width
            output_stream.height = input_stream.codec_context.height
            output_stream.pix_fmt = 'yuv420p'
            
            # CONFIGURAÇÕES CRÍTICAS PARA TAMANHO E PERFORMANCE
            output_stream.options = {
                'crf': '28',          # Maior compressão (antes era 20)
                'preset': 'veryfast', # Codificação mais rápida
                'profile': 'main'
            }
            
            for frame in input_container.decode(video=0):
                packet = output_stream.encode(frame)
                output_container.mux(packet)
            
            # Flush
            packet = output_stream.encode(None)
            output_container.mux(packet)
            
            input_container.close()
            output_container.close()
            
            # Substituir original
            if os.path.exists(video_path):
                os.remove(video_path)
            video_path = h264_path
            
        except Exception as e:
            print(f"⚠️ Erro na compressão H264: {e}. Usando arquivo original.")
            # Se falhar, mantemos o video_path original (que é grande, mas funciona)
        
        end_time = time.time()
        duration = end_time - start_time
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        return video_path, f"✅ Vídeo gerado com sucesso!\\n⏱️ Tempo: {duration:.1f}s\\n💾 Tamanho: {file_size_mb:.1f} MB\\n📁 Salvo em: {os.path.basename(video_path)}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erro fatal: {str(e)}"

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
        
        # Processar imagens sequencialmente para evitar sobrecarga de CPU no Railway
        with ThreadPoolExecutor(max_workers=1) as executor:
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
    (Otimizada V2: Componentes + Interpolação + H.264 Leve)
    """
    try:
        start_time = time.time()
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Erro ao carregar imagem: {image_path}"
        
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        # Ajustar resolução (limitar a 1920x1080 máximo)
        aspect_ratio = img_wd / img_ht
        MAX_HEIGHT = 1080
        MAX_WIDTH = 1920
        
        if img_ht > MAX_HEIGHT or img_wd > MAX_WIDTH:
            if img_wd / MAX_WIDTH > img_ht / MAX_HEIGHT:
                target_wd = MAX_WIDTH
                target_ht = int(target_wd / aspect_ratio)
            else:
                target_ht = MAX_HEIGHT
                target_wd = int(target_ht * aspect_ratio)
        else:
            # Se for muito pequeno, aumentar um pouco para qualidade HD
            if img_ht < 720:
                scale = 720 / img_ht
                target_ht = 720
                target_wd = int(img_wd * scale)
            else:
                target_ht = img_ht
                target_wd = img_wd
        
        # GARANTIR que dimensões sejam divisíveis pelo split_len
        target_wd = (target_wd // split_len) * split_len
        target_ht = (target_ht // split_len) * split_len
        
        target_ht = target_ht if target_ht % 2 == 0 else target_ht - 1
        target_wd = target_wd if target_wd % 2 == 0 else target_wd - 1
        
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
        
        # --- LÓGICA DE COMPONENTES ---
        cut_having_black = (grid_of_cuts < 50) * 1
        cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
        has_drawing_grid = cut_having_black > 5
        
        components = get_sorted_components(img_thresh)
        
        final_draw_queue = []
        visited_grids = set()
        last_grid = (0, 0)
        
        for comp in components:
            comp_grids_indices = []
            
            start_row = max(0, comp['y'] // split_len)
            end_row = min(n_cuts_vertical, (comp['y'] + comp['h']) // split_len + 1)
            start_col = max(0, comp['x'] // split_len)
            end_col = min(n_cuts_horizontal, (comp['x'] + comp['w']) // split_len + 1)
            
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    if has_drawing_grid[r, c] and (r, c) not in visited_grids:
                        comp_grids_indices.append([r, c])
                        visited_grids.add((r, c))
            
            if not comp_grids_indices:
                continue
                
            comp_grids_indices = np.array(comp_grids_indices)
            local_queue = []
            
            if len(local_queue) == 0:
                dists = euc_dist(comp_grids_indices, last_grid)
                current_idx = np.argmin(dists)
            else:
                current_idx = 0
                
            curr_pos = comp_grids_indices[current_idx].copy()
            
            while len(comp_grids_indices) > 0:
                local_queue.append(tuple(comp_grids_indices[current_idx]))
                comp_grids_indices = np.delete(comp_grids_indices, current_idx, axis=0)
                
                if len(comp_grids_indices) == 0:
                    break
                
                dists = euc_dist(comp_grids_indices, curr_pos)
                current_idx = np.argmin(dists)
                curr_pos = comp_grids_indices[current_idx].copy()
            
            final_draw_queue.extend(local_queue)
            if local_queue:
                last_grid = local_queue[-1]
                
        remaining_grids = []
        for r in range(n_cuts_vertical):
            for c in range(n_cuts_horizontal):
                if has_drawing_grid[r, c] and (r, c) not in visited_grids:
                    remaining_grids.append([r, c])
        
        if remaining_grids:
            remaining_grids = np.array(remaining_grids)
            while len(remaining_grids) > 0:
                dists = euc_dist(remaining_grids, last_grid)
                idx = np.argmin(dists)
                final_draw_queue.append(tuple(remaining_grids[idx]))
                last_grid = remaining_grids[idx]
                remaining_grids = np.delete(remaining_grids, idx, axis=0)
        
        # --- DESENHAR ---
        total_steps = len(final_draw_queue)
        last_hand_pos = None
        counter = 0
        
        actual_skip = 1 if total_steps < 50 else int(skip_rate)
        
        for i, (r, c) in enumerate(final_draw_queue):
            y_start = r * split_len
            x_start = c * split_len
            
            temp_drawing = np.zeros((split_len, split_len, 3))
            temp_drawing[:, :, 0] = grid_of_cuts[r][c]
            temp_drawing[:, :, 1] = grid_of_cuts[r][c]
            temp_drawing[:, :, 2] = grid_of_cuts[r][c]
            drawn_frame[y_start:y_start+split_len, x_start:x_start+split_len] = temp_drawing
            
            hand_x = x_start + split_len // 2
            hand_y = y_start + split_len // 2
            current_hand_pos = (hand_y, hand_x)
            
            if last_hand_pos is not None:
                dist = np.sqrt((hand_y - last_hand_pos[0])**2 + (hand_x - last_hand_pos[1])**2)
                if dist > split_len * 3:
                    interp_steps = int(dist / (split_len))
                    interp_steps = min(interp_steps, 15)
                    
                    if interp_steps > 0:
                        travel_points = interpolate_points(last_hand_pos, current_hand_pos, interp_steps)
                        for ty, tx in travel_points:
                            f = draw_hand_on_img(
                                drawn_frame.copy(), hand, tx, ty,
                                hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd
                            )
                            video_object.write(f)
            
            counter += 1
            if counter % actual_skip == 0 or i == total_steps - 1:
                f = draw_hand_on_img(
                    drawn_frame.copy(), hand, hand_x, hand_y,
                    hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd
                )
                video_object.write(f)
            
            last_hand_pos = current_hand_pos
        
        # === FASE 2: COLORIZAÇÃO (Simplificada para batch) ===
        if draw_mode == "Contornos + Colorização":
             img_thresh_inv = cv2.bitwise_not(img_thresh)
             kernel = np.ones((3, 3), np.uint8)
             img_dilated = cv2.dilate(img_thresh_inv, kernel, iterations=1)
             img_regions = cv2.bitwise_not(img_dilated)
             num_labels, labels = cv2.connectedComponents(img_regions)

             regions = []
             for l in range(1, num_labels):
                 mask = (labels == l)
                 if np.sum(mask) > 50:
                    regions.append({'label': l, 'size': np.sum(mask)})
             regions.sort(key=lambda x: x['size'])
             
             for idx, reg in enumerate(regions):
                 mask = (labels == reg['label'])
                 ys, xs = np.where(mask)
                 drawn_frame[ys, xs] = img[ys, xs]
                 
                 cy, cx = int(np.mean(ys)), int(np.mean(xs))
                 
                 if last_hand_pos:
                     dist = np.sqrt((cy - last_hand_pos[0])**2 + (cx - last_hand_pos[1])**2)
                     if dist > split_len * 2:
                         interp_points = interpolate_points(last_hand_pos, (cy, cx), 5)
                         for ty, tx in interp_points:
                             video_object.write(draw_hand_on_img(drawn_frame.copy(), hand, tx, ty, hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd))
                 
                 video_object.write(draw_hand_on_img(drawn_frame.copy(), hand, cx, cy, hand_mask_inv, hand_ht, hand_wd, target_ht, target_wd))
                 last_hand_pos = (cy, cx)
        
        # Adicionar imagem final
        drawn_frame[:, :, :] = img
        
        for i in range(frame_rate * end_duration):
            video_object.write(drawn_frame)
        
        video_object.release()
        
        # OTIMIZAÇÃO DE VÍDEO (H.264)
        try:
            import av
            h264_path = video_path.replace('.mp4', '_optimized.mp4')
            
            input_container = av.open(video_path, mode="r")
            output_container = av.open(h264_path, mode="w")
            
            in_stream = input_container.streams.video[0]
            out_stream = output_container.add_stream("h264", rate=in_stream.average_rate)
            out_stream.width = in_stream.codec_context.width
            out_stream.height = in_stream.codec_context.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "28", "preset": "veryfast", "profile": "main"}
            
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
            # Fallback silencioso
            pass
        
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

def request_otp_action(email):
    """Solicita OTP para o email"""
    success, message = license_manager.request_otp(email)
    
    if success:
        return message, gr.update(visible=False), gr.update(visible=True)
    else:
        return message, gr.update(visible=True), gr.update(visible=False)

def verify_otp_action(email, otp_code):
    """Verifica OTP e faz login"""
    success, message, session_id = license_manager.verify_otp_and_login(email, otp_code)
    
    if success:
        info = license_manager.get_license_info()
        success_msg = f"✅ {message}\n\n🎉 **Bem-vindo!**\n\n📧 Email: {info['email']}\n🎯 Plano: {info['plan'].upper()}"
        # Retornar session_id para ser salvo no localStorage via JavaScript
        return success_msg, session_id, gr.update(visible=False), gr.update(visible=True)
    else:
        return message, "", gr.update(visible=True), gr.update(visible=False)

def logout_action():
    """Faz logout do usuário"""
    license_manager.logout_user()
    # Retornar vazio para limpar localStorage
    return gr.update(visible=True), gr.update(visible=False), ""

def restore_session_from_storage(session_id_stored):
    """Restaura sessão do localStorage ao carregar a página"""
    if not session_id_stored:
        return None
    
    # Validar session_id no backend
    success, email = license_manager.login_with_session(session_id_stored)
    if success:
        return session_id_stored
    else:
        return None

# Interface Gradio Comercial
def create_commercial_interface():
    """Cria interface comercial com licenciamento"""
    
    # Verifica se está licenciado
    is_licensed = license_manager.is_licensed()
    
    with gr.Blocks(title="Whiteboard Animation Pro - Commercial", theme=gr.themes.Soft()) as app:
        
        # Estado para gerenciar sessão persistida
        session_state = gr.State(value=None)
        
        # JavaScript para gerenciar localStorage
        gr.HTML("""
        <script>
        function loadSessionFromStorage() {
            const sessionId = localStorage.getItem('whiteboardpro_session_id');
            return sessionId || '';
        }
        
        function saveSessionToStorage(sessionId) {
            if (sessionId) {
                localStorage.setItem('whiteboardpro_session_id', sessionId);
            }
        }
        
        function clearSessionFromStorage() {
            localStorage.removeItem('whiteboardpro_session_id');
        }
        
        // Carregar sessão ao iniciar
        window.addEventListener('load', function() {
            const sessionId = loadSessionFromStorage();
            if (sessionId) {
                console.log('Sessão restaurada do localStorage');
            }
        });
        </script>
        """)
        
        # ============================================================
        # GRUPO 0: LANDING PAGE (visível quando não logado)
        # ============================================================
        payment_url = license_manager.payment_link or 'https://buy.stripe.com/test_5kQ28rfZdd2RablaVicQU02'
        
        with gr.Group(visible=not is_licensed) as landing_group:
            
            # --- HERO SECTION ---
            gr.HTML(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 60px 20px; border-radius: 16px; margin-bottom: 30px; text-align: center; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><circle cx=%2220%22 cy=%2230%22 r=%2240%22 fill=%22rgba(102,126,234,0.08)%22/><circle cx=%2280%22 cy=%2270%22 r=%2250%22 fill=%22rgba(118,75,162,0.06)%22/></svg>'); background-size: cover;"></div>
                <div style="position: relative; z-index: 1;">
                    <div style="display: inline-block; background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 6px 20px; border-radius: 20px; font-size: 0.85em; font-weight: bold; margin-bottom: 20px; letter-spacing: 1px;">
                        🔥 PROMOÇÃO DE LANÇAMENTO - 50% OFF
                    </div>
                    <h1 style="margin: 0 0 15px 0; font-size: 3em; font-weight: 800; line-height: 1.1; background: linear-gradient(135deg, #fff, #e0e0ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        Transforme Imagens em<br>Vídeos Whiteboard Animados
                    </h1>
                    <p style="margin: 0 auto 30px auto; font-size: 1.25em; opacity: 0.85; max-width: 600px; line-height: 1.6;">
                        Crie vídeos profissionais de animação whiteboard em segundos. Perfeito para aulas, apresentações, reels e stories.
                    </p>
                    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                        <a href="{payment_url}" target="_blank" 
                           style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px 40px; border-radius: 50px; text-decoration: none; font-size: 1.15em; font-weight: bold; box-shadow: 0 8px 25px rgba(102,126,234,0.4); transition: all 0.3s;">
                            🛒 Assinar Agora - <span style="text-decoration: line-through; opacity: 0.7;">R$97,90</span> R$49,90/ano
                        </a>
                        <a href="#login-section" onclick="document.getElementById('login-section').scrollIntoView({{behavior:'smooth'}}); return false;"
                           style="display: inline-block; background: rgba(255,255,255,0.15); color: white; padding: 16px 40px; border-radius: 50px; text-decoration: none; font-size: 1.15em; font-weight: bold; border: 2px solid rgba(255,255,255,0.3); transition: all 0.3s;">
                            🔐 Já sou assinante
                        </a>
                    </div>
                    <p style="margin-top: 15px; font-size: 0.85em; opacity: 0.6;">Pagamento seguro via Stripe. Cancele quando quiser.</p>
                </div>
            </div>
            """)
            
            # --- VÍDEO DEMO (placeholder para YouTube) ---
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 40px;">
                <h2 style="color: #1a1a2e; font-size: 2em; margin-bottom: 20px;">🎬 Veja o resultado</h2>
                <div style="max-width: 720px; margin: 0 auto; background: #000; border-radius: 12px; overflow: hidden; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center;">
                    <!-- SUBSTITUIR pelo embed do YouTube quando tiver o link -->
                    <p style="color: #888; font-size: 1.2em;">🎥 Vídeo demonstrativo em breve</p>
                </div>
            </div>
            """)
            
            # --- COMO FUNCIONA ---
            gr.HTML("""
            <div style="margin-bottom: 40px;">
                <h2 style="text-align: center; color: #1a1a2e; font-size: 2em; margin-bottom: 30px;">⚡ Como Funciona</h2>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 250px; max-width: 320px; background: #f8f9ff; border-radius: 16px; padding: 30px; text-align: center; border: 1px solid #e8eaf6;">
                        <div style="font-size: 3em; margin-bottom: 15px;">📤</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">1. Faça Upload</h3>
                        <p style="color: #666; margin: 0; line-height: 1.6;">Envie sua imagem (slide, ilustração, diagrama). Aceita PNG, JPG e mais.</p>
                    </div>
                    <div style="flex: 1; min-width: 250px; max-width: 320px; background: #f8f9ff; border-radius: 16px; padding: 30px; text-align: center; border: 1px solid #e8eaf6;">
                        <div style="font-size: 3em; margin-bottom: 15px;">🎨</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">2. Processamento</h3>
                        <p style="color: #666; margin: 0; line-height: 1.6;">O app transforma automaticamente em animação whiteboard com mão desenhando.</p>
                    </div>
                    <div style="flex: 1; min-width: 250px; max-width: 320px; background: #f8f9ff; border-radius: 16px; padding: 30px; text-align: center; border: 1px solid #e8eaf6;">
                        <div style="font-size: 3em; margin-bottom: 15px;">📥</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">3. Download</h3>
                        <p style="color: #666; margin: 0; line-height: 1.6;">Baixe o vídeo MP4 pronto para usar em aulas, YouTube, Reels ou Stories.</p>
                    </div>
                </div>
            </div>
            """)
            
            # --- ESTILOS RECOMENDADOS ---
            gr.HTML("""
            <div style="margin-bottom: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 16px; padding: 40px 20px;">
                <h2 style="text-align: center; color: #1a1a2e; font-size: 2em; margin-bottom: 10px;">🖼️ Melhores Estilos de Imagem</h2>
                <p style="text-align: center; color: #666; margin-bottom: 30px; font-size: 1.1em;">O app funciona melhor com imagens nestes estilos. Use IA (ChatGPT, Gemini, etc.) para gerar!</p>
                
                <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
                    <!-- Estilo 1: Line Art Colorido -->
                    <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">RECOMENDADO</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">🎨 Line Art Colorido</h3>
                        <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços pretos nítidos sobre fundo branco com cores leves e translúcidas. Perfeito para slides educativos.</p>
                        <details style="cursor: pointer;">
                            <summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary>
                            <p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">
                                "Crie uma ilustração em line art minimalista técnico para slide de apresentação, estilo esboço profissional clean e didático. Use traços pretos nítidos sobre fundo branco puro 100%. Adicione cor de maneira restrita e elegante: contornos finos de destaque, preenchimento leve/translúcido (opacidade 10-30%). Estilo ultra-clean, técnico, alta legibilidade."
                            </p>
                        </details>
                    </div>
                    
                    <!-- Estilo 2: Preto e Branco -->
                    <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                        <div style="background: #333; color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">CLÁSSICO</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">✒️ Line Art P&B</h3>
                        <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços pretos limpos sobre fundo branco puro. Estilo whiteboard clássico, ideal para ilustrações técnicas.</p>
                        <details style="cursor: pointer;">
                            <summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary>
                            <p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">
                                "Crie uma imagem para slide de apresentação. Estilo Line Art minimalista em preto e branco, com traços pretos nítidos e limpos sobre fundo branco puro. Ilustração simplificada. Estilo de esboço técnico profissional, sem sombras complexas ou cores, apenas contornos e hachuras leves para profundidade."
                            </p>
                        </details>
                    </div>
                    
                    <!-- Estilo 3: Cartoon Educativo -->
                    <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                        <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">DIVERTIDO</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">🎭 Cartoon Educativo</h3>
                        <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços grossos, cores planas e vibrantes. Estilo cartoon moderno, ótimo para conteúdo descontraído.</p>
                        <details style="cursor: pointer;">
                            <summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary>
                            <p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">
                                "Crie uma ilustração em estilo cartoon educativo. Traços pretos grossos, uniformes e nítidos. Contornos limpos e fechados, estilo digital clean. Cores planas (flat colors), sem gradientes. Paleta profissional e vibrante. Fundo 100% branco puro para máxima legibilidade em slides."
                            </p>
                        </details>
                    </div>
                </div>
            </div>
            """)
            
            # --- DIMENSÕES RECOMENDADAS ---
            gr.HTML("""
            <div style="margin-bottom: 40px;">
                <h2 style="text-align: center; color: #1a1a2e; font-size: 2em; margin-bottom: 30px;">📐 Dimensões Recomendadas</h2>
                <div style="display: flex; gap: 30px; flex-wrap: wrap; justify-content: center;">
                    <div style="flex: 1; min-width: 280px; max-width: 400px; background: white; border-radius: 16px; padding: 30px; text-align: center; border: 2px solid #667eea; box-shadow: 0 4px 15px rgba(102,126,234,0.15);">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🖥️</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 5px 0;">16:9 (Paisagem)</h3>
                        <p style="color: #667eea; font-weight: bold; margin: 0 0 10px 0;">1920x1080 px</p>
                        <p style="color: #666; font-size: 0.9em; margin: 0; line-height: 1.5;">Ideal para <strong>slides, YouTube, aulas</strong> e apresentações. Melhor formato para o app.</p>
                    </div>
                    <div style="flex: 1; min-width: 280px; max-width: 400px; background: white; border-radius: 16px; padding: 30px; text-align: center; border: 2px solid #764ba2; box-shadow: 0 4px 15px rgba(118,75,162,0.15);">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">📱</div>
                        <h3 style="color: #1a1a2e; margin: 0 0 5px 0;">9:16 (Retrato)</h3>
                        <p style="color: #764ba2; font-weight: bold; margin: 0 0 10px 0;">1080x1920 px</p>
                        <p style="color: #666; font-size: 0.9em; margin: 0; line-height: 1.5;">Perfeito para <strong>Stories, Reels, TikTok</strong> e conteúdo vertical.</p>
                    </div>
                </div>
            </div>
            """)
            
            # --- PREÇO COM PROMOÇÃO ---
            gr.HTML(f"""
            <div style="margin-bottom: 40px; text-align: center;">
                <h2 style="color: #1a1a2e; font-size: 2em; margin-bottom: 30px;">💰 Investimento</h2>
                <div style="max-width: 420px; margin: 0 auto; background: white; border-radius: 20px; padding: 40px 30px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); border: 2px solid #667eea; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 8px; font-weight: bold; font-size: 0.9em; letter-spacing: 1px;">
                        🔥 PROMOÇÃO DE LANÇAMENTO - 50% OFF - POR TEMPO LIMITADO
                    </div>
                    <div style="margin-top: 30px;">
                        <h3 style="color: #1a1a2e; font-size: 1.5em; margin: 0 0 5px 0;">Plano Anual PRO</h3>
                        <p style="color: #999; margin: 0 0 15px 0;">Acesso completo a todas as funcionalidades</p>
                        <div style="margin: 20px 0;">
                            <span style="color: #999; font-size: 1.3em; text-decoration: line-through;">R$ 97,90</span>
                            <span style="color: #1a1a2e; font-size: 3em; font-weight: 800; margin-left: 10px;">R$ 49,90</span>
                            <span style="color: #666; font-size: 1em;">/ano</span>
                        </div>
                        <p style="color: #667eea; font-weight: bold; margin: 0 0 20px 0;">Apenas R$ 4,16/mês</p>
                        <ul style="text-align: left; color: #555; list-style: none; padding: 0; margin: 0 0 25px 0; line-height: 2;">
                            <li>✅ Processamento individual e em lote</li>
                            <li>✅ Modo Contornos + Colorização</li>
                            <li>✅ Download em MP4 e ZIP</li>
                            <li>✅ Suporte prioritário</li>
                            <li>✅ Todas as atualizações incluídas</li>
                            <li>✅ Cancele quando quiser</li>
                        </ul>
                        <a href="{payment_url}" target="_blank" 
                           style="display: block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 50px; text-decoration: none; font-size: 1.2em; font-weight: bold; box-shadow: 0 8px 25px rgba(102,126,234,0.4); text-align: center;">
                            🛒 Assinar Agora com 50% OFF
                        </a>
                        <p style="color: #999; font-size: 0.8em; margin-top: 12px;">Pagamento seguro via Stripe. Cartão ou Pix.</p>
                    </div>
                </div>
            </div>
            """)
            
            # --- FAQ ---
            gr.HTML("""
            <div style="margin-bottom: 40px; max-width: 700px; margin-left: auto; margin-right: auto;">
                <h2 style="text-align: center; color: #1a1a2e; font-size: 2em; margin-bottom: 30px;">❓ Perguntas Frequentes</h2>
                
                <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;">
                    <summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Que tipo de imagem funciona melhor?</summary>
                    <p style="color: #666; margin-top: 12px; line-height: 1.6;">Imagens com traços nítidos sobre fundo branco funcionam melhor: line art, ilustrações técnicas, diagramas, slides educativos e cartoons. Você pode gerar essas imagens usando IA (ChatGPT, Gemini, Midjourney) com os prompts que disponibilizamos acima.</p>
                </details>
                
                <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;">
                    <summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Posso usar para Stories e Reels?</summary>
                    <p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O app aceita imagens em qualquer dimensão. Para Stories/Reels, use imagens 9:16 (1080x1920). Para YouTube e slides, use 16:9 (1920x1080).</p>
                </details>
                
                <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;">
                    <summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Posso processar várias imagens de uma vez?</summary>
                    <p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O modo lote permite processar múltiplas imagens de uma vez e baixar todos os vídeos em um arquivo ZIP.</p>
                </details>
                
                <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;">
                    <summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Como funciona o pagamento?</summary>
                    <p style="color: #666; margin-top: 12px; line-height: 1.6;">O pagamento é processado pelo Stripe, a plataforma de pagamentos mais segura do mundo. Aceita cartão de crédito e Pix. A assinatura é anual e você pode cancelar a qualquer momento.</p>
                </details>
                
                <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;">
                    <summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">A promoção de 50% é por tempo limitado?</summary>
                    <p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O preço promocional de R$49,90/ano (50% de desconto) é exclusivo para os primeiros assinantes. O preço normal será R$97,90/ano.</p>
                </details>
            </div>
            """)
            
            # --- SEÇÃO DE LOGIN (id para scroll) ---
            gr.HTML('<div id="login-section"></div>')
            
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8eaf6 100%); border-radius: 16px; padding: 30px; margin-bottom: 20px;">
                <h2 style="text-align: center; color: #1a1a2e; margin: 0 0 20px 0;">🔐 Área do Assinante</h2>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div style="background: white; border-radius: 12px; padding: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.06);">
                        <h3 style="color: #495057; margin: 0 0 15px 0;">📦 Ainda não é assinante?</h3>
                        <ol style="color: #495057; line-height: 2; font-size: 1em;">
                            <li>Clique em <strong>"Assinar Agora"</strong></li>
                            <li>Pague com <strong>cartão ou Pix</strong></li>
                            <li>Digite o <strong>email da compra</strong> ao lado</li>
                            <li>Insira o <strong>código de acesso</strong> e pronto!</li>
                        </ol>
                        <div style="text-align: center; margin-top: 15px;">
                            <a href="{payment_url}" target="_blank" 
                               style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 32px; border-radius: 50px; text-decoration: none; font-size: 1.1em; font-weight: bold; box-shadow: 0 4px 15px rgba(102,126,234,0.3);">
                                🛒 Assinar com 50% OFF
                            </a>
                        </div>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #495057; margin: 0 0 15px 0;'>🔐 Já assinou? Faça login:</h3>")
                    
                    # ETAPA 1: Solicitar Email
                    with gr.Group(visible=True) as email_step:
                        email_input = gr.Textbox(
                            label="📧 Email usado na compra",
                            placeholder="seu@email.com",
                            info="Use o mesmo email que você usou no checkout do Stripe"
                        )
                        
                        request_otp_btn = gr.Button(
                            "📨 Enviar Código de Acesso",
                            variant="primary",
                            size="lg"
                        )
                        
                        email_result = gr.Markdown(
                            label="Resultado",
                            visible=True
                        )
                    
                    # ETAPA 2: Verificar OTP
                    with gr.Group(visible=False) as otp_step:
                        otp_input = gr.Textbox(
                            label="🔐 Código de 6 dígitos",
                            placeholder="123456",
                            info="Insira o código exibido acima",
                            max_lines=1
                        )
                        
                        verify_otp_btn = gr.Button(
                            "✅ Verificar e Entrar",
                            variant="primary",
                            size="lg"
                        )
                        
                        session_id_hidden = gr.Textbox(visible=False)
                        
                        otp_result = gr.Markdown(
                            label="Resultado",
                            visible=True
                        )
            
            # --- FOOTER ---
            gr.HTML("""
            <div style="background: #1a1a2e; color: white; border-radius: 12px; padding: 30px; margin-top: 30px; text-align: center;">
                <h3 style="margin: 0 0 10px 0; opacity: 0.9;">🎨 Whiteboard Animation Pro</h3>
                <p style="margin: 0; opacity: 0.6; font-size: 0.9em;">
                    &copy; 2025 Ai Infinitus - Todos os direitos reservados
                </p>
            </div>
            """)
        
        # GRUPO 2: App Completo (inicialmente visível se licenciado)
        with gr.Group(visible=is_licensed) as app_group:
            # Mostrar info da licença (se licenciado)
            if is_licensed:
                license_info = license_manager.get_license_info()
                with gr.Row():
                    with gr.Column(scale=9):
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
                    with gr.Column(scale=1):
                        logout_btn = gr.Button(
                            "🚪 Sair",
                            variant="stop",
                            size="sm",
                            scale=1
                        )
            
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
                    Versão Comercial &copy; 2025 Ai Infinitus - Todos os direitos reservados
                </p>
            </div>
            """)
        
        # Eventos de autenticação - novo fluxo com OTP
        # ETAPA 1: Solicitar OTP
        request_otp_btn.click(
            fn=request_otp_action,
            inputs=[email_input],
            outputs=[email_result, email_step, otp_step]
        )
        
        # ETAPA 2: Verificar OTP e fazer login
        def verify_and_save_session(email, otp_code):
            """Verifica OTP, faz login e salva session_id no localStorage"""
            otp_result_msg, session_id, activation_vis, app_vis = verify_otp_action(email, otp_code)
            
            # Se login bem-sucedido, salvar session_id no localStorage via JavaScript
            if session_id:
                # Adicionar script para salvar no localStorage
                save_script = f"""
                <script>
                saveSessionToStorage('{session_id}');
                console.log('Session salva no localStorage');
                </script>
                """
                otp_result_msg = otp_result_msg + "\n" + save_script
            
            return otp_result_msg, session_id, activation_vis, app_vis
        
        verify_otp_btn.click(
            fn=verify_and_save_session,
            inputs=[email_input, otp_input],
            outputs=[otp_result, session_id_hidden, landing_group, app_group]
        )
        
        # Evento de logout (se licenciado)
        if is_licensed:
            def logout_and_clear_storage():
                """Faz logout e limpa session_id do localStorage"""
                activation_vis, app_vis, _ = logout_action()
                
                # Adicionar script para limpar localStorage
                clear_script = """
                <script>
                clearSessionFromStorage();
                console.log('Session removida do localStorage');
                </script>
                """
                
                return activation_vis, app_vis, clear_script
            
            logout_btn.click(
                fn=logout_and_clear_storage,
                outputs=[landing_group, app_group, session_id_hidden]
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
            stats += f"🔥 Processamento otimizado com resolução HD\\n"
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
