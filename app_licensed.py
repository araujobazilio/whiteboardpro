"""
Image to Sketch Animation - Versão COMERCIAL
Sistema completo com licenciamento integrado via Stripe
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
import stripe
import random
import string
from datetime import datetime, timedelta
import json
import sqlite3
import hashlib
import smtplib
import secrets
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
from engine.settings import ProjectSettings, Quality, HandStyle, SketchColorMode, SequenceMode

# Sistema de Licenciamento Integrado (Stripe API + SQLite)

# Caminho do banco de dados SQLite
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "whiteboardpro.db")

def get_db():
    """Retorna conexão SQLite thread-safe"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    """Cria tabelas se não existirem"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            stripe_customer_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS rate_limit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            action TEXT NOT NULL,
            attempt_count INTEGER DEFAULT 1,
            first_attempt TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, action)
        );
    """)
    conn.commit()
    conn.close()

# Inicializar banco ao importar o módulo
init_db()

class LicenseManager:
    _validated_licenses = {}
    
    def __init__(self):
        self.stripe_secret_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self.stripe_price_id = os.environ.get("STRIPE_PRICE_ID", "")
        self.payment_link = os.environ.get("STRIPE_PAYMENT_LINK", "")
        self._current_license = None
        self._current_session_id = None
        self._demo_mode = not self.stripe_secret_key
        
        if self.stripe_secret_key:
            stripe.api_key = self.stripe_secret_key
    
    # ==========================================
    # AUTENTICAÇÃO COM EMAIL + SENHA (SQLite)
    # ==========================================
    
    def register_user(self, email, password):
        """Cadastra um novo usuário com email e senha"""
        email = email.strip().lower()
        
        if not email or len(email) < 5 or "@" not in email:
            return False, "❌ Por favor, insira um email válido."
        
        if not password or len(password) < 6:
            return False, "❌ A senha deve ter no mínimo 6 caracteres."
        
        # Verificar se o email tem assinatura ativa no Stripe
        result = self.validate_by_email(email)
        if not result.get("valid"):
            return False, f"❌ {result.get('error', 'Email não encontrado no sistema de pagamentos.')}"
        
        # Verificar se já existe cadastro
        conn = get_db()
        try:
            existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                conn.close()
                return False, "❌ Este email já possui cadastro. Use a opção de login."
            
            # Criar hash da senha e salvar
            pw_hash = generate_password_hash(password)
            stripe_cid = result.get("subscription_id", "") or result.get("payment_id", "")
            
            conn.execute(
                "INSERT INTO users (email, password_hash, stripe_customer_id) VALUES (?, ?, ?)",
                (email, pw_hash, stripe_cid)
            )
            conn.commit()
            return True, "✅ Cadastro realizado com sucesso! Agora faça login."
        except Exception as e:
            return False, f"❌ Erro ao cadastrar: {str(e)}"
        finally:
            conn.close()

    def _send_email_smtp(self, recipient_email, subject, body_text):
        """Envia email usando SMTP nativo do Python."""
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
        smtp_port_raw = os.environ.get("SMTP_PORT", "587").strip()
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_pass = os.environ.get("SMTP_PASS", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", smtp_user).strip()

        if not smtp_host or not smtp_user or not smtp_pass or not smtp_from:
            return False, "❌ SMTP não configurado. Defina SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS e SMTP_FROM."

        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            return False, "❌ SMTP_PORT inválida."

        msg = MIMEText(body_text, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = recipient_email

        try:
            # Gmail recomendado: 587 + STARTTLS
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, [recipient_email], msg.as_string())
            return True, "✅ Email enviado."
        except Exception as e:
            return False, f"❌ Falha ao enviar email: {str(e)}"

    def _check_rate_limit(self, identifier, action, max_attempts=5, window_minutes=15):
        """Verifica e atualiza contador de rate limit. Retorna (permitido, minutos_restantes)."""
        identifier = (identifier or "").strip().lower()
        if not identifier:
            identifier = "unknown"

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db()
        try:
            row = conn.execute(
                "SELECT attempt_count, first_attempt FROM rate_limit WHERE identifier = ? AND action = ?",
                (identifier, action),
            ).fetchone()

            if not row:
                conn.execute(
                    "INSERT INTO rate_limit (identifier, action, attempt_count, first_attempt) VALUES (?, ?, 1, ?)",
                    (identifier, action, now_str),
                )
                conn.commit()
                return True, 0

            attempt_count = int(row["attempt_count"])
            first_attempt = datetime.strptime(row["first_attempt"], "%Y-%m-%d %H:%M:%S")
            elapsed = (now - first_attempt).total_seconds() / 60

            if elapsed > window_minutes:
                conn.execute(
                    "UPDATE rate_limit SET attempt_count = 1, first_attempt = ?, updated_at = CURRENT_TIMESTAMP WHERE identifier = ? AND action = ?",
                    (now_str, identifier, action),
                )
                conn.commit()
                return True, 0

            if attempt_count >= max_attempts:
                remaining = max(1, int(window_minutes - elapsed))
                return False, remaining

            conn.execute(
                "UPDATE rate_limit SET attempt_count = attempt_count + 1, updated_at = CURRENT_TIMESTAMP WHERE identifier = ? AND action = ?",
                (identifier, action),
            )
            conn.commit()
            return True, 0
        finally:
            conn.close()

    def _clear_rate_limit(self, identifier, action):
        """Limpa contador após sucesso."""
        identifier = (identifier or "").strip().lower()
        if not identifier:
            return

        conn = get_db()
        try:
            conn.execute("DELETE FROM rate_limit WHERE identifier = ? AND action = ?", (identifier, action))
            conn.commit()
        finally:
            conn.close()

    def request_password_reset(self, email, reset_base_url):
        """Solicita recuperação de senha e envia token por email."""
        email = (email or "").strip().lower()

        if not email or len(email) < 5 or "@" not in email:
            return False, "❌ Informe um email válido."

        if not reset_base_url:
            return False, "❌ URL base de reset não configurada."

        allowed, remaining = self._check_rate_limit(email, "password_reset", max_attempts=5, window_minutes=15)
        if not allowed:
            return False, f"❌ Muitas solicitações de recuperação. Tente novamente em {remaining} min."

        conn = get_db()
        try:
            user = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()

            # Resposta neutra para não vazar existência do email.
            if not user:
                return True, "✅ Se o email existir, enviaremos o link de recuperação."

            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                "UPDATE password_reset_tokens SET used = 1 WHERE email = ? AND used = 0",
                (email,)
            )
            conn.execute(
                "INSERT INTO password_reset_tokens (email, token, expires_at) VALUES (?, ?, ?)",
                (email, token, expires_at)
            )
            conn.commit()

            separator = "&" if "?" in reset_base_url else "?"
            reset_link = f"{reset_base_url}{separator}token={token}"

            body_text = (
                "Você solicitou a redefinição da sua senha no WhiteboardPro.\n\n"
                f"Link de recuperação (expira em 1 hora):\n{reset_link}\n\n"
                "Se você não solicitou, ignore este email."
            )
            sent, send_msg = self._send_email_smtp(
                recipient_email=email,
                subject="Recuperação de senha - WhiteboardPro",
                body_text=body_text,
            )

            if not sent:
                conn.execute("DELETE FROM password_reset_tokens WHERE token = ?", (token,))
                conn.commit()
                return False, send_msg

            self._clear_rate_limit(email, "password_reset")

            return True, "✅ Se o email existir, enviaremos o link de recuperação."
        except Exception as e:
            return False, f"❌ Erro ao solicitar recuperação: {str(e)}"
        finally:
            conn.close()

    def verify_password_reset_token(self, token):
        """Retorna email se token for válido, senão None."""
        token = (token or "").strip()
        if not token:
            return None

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        try:
            row = conn.execute(
                """
                SELECT email FROM password_reset_tokens
                WHERE token = ? AND used = 0 AND expires_at > ?
                """,
                (token, now_str),
            ).fetchone()
            if not row:
                return None
            return row["email"]
        finally:
            conn.close()

    def reset_password_with_token(self, token, new_password):
        """Redefine a senha usando token válido."""
        token = (token or "").strip()
        if not token:
            return False, "❌ Token inválido."

        if not new_password or len(new_password) < 6:
            return False, "❌ A nova senha deve ter no mínimo 6 caracteres."

        email = self.verify_password_reset_token(token)
        if not email:
            return False, "❌ Token inválido ou expirado. Solicite novo link."

        conn = get_db()
        try:
            new_hash = generate_password_hash(new_password)
            conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
            conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
            conn.commit()
            return True, "✅ Senha redefinida com sucesso. Faça login com a nova senha."
        except Exception as e:
            return False, f"❌ Erro ao redefinir senha: {str(e)}"
        finally:
            conn.close()
    
    def login_with_password(self, email, password):
        """Faz login com email e senha, retorna (sucesso, mensagem, session_id)"""
        email = email.strip().lower()
        
        if not email or len(email) < 5 or "@" not in email:
            return False, "❌ Por favor, insira um email válido.", None
        
        if not password:
            return False, "❌ Por favor, insira sua senha.", None

        allowed, remaining = self._check_rate_limit(email, "login", max_attempts=5, window_minutes=15)
        if not allowed:
            return False, f"❌ Muitas tentativas de login. Tente novamente em {remaining} min.", None
        
        conn = get_db()
        try:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            
            if not user:
                return False, "❌ Email não cadastrado. Faça seu cadastro primeiro.", None
            
            if not check_password_hash(user["password_hash"], password):
                return False, "❌ Senha incorreta.", None
            
            # Verificar se assinatura Stripe ainda está ativa
            result = self.validate_by_email(email)
            if not result.get("valid"):
                return False, f"❌ {result.get('error', 'Assinatura não está mais ativa.')}", None
            
            # Criar sessão no SQLite
            session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            conn.execute(
                "INSERT INTO sessions (session_id, email) VALUES (?, ?)",
                (session_id, email)
            )
            conn.commit()
            
            self._current_license = result
            self._current_session_id = session_id

            self._clear_rate_limit(email, "login")
            
            return True, "✅ Login realizado com sucesso!", session_id
        except Exception as e:
            return False, f"❌ Erro ao fazer login: {str(e)}", None
        finally:
            conn.close()
    
    # ==========================================
    # SESSÕES (SQLite persistente)
    # ==========================================
    
    def create_session(self, email):
        """Cria uma sessão persistente no SQLite"""
        email = email.strip().lower()
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, email) VALUES (?, ?)",
                (session_id, email)
            )
            conn.commit()
        finally:
            conn.close()
        
        return session_id
    
    def verify_session(self, session_id):
        """Verifica se a sessão é válida (válida por 30 dias)"""
        conn = get_db()
        try:
            session = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            
            if not session:
                return None
            
            # Verificar expiração (30 dias)
            created = datetime.fromisoformat(session["created_at"])
            if (datetime.now() - created).days > 30:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return None
            
            return session["email"]
        finally:
            conn.close()
    
    def logout(self, session_id):
        """Faz logout removendo a sessão do SQLite"""
        conn = get_db()
        try:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return True
        finally:
            conn.close()
    
    # ==========================================
    # VALIDAÇÃO STRIPE (mantida intacta)
    # ==========================================
    
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
    
    # ==========================================
    # MÉTODOS DE INTERFACE (mantidos compatíveis)
    # ==========================================
    
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
            self.logout(self._current_session_id)
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

def imread_safe(path, flags=cv2.IMREAD_COLOR):
    """
    Lê imagens suportando caminhos com acentos/caracteres especiais no Windows.
    Substituto robusto para cv2.imread.
    """
    try:
        # Lê o arquivo como stream de bytes e decodifica
        # Isso contorna o problema do OpenCV com caminhos não-ASCII no Windows
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, flags)
    except Exception as e:
        print(f"Erro ao ler imagem {path}: {e}")
        return None

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
    hand = imread_safe(hand_path)
    hand_mask = imread_safe(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    
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



def calculate_engine_params(img_shape, duration, fps, quality_preset="HD"):
    """
    Calcula parâmetros técnicos (split_len, skip_rate) baseados em metas de UX (duração, fps).
    """
    height, width = img_shape[:2]
    
    # 1. Definir split_len baseado na qualidade
    if quality_preset == "SD":
        target_split = 15 # Menos detalhes, grids maiores
    else: # HD
        target_split = 10 # Mais detalhes, grids menores
        
    # Ajustar para ser divisor se possível, ou próximo
    # Simplificação: forçar 10 ou 15 para MVP
    split_len = target_split
    
    # 2. Estimar total de grids
    # Redimensionamento acontece dentro de generate_sketch_video, precisamos estimar
    # Assumindo HD 1920x1080
    if quality_preset == "HD":
        est_w, est_h = 1920, 1080
    else:
        est_w, est_h = 720, 480
        
    n_cols = est_w // split_len
    n_rows = est_h // split_len
    total_grids_approx = n_cols * n_rows * 0.2 # Assumindo 20% de área desenhável (white space skip)
    # A estimativa de 20% é conservadora para desenhos de traço. Fotos cheias seriam 100%.
    # Melhor: usar um valor heurístico fixo ou calcular pré-processamento.
    # Para MVP: vamos confiar que generate_sketch_video vai lidar com o ritmo se dermos um skip_rate inicial
    
    # Abordagem reversa: Skip rate controla a velocidade.
    # Total Frames = Duration * FPS
    # Total Steps (grids com tinta) ~ estimated 5000 (exemplo)
    # Skip Rate = Total Steps / Total Frames
    
    # Como não sabemos Total Steps antes de processar a imagem, vamos passar os parametros de alvo
    # para a função principal e deixar ela calcular exato, ou usar uma media.
    # VAMOS ALTERAR generate_sketch_video para aceitar duration_sec e calcular skip_rate INTERNAMENTE.
    
    return split_len

def generate_sketch_video(
    image_path,
    split_len,
    frame_rate,
    skip_rate, # Mantido para compatibilidade, mas pode ser ignorado se duration for passado
    end_duration,
    draw_mode="Apenas Contornos",
    progress=gr.Progress(),
    # Novos parâmetros opcionais para paridade
    sketch_duration_sec=None,
    fill_duration_sec=None
):
    """
    Gera o vídeo de animação whiteboard a partir da imagem.
    
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
        img = imread_safe(image_path)
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
        
        # --- CÁLCULO DINÂMICO DE VELOCIDADE (MVP) ---
        # Se sketch_duration_sec foi passado, ignorar skip_rate e calcular um novo
        if sketch_duration_sec is not None and sketch_duration_sec > 0:
            total_frames_target = sketch_duration_sec * frame_rate
            # Quantos steps de desenho temos? total_cuts
            # Queremos que total_cuts / novo_skip = total_frames_target
            # Logo: novo_skip = total_cuts / total_frames_target
            calc_skip = total_cuts / total_frames_target
            skip_rate = max(1, int(calc_skip))
            # Ajuste fino: se skip for muito alto, vai ficar muito rápido/picotado.
            # Se skip for 1, vai demorar total_cuts frames.
            
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
                # FIX: Aumentar threshold para ignorar mais "brancos sujos"
                if np.all(mean_color > 225): # Antes 245
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
            
            # --- CÁLCULO DINÂMICO DE PREENCHIMENTO ---
            if fill_duration_sec is not None and fill_duration_sec > 0:
                # Estimativa grosseira de passos de preenchimento
                # No loop abaixo, iteramos por regiões e depois por blocos
                # Difícil prever total de blocos sem iterar.
                # Vamos assumir que color_skip atual (metade do skip de traço) é uma base razoável,
                # mas idealmente deveríamos contar blocos antes.
                # Para MVP: Manter lógica baseada no skip_rate do traço, ou definir fixo.
                # Vamos tentar adaptar proporcionalmente.
                color_skip = max(1, int(skip_rate * 0.5)) 
            else:
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
                
                # Ordenar blocos por linha e coluna (rápido e natural)
                blocks.sort(key=lambda b: (b[2], b[3]))
                
                # Pintar bloco por bloco com animação
                for block_ys, block_xs, gr_row, gr_col in blocks:
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
    progress=gr.Progress(),
    sketch_duration_sec=None,
    fill_duration_sec=None,
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
                    image_path, split_len, frame_rate, skip_rate, end_duration, draw_mode,
                    sketch_duration_sec, fill_duration_sec
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
    draw_mode="Apenas Contornos",
    sketch_duration_sec=None,
    fill_duration_sec=None,
    hand_style="default"
):
    """
    Versão simplificada da função original para uso em batch processing
    """
    try:
        # Carregar imagem
        img = imread_safe(image_path)
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
            target_ht = img_ht
            target_wd = img_wd
        
        # GARANTIR que dimensões sejam divisíveis pelo split_len
        target_wd = (target_wd // split_len) * split_len
        target_ht = (target_ht // split_len) * split_len
        
        # Garantir dimensões mínimas
        min_dim = split_len * 2
        target_wd = max(target_wd, min_dim)
        target_ht = max(target_ht, min_dim)
        
        # Ajustar para valores pares
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
            img_thresh_inv = cv2.bitwise_not(img_thresh)
            kernel = np.ones((3, 3), np.uint8)
            img_thresh_dilated = cv2.dilate(img_thresh_inv, kernel, iterations=1)
            img_thresh_for_regions = cv2.bitwise_not(img_thresh_dilated)
            
            num_labels, labels = cv2.connectedComponents(img_thresh_for_regions)
            
            region_info = []
            for label_id in range(1, num_labels):
                region_mask = (labels == label_id)
                region_size = np.sum(region_mask)
                
                if region_size < 50:
                    continue
                
                ys, xs = np.where(region_mask)
                if len(ys) == 0:
                    continue
                
                mean_color = np.mean(img[ys, xs], axis=0)
                # Ignorar cores muito claras (fundo), não apenas branco puro
                # Valor reduzido de 245 para 225 para pegar cinzas claros
                # Se todos os canais forem > 225, considera "branco/fundo" e pula
                if np.all(mean_color > 225):
                    continue
                
                region_info.append({
                    'label_id': label_id,
                    'size': region_size,
                    'ys': ys,
                    'xs': xs
                })
            
            region_info.sort(key=lambda r: r['size'])
            
            if fill_duration_sec is not None and fill_duration_sec > 0:
                color_skip = max(1, int(skip_rate * 0.5))
            else:
                color_skip = max(1, skip_rate // 2)

            block_counter = 0
            
            for region in region_info:
                ys, xs = region['ys'], region['xs']
                
                # Agrupar pixels em blocos de grid usando NumPy (vetorizado)
                grid_rows = ys // split_len
                grid_cols = xs // split_len
                grid_keys_arr = grid_rows * 10000 + grid_cols
                unique_keys = np.unique(grid_keys_arr)
                
                blocks = []
                for key in unique_keys:
                    mask = grid_keys_arr == key
                    blocks.append((ys[mask], xs[mask], int(key // 10000), int(key % 10000)))
                
                if len(blocks) == 0:
                    continue
                
                # Ordenar blocos por linha e coluna (rápido e natural)
                blocks.sort(key=lambda b: (b[2], b[3]))
                
                # Pintar bloco por bloco com animação
                for block_ys, block_xs, gr_row, gr_col in blocks:
                    drawn_frame[block_ys, block_xs] = img[block_ys, block_xs]
                    
                    block_counter += 1
                    if block_counter % color_skip == 0:
                        hx = min(gr_col * split_len + split_len // 2, target_wd - 1)
                        hy = min(gr_row * split_len + split_len // 2, target_ht - 1)
                        
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
        img = imread_safe(image) # Use safe version (not part of cv2 module)
        if img is None: 
             return "Erro ao ler imagem."
             
        img_ht, img_wd = img.shape[0], img.shape[1]
        
        info = f"📐 Resolução original: {img_wd} x {img_ht}\\n"
        
        # Simplesmente mostrar info, sem sugerir split_len
        return info
        
    except Exception as e:
        return f"❌ Erro ao processar imagem: {e}"

# Funções de licenciamento para interface
def check_license_status():
    """Verifica status da licença"""
    if license_manager.is_licensed():
        info = license_manager.get_license_info()
        return f"✅ **LICENÇA ATIVADA**\\n\\n📧 Email: {info['email']}\\n🎯 Plano: {info['plan'].upper()}\\n📅 Ativada em: {info['activated_at'][:10]}"
    else:
        return "❌ **LICENÇA NÃO ATIVADA**\\n\\nPor favor, ative sua licença para usar todas as funcionalidades."

def register_action(email, password, password_confirm):
    """Cadastra novo usuário"""
    if password != password_confirm:
        return "❌ As senhas não coincidem."
    
    success, message = license_manager.register_user(email, password)
    return message

def login_action(email, password):
    """Faz login com email e senha"""
    success, message, session_id = license_manager.login_with_password(email, password)
    
    if success:
        info = license_manager.get_license_info()
        success_msg = f"✅ {message}\n\n🎉 **Bem-vindo!**\n\n📧 Email: {info['email']}\n🎯 Plano: {info['plan'].upper()}"
        return success_msg, session_id, gr.update(visible=False), gr.update(visible=True)
    else:
        return message, "", gr.update(visible=True), gr.update(visible=False)

def request_password_reset_action(email):
    """Solicita link de recuperação de senha por email."""
    reset_base_url = os.environ.get("PASSWORD_RESET_BASE_URL", "").strip()
    if not reset_base_url:
        reset_base_url = os.environ.get("APP_BASE_URL", "").strip()
    if not reset_base_url:
        reset_base_url = "http://localhost:7860"

    success, message = license_manager.request_password_reset(email, reset_base_url)
    return message

def reset_password_with_token_action(new_password, confirm_password, request: gr.Request):
    """Redefine senha usando token do link de recuperação (query string)."""
    token = ""
    if request is not None:
        token = (request.query_params.get("token", "") or "").strip()

    if new_password != confirm_password:
        return "❌ As senhas não coincidem."

    if not token:
        return "❌ Link de recuperação inválido. Solicite um novo email de recuperação."

    success, message = license_manager.reset_password_with_token(token, new_password)
    return message

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

def _build_license_bar(lm):
    """Gera HTML da barra de licença ativa para o app"""
    info = lm.get_license_info()
    if not info:
        return ""
    return f"""
    <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <h3 style="color: #155724; margin: 0 0 10px 0;">✅ Licença Ativada</h3>
        <p style="color: #155724; margin: 0;">
            <strong>Email:</strong> {info['email']} | 
            <strong>Plano:</strong> {info['plan'].upper()} | 
            <strong>Ativada em:</strong> {info['activated_at'][:10]}
        </p>
    </div>
    """

# Interface Gradio Comercial
def create_commercial_interface():
    """Cria interface comercial com licenciamento"""
    
    # Verifica se está licenciado
    is_licensed = license_manager.is_licensed()
    
    with gr.Blocks(title="Whiteboard Animation Pro - Commercial") as app:
        
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

        const RESET_TOKEN_STORAGE_KEY = 'whiteboardpro_reset_token';

        function getPasswordResetTokenFromUrl() {
            try {
                const params = new URLSearchParams(window.location.search);
                return params.get('token') || '';
            } catch (e) {
                return '';
            }
        }

        function cachePasswordResetToken() {
            const urlToken = getPasswordResetTokenFromUrl();
            if (urlToken) {
                sessionStorage.setItem(RESET_TOKEN_STORAGE_KEY, urlToken);
            }
            return sessionStorage.getItem(RESET_TOKEN_STORAGE_KEY) || '';
        }

        function clearPasswordResetTokenFromUrl() {
            try {
                const cleanUrl = new URL(window.location.href);
                cleanUrl.searchParams.delete('token');
                window.history.replaceState({}, '', cleanUrl.toString());
            } catch (e) {
                // noop
            }
        }

        function clickTabByText(fragment) {
            const target = (fragment || '').toLowerCase();
            if (!target) return false;

            const tabs = Array.from(document.querySelectorAll('button,[role="tab"]'));
            const found = tabs.find((el) => ((el.textContent || '').toLowerCase().includes(target)));
            if (found) {
                found.click();
                return true;
            }
            return false;
        }

        function hydratePasswordResetFromLink() {
            const token = cachePasswordResetToken();
            if (!token) return;

            const applyTokenToUi = () => {
                // Garante navegação Início -> Entrar -> Recuperar Senha
                clickTabByText('entrar');
                clickTabByText('recuperar senha');

                // Finaliza quando os campos de nova senha estiverem montados
                const passwordInput = document.querySelector('input[placeholder="Mínimo 6 caracteres"]');
                if (passwordInput) {
                    passwordInput.focus();
                    return true;
                }
                return false;
            };

            let attempts = 0;
            const maxAttempts = 25;
            const intervalId = setInterval(() => {
                attempts += 1;
                const applied = applyTokenToUi();
                if (applied || attempts >= maxAttempts) {
                    clearInterval(intervalId);
                }
            }, 300);
        }
        
        // Carregar sessão ao iniciar
        window.addEventListener('load', function() {
            const sessionId = loadSessionFromStorage();
            if (sessionId) {
                console.log('Sessão restaurada do localStorage');
            }

            hydratePasswordResetFromLink();
        });
        </script>
        """)
        
        payment_url = license_manager.payment_link or 'https://buy.stripe.com/test_5kQ28rfZdd2RablaVicQU02'
        
        # ============================================================
        # ÁREA PRÉ-LOGIN: Tabs Landing + Entrar (visível quando NÃO logado)
        # ============================================================
        with gr.Group(visible=not is_licensed) as landing_group:
            
            with gr.Tabs() as pre_login_tabs:
                
                # ==========================================
                # TAB 1: LANDING PAGE (100% foco em vendas)
                # ==========================================
                with gr.TabItem("🏠 Início", id="tab-inicio"):
            
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
                            </div>
                            <p style="margin-top: 15px; font-size: 0.85em; opacity: 0.6;">Pagamento seguro via Stripe. Cancele quando quiser.</p>
                        </div>
                    </div>
                    """)
                    
                    # --- SOCIAL PROOF ---
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 40px;">
                        <div style="display: flex; gap: 30px; justify-content: center; flex-wrap: wrap;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.5em;">⭐⭐⭐⭐⭐</span>
                                <span style="color: #555; font-size: 0.95em; font-weight: 500;">4.9/5 de satisfação</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.3em;">👥</span>
                                <span style="color: #555; font-size: 0.95em; font-weight: 500;">+500 criadores de conteúdo</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.3em;">🎬</span>
                                <span style="color: #555; font-size: 0.95em; font-weight: 500;">+10.000 vídeos gerados</span>
                            </div>
                        </div>
                    </div>
                    """)
                    
                    # --- VÍDEO DEMO (placeholder para YouTube) ---
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 40px;">
                        <h2 style="color: #1a1a2e; font-size: 2em; margin-bottom: 20px;">🎬 Veja o resultado</h2>
                        <div style="max-width: 720px; margin: 0 auto; background: linear-gradient(135deg, #1a1a2e, #0f3460); border-radius: 12px; overflow: hidden; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 15px;">
                            <!-- SUBSTITUIR pelo embed do YouTube quando tiver o link -->
                            <div style="font-size: 4em;">🎥</div>
                            <p style="color: #aaa; font-size: 1.2em; margin: 0;">Vídeo demonstrativo em breve</p>
                            <p style="color: #667eea; font-size: 0.9em; margin: 0;">Imagem → Animação Whiteboard em segundos</p>
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
                            <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">RECOMENDADO</div>
                                <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">🎨 Line Art Colorido</h3>
                                <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços pretos nítidos sobre fundo branco com cores leves e translúcidas. Perfeito para slides educativos.</p>
                                <details style="cursor: pointer;"><summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary><p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">"Crie uma ilustração em line art minimalista técnico para slide de apresentação, estilo esboço profissional clean e didático. Use traços pretos nítidos sobre fundo branco puro 100%. Adicione cor de maneira restrita e elegante: contornos finos de destaque, preenchimento leve/translúcido (opacidade 10-30%). Estilo ultra-clean, técnico, alta legibilidade."</p></details>
                            </div>
                            <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                                <div style="background: #333; color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">CLÁSSICO</div>
                                <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">✒️ Line Art P&B</h3>
                                <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços pretos limpos sobre fundo branco puro. Estilo whiteboard clássico, ideal para ilustrações técnicas.</p>
                                <details style="cursor: pointer;"><summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary><p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">"Crie uma imagem para slide de apresentação. Estilo Line Art minimalista em preto e branco, com traços pretos nítidos e limpos sobre fundo branco puro. Ilustração simplificada. Estilo de esboço técnico profissional, sem sombras complexas ou cores, apenas contornos e hachuras leves para profundidade."</p></details>
                            </div>
                            <div style="flex: 1; min-width: 280px; max-width: 350px; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                                <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 15px;">DIVERTIDO</div>
                                <h3 style="color: #1a1a2e; margin: 0 0 10px 0;">🎭 Cartoon Educativo</h3>
                                <p style="color: #666; font-size: 0.9em; line-height: 1.6; margin-bottom: 15px;">Traços grossos, cores planas e vibrantes. Estilo cartoon moderno, ótimo para conteúdo descontraído.</p>
                                <details style="cursor: pointer;"><summary style="color: #667eea; font-weight: bold; font-size: 0.9em;">📋 Ver prompt para IA</summary><p style="background: #f8f9fa; padding: 12px; border-radius: 8px; font-size: 0.8em; color: #555; margin-top: 10px; line-height: 1.5;">"Crie uma ilustração em estilo cartoon educativo. Traços pretos grossos, uniformes e nítidos. Contornos limpos e fechados, estilo digital clean. Cores planas (flat colors), sem gradientes. Paleta profissional e vibrante. Fundo 100% branco puro para máxima legibilidade em slides."</p></details>
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
                                <p style="color: #666; font-size: 0.9em; margin: 0; line-height: 1.5;">Ideal para <strong>slides, YouTube, aulas</strong> e apresentações.</p>
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
                            <div style="position: absolute; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 8px; font-weight: bold; font-size: 0.9em; letter-spacing: 1px;">🔥 PROMOÇÃO DE LANÇAMENTO - 50% OFF</div>
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
                                <a href="{payment_url}" target="_blank" style="display: block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 50px; text-decoration: none; font-size: 1.2em; font-weight: bold; box-shadow: 0 8px 25px rgba(102,126,234,0.4); text-align: center;">🛒 Assinar Agora com 50% OFF</a>
                                <p style="color: #999; font-size: 0.8em; margin-top: 12px;">Pagamento seguro via Stripe. Cartão ou Pix.</p>
                            </div>
                        </div>
                    </div>
                    """)
                    
                    # --- GARANTIA ---
                    gr.HTML("""
                    <div style="margin-bottom: 40px; text-align: center;">
                        <div style="max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-radius: 16px; padding: 35px 30px; border: 2px solid #4caf50;">
                            <div style="font-size: 3.5em; margin-bottom: 10px;">🛡️</div>
                            <h3 style="color: #2e7d32; font-size: 1.5em; margin: 0 0 10px 0;">Garantia de 7 Dias</h3>
                            <p style="color: #555; font-size: 1.05em; line-height: 1.7; margin: 0;">Se por qualquer motivo você não ficar satisfeito, devolvemos <strong>100% do seu dinheiro</strong> em até 7 dias após a compra. Sem perguntas, sem burocracia.</p>
                        </div>
                    </div>
                    """)
                    
                    # --- FAQ ---
                    gr.HTML("""
                    <div style="margin-bottom: 40px; max-width: 700px; margin-left: auto; margin-right: auto;">
                        <h2 style="text-align: center; color: #1a1a2e; font-size: 2em; margin-bottom: 30px;">❓ Perguntas Frequentes</h2>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Que tipo de imagem funciona melhor?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">Imagens com traços nítidos sobre fundo branco funcionam melhor: line art, ilustrações técnicas, diagramas, slides educativos e cartoons. Você pode gerar essas imagens usando IA (ChatGPT, Gemini, Midjourney) com os prompts que disponibilizamos acima.</p></details>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Posso usar para Stories e Reels?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O app aceita imagens em qualquer dimensão. Para Stories/Reels, use imagens 9:16 (1080x1920). Para YouTube e slides, use 16:9 (1920x1080).</p></details>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Posso processar várias imagens de uma vez?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O modo lote permite processar múltiplas imagens de uma vez e baixar todos os vídeos em um arquivo ZIP.</p></details>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Como funciona o pagamento?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">O pagamento é processado pelo Stripe, a plataforma de pagamentos mais segura do mundo. Aceita cartão de crédito e Pix. A assinatura é anual e você pode cancelar a qualquer momento.</p></details>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">A promoção de 50% é por tempo limitado?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! O preço promocional de R$49,90/ano (50% de desconto) é exclusivo para os primeiros assinantes. O preço normal será R$97,90/ano.</p></details>
                        <details style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); cursor: pointer;"><summary style="font-weight: bold; color: #1a1a2e; font-size: 1.05em;">Tem garantia?</summary><p style="color: #666; margin-top: 12px; line-height: 1.6;">Sim! Oferecemos garantia incondicional de 7 dias. Se não ficar satisfeito, devolvemos 100% do valor pago, sem perguntas.</p></details>
                    </div>
                    """)
                    
                    # --- CTA FINAL ---
                    gr.HTML(f"""
                    <div style="text-align: center; margin-bottom: 40px; background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%); border-radius: 16px; padding: 40px 20px;">
                        <h2 style="color: white; font-size: 1.8em; margin: 0 0 15px 0;">Pronto para criar vídeos incríveis?</h2>
                        <p style="color: rgba(255,255,255,0.7); font-size: 1.1em; margin: 0 0 25px 0;">Junte-se a +500 criadores que já usam o Whiteboard Animation Pro</p>
                        <a href="{payment_url}" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px 40px; border-radius: 50px; text-decoration: none; font-size: 1.15em; font-weight: bold; box-shadow: 0 8px 25px rgba(102,126,234,0.4);">� Assinar Agora com 50% OFF - R$49,90/ano</a>
                        <p style="color: rgba(255,255,255,0.5); font-size: 0.85em; margin-top: 12px;">🛡️ Garantia de 7 dias · Cancele quando quiser</p>
                    </div>
                    """)
                    
                    # --- FOOTER LANDING ---
                    gr.HTML("""
                    <div style="background: #1a1a2e; color: white; border-radius: 12px; padding: 30px; text-align: center;">
                        <h3 style="margin: 0 0 10px 0; opacity: 0.9;">🎨 Whiteboard Animation Pro</h3>
                        <p style="margin: 0; opacity: 0.6; font-size: 0.9em;">&copy; 2026 Ai Infinitus - Todos os direitos reservados</p>
                    </div>
                    """)
                
                # ==========================================
                # TAB 2: ENTRAR (login com email + senha)
                # ==========================================
                with gr.TabItem("🔐 Entrar", id="tab-entrar"):
                    
                    gr.HTML("""
                    <div style="max-width: 500px; margin: 30px auto; text-align: center;">
                        <div style="font-size: 3em; margin-bottom: 15px;">🎨</div>
                        <h2 style="color: #1a1a2e; margin: 0 0 10px 0; font-size: 1.8em;">Área do Assinante</h2>
                        <p style="color: #666; margin: 0 0 10px 0; font-size: 1.05em;">Faça login ou cadastre-se para acessar o app.</p>
                    </div>
                    """)
                    
                    session_id_hidden = gr.Textbox(visible=False)
                    session_storage_bridge = gr.HTML(value="", visible=False)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            pass
                        with gr.Column(scale=2):
                            with gr.Tabs() as auth_tabs:
                                # Sub-tab LOGIN
                                with gr.TabItem("🔑 Login"):
                                    login_email = gr.Textbox(
                                        label="📧 Email",
                                        placeholder="seu@email.com",
                                        info="Use o mesmo email da compra no Stripe"
                                    )
                                    login_password = gr.Textbox(
                                        label="🔒 Senha",
                                        placeholder="Sua senha",
                                        type="password"
                                    )
                                    login_btn = gr.Button(
                                        "� Entrar",
                                        variant="primary",
                                        size="lg"
                                    )
                                    login_result = gr.Markdown(label="Resultado", visible=True)

                                # Sub-tab RECUPERAR SENHA
                                with gr.TabItem("🔁 Recuperar Senha"):
                                    gr.HTML("""
                                    <div style="background: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                                        <p style="color: #8a6d3b; margin: 0; font-size: 0.9em;">
                                            Informe seu email para receber o link de recuperação. Depois, abra o link recebido e defina sua nova senha.
                                        </p>
                                    </div>
                                    """)

                                    reset_email = gr.Textbox(
                                        label="📧 Email da conta",
                                        placeholder="seu@email.com"
                                    )
                                    request_reset_btn = gr.Button(
                                        "📨 Enviar Link de Recuperação",
                                        variant="secondary"
                                    )
                                    request_reset_result = gr.Markdown(label="Status do envio", visible=True)

                                    reset_new_password = gr.Textbox(
                                        label="🔒 Nova senha",
                                        placeholder="Mínimo 6 caracteres",
                                        type="password"
                                    )
                                    reset_new_password_confirm = gr.Textbox(
                                        label="🔒 Confirmar nova senha",
                                        placeholder="Repita a nova senha",
                                        type="password"
                                    )
                                    confirm_reset_btn = gr.Button(
                                        "✅ Redefinir Senha",
                                        variant="primary"
                                    )
                                    confirm_reset_result = gr.Markdown(label="Resultado", visible=True)
                                
                                # Sub-tab CADASTRO
                                with gr.TabItem("📝 Primeiro Acesso"):
                                    gr.HTML("""
                                    <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                                        <p style="color: #0066cc; margin: 0; font-size: 0.9em;">⚡ Já pagou? Cadastre aqui seu email e senha para acessar o app. Use o <strong>mesmo email</strong> da compra no Stripe.</p>
                                    </div>
                                    """)
                                    reg_email = gr.Textbox(
                                        label="� Email usado na compra",
                                        placeholder="seu@email.com",
                                        info="Deve ser o mesmo email do checkout Stripe"
                                    )
                                    reg_password = gr.Textbox(
                                        label="🔒 Criar senha",
                                        placeholder="Mínimo 6 caracteres",
                                        type="password"
                                    )
                                    reg_password_confirm = gr.Textbox(
                                        label="🔒 Confirmar senha",
                                        placeholder="Repita a senha",
                                        type="password"
                                    )
                                    reg_btn = gr.Button(
                                        "📝 Cadastrar",
                                        variant="primary",
                                        size="lg"
                                    )
                                    reg_result = gr.Markdown(label="Resultado", visible=True)
                        with gr.Column(scale=1):
                            pass
                    
                    gr.HTML(f"""
                    <div style="max-width: 500px; margin: 30px auto; text-align: center;">
                        <div style="background: #f8f9ff; border-radius: 12px; padding: 25px; border: 1px solid #e8eaf6;">
                            <h4 style="color: #495057; margin: 0 0 12px 0;">📦 Ainda não é assinante?</h4>
                            <p style="color: #666; margin: 0 0 15px 0; font-size: 0.95em;">Assine agora e comece a criar vídeos whiteboard incríveis!</p>
                            <a href="{payment_url}" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 28px; border-radius: 50px; text-decoration: none; font-size: 1em; font-weight: bold; box-shadow: 0 4px 15px rgba(102,126,234,0.3);">🛒 Assinar com 50% OFF - R$49,90/ano</a>
                        </div>
                    </div>
                    """)
        
        # ============================================================
        # ÁREA PÓS-LOGIN: App Completo (visível após login)
        # ============================================================
        with gr.Group(visible=is_licensed) as app_group:
            
            # Barra superior com info da licença + logout (SEMPRE disponível)
            with gr.Row():
                with gr.Column(scale=9):
                    license_status_html = gr.HTML(
                        value=_build_license_bar(license_manager) if is_licensed else ""
                    )
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
                                ui_preset = gr.Dropdown(
                                    choices=[p.name for p in Quality],
                                    value="HD",
                                    label="Qualidade / Preset"
                                )
                                
                                ui_fps = gr.Radio(
                                    choices=[30, 60],
                                    value=30,
                                    label="FPS (Quadros por segundo)"
                                )
                                
                                ui_sketch_duration = gr.Slider(
                                    minimum=1, maximum=60, value=12, step=1,
                                    label="Duração do Traço (segundos)"
                                )
                                
                                ui_fill_duration = gr.Slider(
                                    minimum=0, maximum=30, value=6, step=0.5,
                                    label="Duração do Preenchimento (segundos)"
                                )
                                
                                ui_end_duration = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="Duração da Imagem Final (segundos)"
                                )
                                
                                ui_hand_style = gr.Dropdown(
                                    choices=[h.value for h in HandStyle],
                                    value="default",
                                    label="Estilo da Mão"
                                )
                                
                                draw_mode = gr.Radio(
                                    choices=["Apenas Contornos", "Contornos + Colorização"],
                                    value="Apenas Contornos",
                                    label="🎨 Modo de Desenho",
                                    info="'Apenas Contornos' = whiteboard clássico. 'Colorização' = preenche com cores."
                                )
                                
                                # Inputs ocultos para compatibilidade
                                hidden_split_len = gr.Number(value=10, visible=False)
                                hidden_skip_rate = gr.Number(value=5, visible=False)
                            
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
                                batch_ui_preset = gr.Dropdown(
                                    choices=[p.name for p in Quality],
                                    value="HD",
                                    label="Qualidade / Preset"
                                )
                                
                                batch_ui_fps = gr.Radio(
                                    choices=[30, 60],
                                    value=30,
                                    label="FPS (Quadros por segundo)"
                                )
                                
                                batch_ui_sketch_duration = gr.Slider(
                                    minimum=1, maximum=60, value=12, step=1,
                                    label="Duração do Traço (segundos)"
                                )
                                
                                batch_ui_fill_duration = gr.Slider(
                                    minimum=0, maximum=30, value=6, step=0.5,
                                    label="Duração do Preenchimento (segundos)"
                                )
                                
                                batch_ui_end_duration = gr.Slider(
                                    minimum=1, maximum=10, value=3, step=1,
                                    label="Duração da Imagem Final (segundos)"
                                )
                                
                                batch_draw_mode = gr.Radio(
                                    choices=["Apenas Contornos", "Contornos + Colorização"],
                                    value="Apenas Contornos",
                                    label="🎨 Modo de Desenho",
                                    info="'Apenas Contornos' = whiteboard clássico. 'Colorização' = preenche com cores."
                                )
                                
                                # Inputs ocultos batch
                                batch_hidden_split_len = gr.Number(value=10, visible=False)
                                batch_hidden_skip_rate = gr.Number(value=5, visible=False)
                            
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
                    Versão Comercial &copy; 2026 Ai Infinitus - Todos os direitos reservados
                </p>
            </div>
            """)
        
        # Eventos de autenticação - Login com email + senha
        
        # CADASTRO
        reg_btn.click(
            fn=register_action,
            inputs=[reg_email, reg_password, reg_password_confirm],
            outputs=[reg_result]
        )
        
        # LOGIN
        def login_and_save_session(email, password):
            """Faz login e salva session_id no localStorage"""
            result_msg, session_id, landing_vis, app_vis = login_action(email, password)
            
            license_bar_html = ""
            bridge_html = ""
            if session_id:
                license_bar_html = _build_license_bar(license_manager)
                bridge_html = f"""
                <script>
                (function() {{
                    localStorage.setItem('whiteboardpro_session_id', '{session_id}');
                }})();
                </script>
                """
            
            return result_msg, session_id, landing_vis, app_vis, license_bar_html, bridge_html

        login_event = login_btn.click(
            fn=login_and_save_session,
            inputs=[login_email, login_password],
            outputs=[login_result, session_id_hidden, landing_group, app_group, license_status_html, session_storage_bridge]
        )

        # RECUPERAÇÃO DE SENHA - solicitar email
        request_reset_btn.click(
            fn=request_password_reset_action,
            inputs=[reset_email],
            outputs=[request_reset_result]
        )

        # RECUPERAÇÃO DE SENHA - confirmar nova senha
        confirm_reset_btn.click(
            fn=reset_password_with_token_action,
            inputs=[reset_new_password, reset_new_password_confirm],
            outputs=[confirm_reset_result]
        )
        
        # Evento de logout (SEMPRE disponível, não depende de is_licensed no boot)
        def logout_and_clear_storage():
            """Faz logout e limpa session_id do localStorage"""
            activation_vis, app_vis, _ = logout_action()
            bridge_html = """
            <script>
            (function() {
                localStorage.removeItem('whiteboardpro_session_id');
            })();
            </script>
            """

            return activation_vis, app_vis, "", bridge_html

        logout_event = logout_btn.click(
            fn=logout_and_clear_storage,
            outputs=[landing_group, app_group, session_id_hidden, session_storage_bridge]
        )

        def restore_session_ui(session_id_stored):
            """Restaura UI a partir de session_id salvo no localStorage."""
            restored_session_id = restore_session_from_storage(session_id_stored)
            if restored_session_id:
                return restored_session_id, gr.update(visible=False), gr.update(visible=True), _build_license_bar(license_manager)

            return "", gr.update(visible=True), gr.update(visible=False), ""

        app.load(
            fn=restore_session_ui,
            inputs=[session_id_hidden],
            outputs=[session_id_hidden, landing_group, app_group, license_status_html],
            js="""
            () => {
                return [localStorage.getItem('whiteboardpro_session_id') || ''];
            }
            """
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
        
        def process_batch_images(files, split_len, frame_rate, skip_rate, end_duration, draw_mode, sketch_duration, fill_duration, progress=gr.Progress()):
            if files is None or len(files) == 0:
                return None, "❌ Nenhuma imagem selecionada", "Nenhuma imagem para processar"
            
            # Extrair caminhos dos arquivos (compatível com Gradio 5 e 6)
            image_paths = [file if isinstance(file, str) else file.name for file in files]
            
            # Processar em lote
            zip_path, message = generate_sketch_video_batch(
                image_paths, split_len, frame_rate, skip_rate, end_duration, draw_mode, progress,
                sketch_duration, fill_duration
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
        # Eventos - Processamento Individual
        image_input.change(
            fn=get_image_info,
            inputs=[image_input],
            outputs=[image_info]
        )
        
        # Evento principal atualizado
        generate_btn.click(
            fn=generate_sketch_video,
            # Ordem: image_path, split_len, frame_rate, skip_rate, end_duration, draw_mode, progress, sketch_duration, fill_duration
            inputs=[
                image_input, 
                hidden_split_len, 
                ui_fps, 
                hidden_skip_rate, 
                ui_end_duration, 
                draw_mode,
                # Novos argumentos
                ui_sketch_duration,
                ui_fill_duration
            ],
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
            inputs=[
                batch_images, 
                batch_hidden_split_len, 
                batch_ui_fps, 
                batch_hidden_skip_rate, 
                batch_ui_end_duration, 
                batch_draw_mode,
                # Novos
                batch_ui_sketch_duration,
                batch_ui_fill_duration
            ],
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
        inbrowser=True,
        theme=gr.themes.Soft()
    )
