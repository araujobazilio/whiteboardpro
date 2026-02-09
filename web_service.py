"""
Web Service para Deploy no Render
Serve o Gradio diretamente na porta configurada
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app_licensed

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    
    print("=" * 60)
    print("🎨 WHITEBOARD ANIMATION PRO - PRODUÇÃO")
    print("=" * 60)
    print(f"🌐 Porta: {port}")
    print("🚀 Iniciando servidor...")
    
    # Cria a interface Gradio
    gradio_app = app_licensed.create_commercial_interface()
    
    # Lança o Gradio diretamente (funciona no Render)
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
