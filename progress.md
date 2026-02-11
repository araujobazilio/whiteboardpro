# Progresso - Human Speedpaint

## 2026-02-11
- Iniciado processo de refatoracao do desenho para o algoritmo Human Speedpaint no arquivo app_licensed.py (base antiga).
- Planejado: inserir helpers (connected components, easing, stroke grouping) e atualizar generate_sketch_video / generate_sketch_video_single mantendo licenciamento e UI.
- Helpers do Human Speedpaint adicionadas (ease_in_out_quad, resize_smart, group_into_strokes, etc.).
- generate_sketch_video refatorada para desenho por objetos (Connected Components), easing natural e stroke grouping.
- generate_sketch_video_single refatorada com o mesmo algoritmo Human Speedpaint.
- Validação manual de consistência: fluxo principal, paths globais, conversão H264 e integração com UI preservados.
