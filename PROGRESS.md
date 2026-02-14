# Progresso de Desenvolvimento

## 2026-02-14

### Etapa concluída: Backend de recuperação de senha (SMTP + token)

Arquivo alterado:
- `app_licensed.py`

Mudanças realizadas:
1. Imports adicionados para SMTP e token seguro:
   - `smtplib`
   - `secrets`
   - `MIMEText`
2. Nova tabela SQLite criada em `init_db()`:
   - `password_reset_tokens`
3. Novos métodos adicionados em `LicenseManager`:
   - `_send_email_smtp(recipient_email, subject, body_text)`
   - `request_password_reset(email, reset_base_url)`
   - `verify_password_reset_token(token)`
   - `reset_password_with_token(token, new_password)`

Regras de segurança aplicadas:
- Não vaza se o email existe no sistema (mensagem neutra).
- Token único com expiração de 1 hora.
- Tokens antigos do mesmo email são invalidados.
- Token é marcado como usado após reset de senha.
- Credenciais SMTP via variáveis de ambiente (sem hardcode no código).

Pendências da próxima etapa:
- Adicionar UI no Gradio para "Esqueci minha senha".
- Conectar botões/eventos ao backend de reset.
- Validar fluxo ponta a ponta.

### Etapa concluída: UI de recuperação de senha integrada no Gradio

Arquivo alterado:
- `app_licensed.py`

Mudanças realizadas:
1. Novas actions para recuperação de senha:
   - `request_password_reset_action(email)`
   - `reset_password_with_token_action(token, new_password, confirm_password)`
2. Nova sub-aba de autenticação:
   - `🔁 Recuperar Senha`
3. Novos componentes de interface:
   - email para solicitar link
   - token de recuperação
   - nova senha + confirmação
   - botões de enviar link e redefinir senha
4. Eventos conectados no fluxo Gradio:
   - `request_reset_btn.click(...)`
   - `confirm_reset_btn.click(...)`

Pendências da próxima etapa:
- Adicionar rate limiting no login/recuperação.
- Validar fluxo ponta a ponta com SMTP real no ambiente.

### Etapa concluída: Rate limiting básico no backend

Arquivo alterado:
- `app_licensed.py`

Mudanças realizadas:
1. Nova tabela SQLite para controle de tentativas:
   - `rate_limit`
2. Novos métodos internos no `LicenseManager`:
   - `_check_rate_limit(identifier, action, max_attempts, window_minutes)`
   - `_clear_rate_limit(identifier, action)`
3. Proteção aplicada no login:
   - bloqueia após 5 tentativas em 15 minutos
   - limpa contador após login bem-sucedido
4. Proteção aplicada na recuperação de senha:
   - bloqueia após 3 solicitações em 15 minutos
   - limpa contador quando o email de recuperação é enviado com sucesso

Pendência atual:
- Validar fluxo ponta a ponta com SMTP real no ambiente.

### Etapa concluída: Validação SMTP real (envio de email)

Validação executada em ambiente local com `venv` ativo.

Resultado:
- `SMTP_OK=True`
- Mensagem: `✅ Email enviado.`

Observação:
- O envio SMTP está funcional com Gmail (`smtp.gmail.com:587`, STARTTLS).
- Ainda falta validar o fluxo completo com token (solicitar reset + redefinir senha via UI) com um usuário real cadastrado.

### Etapa concluída: Validação E2E do fluxo de recuperação

Validação executada com `venv` ativo e SMTP configurado por variáveis de ambiente.

Fluxo testado:
1. Usuário de teste preparado/atualizado no SQLite.
2. Solicitação de recuperação por email (`request_password_reset`).
3. Token gerado e recuperado do banco.
4. Redefinição de senha com token (`reset_password_with_token`).
5. Login com nova senha (`login_with_password`).

Resultado:
- `REQUEST_OK=True`
- `TOKEN_OK=True`
- `RESET_OK=True`
- `LOGIN_OK=True`

Status final:
- Fluxo backend de recuperação de senha validado ponta a ponta com sucesso.

### Etapa concluída: Correção de hidratação do token no fluxo com landing

Arquivo alterado:
- `app_licensed.py`

Problema observado:
- Ao abrir o link de recuperação (`?token=...`), o app iniciava na landing (`Início`) e o token não era aplicado automaticamente.

Correção aplicada:
1. Persistência temporária do token em `sessionStorage`.
2. Navegação automática para `Entrar` -> `Recuperar Senha`.
3. Retry para preencher o token somente quando o input existir no DOM do Gradio.
4. Limpeza do token da URL apenas após preenchimento bem-sucedido.

Resultado esperado:
- Clicar no link recebido por email abre o app e já prepara a interface para redefinição de senha, com token preenchido automaticamente.

### Etapa concluída: Mudança de UX para reset por link direto (sem token manual)

Arquivo alterado:
- `app_licensed.py`

Mudanças aplicadas:
1. Removido campo manual de token da aba `Recuperar Senha`.
2. `reset_password_with_token_action` agora lê o token diretamente da query string via `gr.Request`.
3. Mensagem da interface ajustada para orientar abertura do link recebido por email.
4. Botão `Redefinir Senha` passa apenas `nova senha` + `confirmar senha` para o backend.
5. Script de hidratação mantém foco em abrir automaticamente `Entrar -> Recuperar Senha` quando há token na URL.

Resultado esperado:
- Usuário não precisa copiar/colar token.
- Fluxo de redefinição ocorre diretamente pelo link de recuperação.

### Etapa concluída: Persistência de sessão após refresh + limpeza de script visível na UI

Arquivos alterados:
- `app_licensed.py`

Problemas observados:
1. Um bloco `<script>` aparecia como texto na interface após login.
2. Ao atualizar a página, o usuário era deslogado e precisava autenticar novamente.

Correções aplicadas:
1. Removida concatenação de `<script>` no `login_result`.
2. Persistência de `session_id` em `localStorage` movida para callback JS (`login_event.then`).
3. Restauração automática de sessão no `app.load`, validando `session_id` no backend e ajustando visibilidade de `landing_group`/`app_group`.
4. Limpeza de `localStorage` no logout via callback JS (`logout_event.then`).

Resultado esperado:
- Não exibe mais código JS na tela.
- Sessão permanece ativa após atualizar a página, até o usuário clicar em `Sair`.
