# Usa a imagem base do RunPod que você já escolheu
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Instala o uv copiando o binário oficial (muito mais rápido que pip install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configura o uv para instalar pacotes no Python do sistema e não gerar cache na imagem final
ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copia apenas o requirements.txt primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências sem criar cache interno (reduz o tamanho da imagem)
RUN uv pip install --no-cache -r requirements.txt

# Copia o restante dos arquivos (handler.py, etc)
COPY . .

# Comando de inicialização direta para o menor cold start possível
CMD ["python", "-u", "handler.py"]
