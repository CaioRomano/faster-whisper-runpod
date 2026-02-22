# Usa a imagem base do RunPod que você já escolheu
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# 2. Copia o binário do 'uv' (camada muito leve, executada rapidamente)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python-is-python3 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia apenas o requirements.txt primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências sem criar cache interno (reduz o tamanho da imagem)
RUN uv pip install -r requirements.txt

# Copia o restante dos arquivos (handler.py, etc)
COPY . .

# Comando de inicialização direta para o menor cold start possível
CMD ["python", "-u", "handler.py"]
