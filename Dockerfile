# Usa a imagem base do RunPod que você já escolheu
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Faz o link do python3 para python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

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
