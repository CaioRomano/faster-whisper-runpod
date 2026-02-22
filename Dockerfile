FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

RUN pip install uv

RUN uv pip install --no-cache -r requirements.txt

COPY handler.py .

CMD ["uv", "run", "handler.py"]
