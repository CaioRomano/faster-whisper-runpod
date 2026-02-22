import runpod
import io
import base64
from faster_whisper import WhisperModel
import os


# --- 1. CONFIGURAÇÕES GLOBAIS E VARIÁVEIS DE AMBIENTE ---
MODEL_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

AVAILABLE_MODELS = {
    "tiny", "base", "small", "medium",
    "large-v1", "large-v2", "large-v3", "turbo",
}

# Definições do Modelo
MODEL_NAME = os.getenv('MODEL_NAME', 'turbo')
if MODEL_NAME not in AVAILABLE_MODELS:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

# CTranslate2 aceita 'cuda', 'cpu' ou 'auto' (não 'gpu')
DEVICE = os.getenv('MODEL_DEVICE', 'cuda') 
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', 'int8_float16') 

# Variáveis padrão de inferência 
LANGUAGE = os.getenv('LANGUAGE', 'pt')
BEAM_SIZE = int(os.getenv('BEAM_SIZE', 5))

# --- 2. INICIALIZAÇÃO DO MODELO (Cold Start) ---
model = WhisperModel(
    MODEL_NAME, # Corrigido: usando a variável dinamicamente
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    local_files_only=True,
    cpu_threads=os.cpu_count(), # Corrigido: vírgula adicionada
    download_root=MODEL_CACHE_DIR
)


def handler(event):
    try:
        input = event['input']
    
        language = LANGUAGE
        beam_size = BEAM_SIZE
    
        speaker_name = input.get('speaker_name')
        speaker_id = input.get('speaker_id')
        audio_base64 = base64.b64decode(input.get('audio_base64'))
    
        if not audio_base64:
                return {"error": "O campo 'audio_base64' é obrigatório no input."}
            
        buffer_audio = io.BytesIO(audio_base64)
        timestamp = input.get('timestamp')
    
        segments, _ = model.transcribe(buffer_audio, language=language, multilingual=True, beam_size=beam_size)
        segments = list(segments)
    
        text_transcribed = ' '.join([segment.text for segment in segments])
    
        response = {
            'speaker_id': speaker_id,
            'speaker_name': speaker_name,
            'timestamp': timestamp,
            'text_transcribed': text_transcribed
        }

        return response
        
    except Exception as e:
        # Evita que a worker "morra" ao retornar uma string limpa de erro
        return {"error": str(e)}


if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        # 'concurrency_modifier': 2
    })
