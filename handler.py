import runpod
import io
import base64
from faster_whisper import WhisperModel
import os


def handler(event):
    input = event['input']

    language = os.getenv('LANGUAGE', 'pt')
    beam_size = os.getenv('BEAM_SIZE', 5)

    speaker_name = input.get('speaker_name')
    speaker_id = input.get('speaker_id')
    audio_base64 = base64.b64decode(input.get('audio_base64'))
    buffer_audio = io.BytesIO(audio_base64)
    timestamp = input.get('timestamp')

    cache_root = "/runpod-volume/huggingface-cache/hub"

    if os.path.exists(cache_root):
        print(f"Cache root exists: {cache_root}")
        for item in os.listdir(cache_root):
            print(f"  {item}")
    else:
        print(f"Cache root does NOT exist: {cache_root}")

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


required_vars = ["MODEL_NAME", "MODEL_DEVICE", "COMPUTE_TYPE", "LANGUAGE", "BEAM_SIZE"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


MODEL_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

AVAILABLE_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
}

MODEL_NAME = os.getenv('MODEL_NAME', 'turbo')

if MODEL_NAME not in AVAILABLE_MODELS:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

DEVICE = os.getenv('MODEL_DEVICE', 'gpu')
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', 'int8_float16')

model = WhisperModel(
    'turbo',
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    local_files_only=True,
    cpu_threads=os.cpu_count()
    download_root=MODEL_CACHE_DIR
)


if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        # 'concurrency_modifier': 2
    })
