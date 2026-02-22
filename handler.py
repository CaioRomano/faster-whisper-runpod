import runpod
import io
import base64
from faster_whisper import WhisperModel
import os


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a cached model.

    Args:
        model_id: The model name from Hugging Face (e.g., 'microsoft/Phi-3-mini-4k-instruct')

    Returns:
        The full path to the cached model snapshot
    """
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' is not in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_DIR, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    print(f"[ModelStore] MODEL_ID: {model_id}")
    print(f"[ModelStore] Model root: {model_root}")

    # Try to read the snapshot hash from refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            print(f"[ModelStore] Using snapshot from refs/main: {candidate}")
            return candidate

    # Fall back to first available snapshot
    if not os.path.isdir(snapshots_dir):
        raise RuntimeError(f"[ModelStore] snapshots directory not found: {snapshots_dir}")

    versions = [
        d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))
    ]

    if not versions:
        raise RuntimeError(f"[ModelStore] No snapshot subdirectories found under {snapshots_dir}")

    versions.sort()
    chosen = os.path.join(snapshots_dir, versions[0])
    print(f"[ModelStore] Using first available snapshot: {chosen}")
    return chosen

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


HF_CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
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

MODEL_ID = f'Systran/faster-whisper-{MODEL_NAME}'
LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)
print(f"[ModelStore] Resolved local model path: {LOCAL_MODEL_PATH}")


model = WhisperModel(
    LOCAL_MODEL_PATH,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    local_files_only=True
)


if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        # 'concurrency_modifier': 2
    })
