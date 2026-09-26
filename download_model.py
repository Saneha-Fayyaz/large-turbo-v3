import os
from huggingface_hub import snapshot_download

# Local target directory
LOCAL_MODEL_DIR = "./models/whisper-large-v3-turbo-urdu-ct2"

print("Downloading Urdu Fine-Tuned Whisper Large V3 Turbo (CT2 Format)...")
import os
from huggingface_hub import snapshot_download

LOCAL_MODEL_DIR = "./models/whisper-large-v3-turbo-urdu-ct2"

print(
    "Downloading Urdu Fine-Tuned Whisper Large V3 Turbo (CT2 Format)..."
)

# Download all required CTranslate2 files including model.bin
snapshot_download(
    repo_id="kingabzpro/whisper-large-v3-urdu-ct2",
    local_dir=LOCAL_MODEL_DIR,
    # Do NOT ignore *.bin because model.bin is required by CTranslate2
)

print(
    f"\nModel successfully saved locally at: {os.path.abspath(LOCAL_MODEL_DIR)}"
)
# Pull pre-converted CTranslate2 weights
snapshot_download(
    repo_id="kingabzpro/whisper-large-v3-urdu-ct2",
    local_dir=LOCAL_MODEL_DIR,
    ignore_patterns=["*.pt", "*.bin"], # Download safetensors / CT2 bin files only
)

print(f"\nModel successfully saved locally at: {os.path.abspath(LOCAL_MODEL_DIR)}")
print("You can now run inference 100% offline without internet connections.")