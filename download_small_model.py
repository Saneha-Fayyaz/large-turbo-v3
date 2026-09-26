"""
Run this ONCE, with internet access, to download the faster-whisper
'small' model to a local folder. After this finishes, you can go
offline and realtime_stt.py will still work, because it loads the
model with local_files_only=True from the folder created here.
"""

from huggingface_hub import snapshot_download

# Official faster-whisper (CTranslate2) conversion of Whisper "small"
REPO_ID = "Systran/faster-whisper-small"
LOCAL_DIR = "./models/faster-whisper-small"

if __name__ == "__main__":
    print(f"Downloading '{REPO_ID}' to '{LOCAL_DIR}' ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,   # real files, not symlinks into the HF cache
    )
    print(f"Done. Model files saved at: {path}")
    print("You can now run realtime_stt.py fully offline.")