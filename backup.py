import sys
import time
import queue
import threading
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# ==================== CONFIGURATION ====================
LOCAL_MODEL_PATH = "./models/whisper-large-v3-turbo-urdu-ct2"

# Audio Recording Settings
SAMPLE_RATE = 16000          # Whisper expects 16kHz audio
CHUNK_SIZE = 1024            # Mic buffer frame size

# Pause/Silence Detection Settings
SILENCE_THRESHOLD = 0.01     # Amplitude threshold to detect silence (adjust if mic is sensitive)
PAUSE_DURATION_SEC = 4.0     # Pause duration (in seconds) to trigger transcription
MIN_AUDIO_DURATION_SEC = 0.8 # Ignore accidental noises shorter than this

# Hardware Engine Settings
DEVICE = "cpu"
COMPUTE_TYPE = "int8"         # "int8" for CPU, "float16" for GPU
CPU_THREADS = 4              # Adjust based on your laptop CPU cores

# ==================== AUDIO STREAM HANDLER ====================
class RealtimeSTT:
    def __init__(self, model_path: str, device: str, compute_type: str):
        print(f"Loading local model from '{model_path}' on {device.upper()} ({compute_type})...")
        
        # Load local Faster-Whisper model
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=CPU_THREADS,
            local_files_only=True
        )
        print("Model loaded successfully!")

        self.audio_queue = queue.Queue()
        self.is_running = False
        
        # Speech Accumulation Buffer
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.last_speech_time = time.time()
        self.is_speaking = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback to receive audio data from PyAudio stream."""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)

    def start_listening(self):
        """Starts real-time microphone capture and processing loop."""
        self.is_running = True
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback
        )

        print("\n" + "="*50)
        print(f"Listening... Speak in Urdu. Transcriptions trigger after a {PAUSE_DURATION_SEC}s pause.")
        print("Press CTRL+C to stop.")
        print("="*50 + "\n")

        self.stream.start_stream()

        # Processing Thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_listening()

    def _process_audio(self):
        """Monitors audio for pauses and triggers Whisper on completed utterances."""
        while self.is_running:
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                self.audio_buffer = np.append(self.audio_buffer, chunk)

                # Measure energy level (RMS) of the incoming audio chunk
                rms = np.sqrt(np.mean(chunk**2))

                if rms > SILENCE_THRESHOLD:
                    self.last_speech_time = time.time()
                    if not self.is_speaking:
                        self.is_speaking = True
                        print("\n[Listening...]", end="", flush=True)

            # Check if silence duration exceeded threshold
            silence_duration = time.time() - self.last_speech_time

            if self.is_speaking and silence_duration >= PAUSE_DURATION_SEC:
                # Transcribe accumulated audio buffer
                total_duration = len(self.audio_buffer) / SAMPLE_RATE
                
                if total_duration >= MIN_AUDIO_DURATION_SEC:
                    print(f"\r[Transcribing phrase...]", end="", flush=True)
                    
                    segments, _ = self.model.transcribe(
                        self.audio_buffer,
                        language="ur",
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300)
                    )

                    text_segments = [s.text.strip() for s in segments if s.text.strip()]
                    full_text = " ".join(text_segments)

                    if full_text:
                        print(f"\r[Urdu]: {full_text}")
                    else:
                        print("\r", end="", flush=True)

                # Reset state for the next sentence
                self.audio_buffer = np.zeros(0, dtype=np.float32)
                self.is_speaking = False

            time.sleep(0.05)

    def stop_listening(self):
        """Gracefully halts input stream."""
        print("\nStopping listener...")
        self.is_running = False
        time.sleep(0.2)

        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, "p"):
            self.p.terminate()

        print("Stopped successfully.")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    stt_system = RealtimeSTT(
        model_path=LOCAL_MODEL_PATH,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    stt_system.start_listening()