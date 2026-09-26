import time
import queue
import threading
import collections

import numpy as np
import pyaudio
import webrtcvad                      # pip install webrtcvad-wheels
from faster_whisper import WhisperModel

# ==================== CONFIGURATION ====================
LOCAL_MODEL_PATH = "./models/whisper-large-v3-turbo-urdu-ct2"

# Audio (webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz)
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000      # 480 samples

# --- Noise / speech detection ---
VAD_AGGRESSIVENESS = 3        # 0-3. Higher = stricter about what counts as speech
NOISE_CALIBRATION_SEC = 1.5   # stay quiet for this long at startup
NOISE_GATE_FACTOR = 3.0       # a frame must be this many times louder than room noise
MIN_GATE_RMS = 0.005          # absolute minimum gate (raise if still too sensitive)
START_VOICED_FRAMES = 6       # need 6 voiced frames...
START_WINDOW_FRAMES = 10      # ...out of the last 10 (~180 ms of 300 ms) to start
MIN_SPEECH_SEC = 0.5          # discard utterances with less speech than this

# --- Pause / segmentation ---
PAUSE_DURATION_SEC = 1.0      # silence that ends a phrase (was 3.0)
MAX_UTTERANCE_SEC = 20.0      # force a cut so latency never grows unbounded
PRE_ROLL_SEC = 0.5            # audio kept from before speech was detected
TAIL_KEEP_SEC = 0.2           # trailing silence kept when sending to Whisper

# --- Hardware ---
DEVICE = "cpu"                # "cuda" if you have an NVIDIA GPU (biggest speedup)
COMPUTE_TYPE = "int8"         # "float16" for GPU
CPU_THREADS = 4               # set to your number of PHYSICAL cores


# ==================== STT ENGINE ====================
class RealtimeSTT:
    def __init__(self, model_path: str, device: str, compute_type: str):
        print(f"Loading local model from '{model_path}' on {device.upper()} ({compute_type})...")
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=CPU_THREADS,
            local_files_only=True,
        )
        self._warmup()
        print("Model loaded successfully!")

        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_queue = queue.Queue()   # mic frames -> detector
        self.job_queue = queue.Queue()     # finished phrases -> Whisper worker
        self.noise_floor = 0.003
        self.is_running = False

    # ---------- helpers ----------
    def _warmup(self):
        """First inference is always slow (lazy init); pay that cost at startup."""
        print("Warming up model...")
        segs, _ = self.model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32), language="ur", beam_size=1
        )
        list(segs)

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)))

    def _gate(self) -> float:
        return max(self.noise_floor * NOISE_GATE_FACTOR, MIN_GATE_RMS)

    def _classify(self, frame: np.ndarray):
        """Returns (is_speech, rms). Must pass BOTH the loudness gate and the VAD."""
        if len(frame) != FRAME_SAMPLES:
            return False, 0.0
        rms = self._rms(frame)
        if rms < self._gate():
            return False, rms
        return self.vad.is_speech(frame.tobytes(), SAMPLE_RATE), rms

    def _calibrate_noise(self):
        print(f"Calibrating background noise for {NOISE_CALIBRATION_SEC}s... stay quiet.")
        levels = []
        end = time.perf_counter() + NOISE_CALIBRATION_SEC
        while time.perf_counter() < end:
            try:
                levels.append(self._rms(self.frame_queue.get(timeout=0.1)))
            except queue.Empty:
                pass
        if levels:
            self.noise_floor = float(np.median(levels))
        print(f"Noise floor: {self.noise_floor:.4f} | speech gate: {self._gate():.4f}")

    # ---------- audio input ----------
    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.frame_queue.put(np.frombuffer(in_data, dtype=np.int16))
        return (None, pyaudio.paContinue)

    def start_listening(self):
        self.is_running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()

        self._calibrate_noise()

        print("\n" + "=" * 50)
        print(f"Listening... Speak in Urdu. A {PAUSE_DURATION_SEC}s pause triggers transcription.")
        print("Press CTRL+C to stop.")
        print("=" * 50 + "\n")

        threading.Thread(target=self._detect_speech, daemon=True).start()
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_listening()

    # ---------- speech detection (never blocks on Whisper) ----------
    def _detect_speech(self):
        pre_roll = collections.deque(maxlen=int(PRE_ROLL_SEC * 1000 / FRAME_MS))
        recent = collections.deque(maxlen=START_WINDOW_FRAMES)
        pause_frames_needed = int(PAUSE_DURATION_SEC * 1000 / FRAME_MS)
        max_frames = int(MAX_UTTERANCE_SEC * 1000 / FRAME_MS)
        tail_keep = int(TAIL_KEEP_SEC * 1000 / FRAME_MS)

        speaking = False
        frames = []
        silence_frames = 0
        voiced_frames = 0
        last_voice_time = time.perf_counter()

        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speech, rms = self._classify(frame)

            if not speaking:
                pre_roll.append(frame)
                recent.append(is_speech)

                # Slowly adapt to changing room noise while idle
                if not is_speech and rms < self._gate():
                    self.noise_floor = 0.99 * self.noise_floor + 0.01 * rms

                # Debounce: a click or door slam won't pass this
                if sum(recent) >= START_VOICED_FRAMES:
                    speaking = True
                    frames = list(pre_roll)
                    voiced_frames = sum(recent)
                    silence_frames = 0
                    last_voice_time = time.perf_counter()
                    recent.clear()
                    print("[Listening...]", flush=True)
                continue

            # ---- currently inside a phrase ----
            frames.append(frame)
            if is_speech:
                voiced_frames += 1
                silence_frames = 0
                last_voice_time = time.perf_counter()
            else:
                silence_frames += 1

            if silence_frames >= pause_frames_needed or len(frames) >= max_frames:
                # Trim most of the trailing silence before sending to Whisper
                trim = max(silence_frames - tail_keep, 0)
                if trim:
                    frames = frames[:-trim]

                if voiced_frames * FRAME_MS / 1000 >= MIN_SPEECH_SEC:
                    audio = np.concatenate(frames).astype(np.float32) / 32768.0
                    self.job_queue.put((audio, last_voice_time))
                    print("[Transcribing...]", flush=True)
                else:
                    print("[Ignored short noise]", flush=True)

                speaking = False
                frames = []
                pre_roll.clear()
                recent.clear()

    # ---------- Whisper worker (runs in its own thread) ----------
    def _transcribe_worker(self):
        while self.is_running:
            try:
                audio, last_voice_time = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            segments, _ = self.model.transcribe(
                audio,
                language="ur",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                without_timestamps=True,          # fewer decoder tokens = faster
                condition_on_previous_text=False, # avoids repetition/hallucination loops
                vad_filter=False,                 # we already did VAD ourselves
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            text = " ".join(
                s.text.strip()
                for s in segments
                if s.text.strip() and s.no_speech_prob < 0.6
            )
            t1 = time.perf_counter()

            if text:
                print(f"[Urdu]: {text}")
                print(
                    f" └─ Audio: {len(audio) / SAMPLE_RATE:.2f}s | "
                    f"Inference: {(t1 - t0) * 1000:.0f}ms | "
                    f"Total latency (last word -> text): {t1 - last_voice_time:.2f}s"
                )

    # ---------- shutdown ----------
    def stop_listening(self):
        print("\nStopping listener...")
        self.is_running = False
        time.sleep(0.3)
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, "p"):
            self.p.terminate()
        print("Stopped successfully.")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    stt = RealtimeSTT(
        model_path=LOCAL_MODEL_PATH,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    stt.start_listening()