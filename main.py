import os
from pathlib import Path
import wave
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel

# -----------------------------
# Directories
# -----------------------------
PROJECT_DIR = Path(__file__).parent

AUDIO_DIR = PROJECT_DIR / "recordings"
AUDIO_DIR.mkdir(exist_ok=True)

TRANSLATION_DIR = PROJECT_DIR / "translation"
TRANSLATION_DIR.mkdir(exist_ok=True)

obs_file = TRANSLATION_DIR / "live_translation.txt"
transcription_log = TRANSLATION_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_transcription.txt"
translation_log = TRANSLATION_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_translation.txt"

# -----------------------------
# Model setup
# -----------------------------
device = "metal"  # Apple Silicon GPU
compute_type = "float16"  # float16 for speed, int8 for CPU fallback

print("Loading faster-whisper model...")
model = WhisperModel("large-v3", device=device, compute_type=compute_type)

# -----------------------------
# Recording helper
# -----------------------------
FS = 16000
DURATION = 6          # seconds per chunk
SILENCE_RMS = 0.005   # skip silent chunks

def record_chunk(duration=DURATION, fs=FS):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)

    rms = np.sqrt(np.mean(audio**2))
    return audio, rms

def save_wav(samples: np.ndarray, filename: Path, fs=FS):
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(fs)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())

# -----------------------------
# Live loop
# -----------------------------
print("Live translation started (ES → EN). Press Ctrl+C to stop.")

try:
    while True:
        audio, rms = record_chunk()
        if rms < SILENCE_RMS:
            print(f"(skip silent chunk, rms={rms:.4f})")
            continue

        tmp_wav = AUDIO_DIR / "tmp_audio.wav"
        save_wav(audio, tmp_wav)
        print(f"WAV saved at: {tmp_wav}")

        # Transcribe (Spanish)
        segments_es, info_es = model.transcribe(
            str(tmp_wav),
            task="transcribe",  # Spanish transcription
            language="es",
            beam_size=5,
            vad_filter=True
        )
        spanish_text = " ".join(s.text for s in segments_es).strip()

        # Translate (English)
        segments_en, info_en = model.transcribe(
            str(tmp_wav),
            task="translate",  # Spanish → English
            language="es",
            beam_size=5,
            vad_filter=True
        )
        english_text = " ".join(s.text for s in segments_en).strip()

        # Print to console
        print(f"ES: {spanish_text}")
        print(f"EN: {english_text}")
        print("-" * 30)

        # Save logs
        if spanish_text:
            with open(transcription_log, "a", encoding="utf-8") as f:
                f.write(spanish_text + "\n")
        if english_text:
            with open(translation_log, "a", encoding="utf-8") as f:
                f.write(english_text + "\n")

        # Update OBS live caption file
        with open(obs_file, "w", encoding="utf-8") as f:
            f.write(english_text)

except KeyboardInterrupt:
    print("\nStopped by user")
