import os
from pathlib import Path
import tempfile
import wave
import numpy as np
import sounddevice as sd
import whisper
from transformers import MarianMTModel, MarianTokenizer

PROJECT_DIR = Path(__file__).parent


# -----------------------------
# Load models
# -----------------------------
print("Loading Whisper model...")
asr_model = whisper.load_model("small")  # small/medium/large depending on hardware

print("Loading MarianMT translation model...")
mt_model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name)

# -----------------------------
# Helper functions
# -----------------------------
def record_chunk(duration=5, fs=16000):
    """Record audio from mic for a short duration and return temp WAV filename."""
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = np.squeeze(audio)

    # Define project subdirectory
    audio_dir = PROJECT_DIR / "recordings"
    audio_dir.mkdir(exist_ok=True)

    # Save WAV file in subdirectory
    tmp_file = audio_dir / "tmp_audio.wav"

    # Write WAV file
    with wave.open(str(tmp_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    print(f"WAV saved at: {tmp_file}")
    return str(tmp_file)

def translate_text(text):
    """Translate Spanish text to English using MarianMT."""
    batch = tokenizer([text], return_tensors="pt", padding=True)
    gen = mt_model.generate(**batch)
    translation = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translation

# -----------------------------
# Live loop
# -----------------------------
# Output file for OBS live captions
translation_dir = PROJECT_DIR / "translation"
translation_dir.mkdir(exist_ok=True)
obs_file = translation_dir / "live_translation.txt"

print("Live translation started. Press Ctrl+C to stop.")

try:
    while True:
        tmp_wav = record_chunk(duration=5)  # adjust duration for responsiveness
        asr_result = asr_model.transcribe(tmp_wav, language="es")
        spanish_text = asr_result["text"]
        english_text = translate_text(spanish_text)

        # Print to console
        print(f"ES: {spanish_text}")
        print(f"EN: {english_text}")
        print("-" * 30)

        # Write live translation to file for OBS
        with open(obs_file, "w", encoding="utf-8") as f:
            f.write(english_text)

        # os.remove(tmp_wav)

except KeyboardInterrupt:
    print("\nStopped by user")
