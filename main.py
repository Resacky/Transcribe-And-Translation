import os
from dotenv import load_dotenv
import queue
import threading
import pyaudio
import whisper
from transformers import MarianMTModel, MarianTokenizer

# --- Settings ---
AUDIO_DEVICE_INDEX = 2   # set to your mic or OBS virtual audio device
OUTPUT_FILE = "captions.txt"
LANGUAGE = "es"  # input language = Spanish

import shutil, sys
ffmpeg_path = r"C:\Users\Joseph Ruiz\scoop\apps\ffmpeg\current\bin"
# # Try to find ffmpeg automatically
# ffmpeg_path = shutil.which("ffmpeg")

# # If not found, check if user set an env var
# load_dotenv()
# if not ffmpeg_path:
#     ffmpeg_path = os.getenv("FFMPEG_PATH")

# # If still not found, fail gracefully
# if not ffmpeg_path:
#     sys.exit("ERROR: ffmpeg not found. Please install it or set FFMPEG_PATH env var.")

# # Add ffmpeg's folder to PATH for Whisper subprocess calls
# if os.path.dirname(ffmpeg_path) not in os.environ["PATH"].split(os.pathsep):
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

print("Using ffmpeg at:", ffmpeg_path)

# --- Load models ---
print("Loading models...")
asr_model = whisper.load_model("base")  # try "small" or "medium" if you want better accuracy
translator_model_name = "Helsinki-NLP/opus-mt-es-en"
tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator = MarianMTModel.from_pretrained(translator_model_name)

# --- Translation helper ---
def translate_spanish_to_english(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    outputs = translator.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Audio Capture ---
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def start_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=AUDIO_DEVICE_INDEX,
                    frames_per_buffer=4096,
                    stream_callback=audio_callback)
    stream.start_stream()
    return stream, p

# --- Processing Thread ---
def process_audio():
    import numpy as np
    import tempfile
    import wave

    while True:
        # collect ~5 seconds of audio
        frames = []
        for _ in range(0, int(16000 / 4096 * 5)):
            data = audio_queue.get()
            frames.append(data)

        # save temporary wav file
        tmp_wav = tempfile.mktemp(suffix=".wav")
        wf = wave.open(tmp_wav, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

        # run ASR
        result = asr_model.transcribe(tmp_wav, language=LANGUAGE)
        spanish_text = result["text"].strip()
        if not spanish_text:
            continue

        # translate
        english_text = translate_spanish_to_english(spanish_text)
        print(f"[ES] {spanish_text}\n[EN] {english_text}\n")

        # update captions file for OBS
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(english_text)

        os.remove(tmp_wav)

# --- Main ---
if __name__ == "__main__":
    stream, p = start_stream()
    threading.Thread(target=process_audio, daemon=True).start()
    print("🎤 Live translation started. Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
