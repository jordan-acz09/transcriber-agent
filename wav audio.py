import numpy as np
import whisper
from pydub import AudioSegment
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox


def preprocess_audio_np(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)  # normalize 16-bit PCM
    return samples

def main():
    model_size = "small"
    model = whisper.load_model(model_size)

    audio_path = input("Enter path to audio file (WAV/MP3 etc): ").strip('"')

    try:
        print(f"📂 Preprocessing input file: {audio_path}")
        audio_np = preprocess_audio_np(audio_path)

        print("🔍 Loading audio into Whisper from numpy array...")
        audio = whisper.pad_or_trim(whisper.log_mel_spectrogram(audio_np))

        print("🧠 Transcribing...")
        result = model.transcribe(audio_np, language="en")  # direct numpy input if supported

        transcript = result["text"]
        print("\n=== TRANSCRIPTION ===\n")
        print(transcript)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcription_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"\n💾 Transcription saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()


