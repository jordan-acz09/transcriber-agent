import numpy as np
import whisper
from pydub import AudioSegment
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Load whisper model once (small size)
model = whisper.load_model("small")

selected_file = None

def preprocess_audio_np(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)  # normalize 16-bit PCM
    return samples

def transcribe_file(path):
    audio_np = preprocess_audio_np(path)
    # Pass raw audio samples directly — no pad_or_trim or mel spectrogram here!
    result = model.transcribe(audio_np, language="en")
    return result["text"]

def select_file():
    global selected_file
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a")]
    )
    if file_path:
        selected_file = file_path
        file_label.config(text=file_path)

def transcribe():
    if not selected_file:
        messagebox.showwarning("No file selected", "Please select an audio file first.")
        return
    try:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Transcribing...\n")
        transcript = transcribe_file(selected_file)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, transcript)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcription_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        messagebox.showinfo("Saved", f"Transcription saved to {output_file}")

    except Exception as e:
        print(f"❌ Error during transcription: {e}")  # print error to terminal
        messagebox.showerror("Error", f"Failed to transcribe:\n{e}")

# Tkinter UI setup
root = tk.Tk()
root.title("Audio Transcription App")

select_btn = tk.Button(root, text="Select Audio File (Must be a .wav File)", command=select_file)
select_btn.pack(pady=10)

file_label = tk.Label(root, text="No file selected")
file_label.pack()

transcribe_btn = tk.Button(root, text="Transcribe", command=transcribe)
transcribe_btn.pack(pady=10)

result_text = tk.Text(root, height=15, width=60)
result_text.pack()

root.mainloop()
