import torch
import torchaudio
import whisper
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Load Whisper model once
model = whisper.load_model("base")


def input_audio(file_path):
    # Load waveform and sample rate using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono by averaging channels if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform.squeeze(0)  # Remove channel dim, shape: [num_samples]


def transcribe_audio(waveform):
    # Whisper expects numpy array or torch tensor waveform sampled at 16kHz
    # Convert to numpy for whisper (torch tensor also accepted in latest whisper)
    audio_np = waveform.numpy()

    # Pad or trim audio to fit model input length
    audio_np = whisper.pad_or_trim(audio_np)

    # Get log mel spectrogram for input
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio_np))

    # Transcribe
    result = model.transcribe(mel)
    return result['text']


def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a")]
    )
    if file_path:
        file_label.config(text=file_path)
        global selected_file
        selected_file = file_path


def transcribe():
    if not selected_file:
        messagebox.showwarning("No file", "Please select an audio file first.")
        return
    try:
        waveform = input_audio(selected_file)
        transcript = transcribe_audio(waveform)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, transcript)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to transcribe: {e}")


# Tkinter UI setup
root = tk.Tk()
root.title("Audio Transcription App")

selected_file = None

select_btn = tk.Button(root, text="Select Audio File", command=select_file)
select_btn.pack(pady=10)

file_label = tk.Label(root, text="No file selected")
file_label.pack()

transcribe_btn = tk.Button(root, text="Transcribe", command=transcribe)
transcribe_btn.pack(pady=10)

result_text = tk.Text(root, height=10, width=50)
result_text.pack()

root.mainloop()
