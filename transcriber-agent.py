#1 importing of necessary libraries
import whisper # package for transcription 
import openai #the LLM of the agent, might be changed
import os #helps the codebase to navigate through system
from datetime import datetime #timestamping files
from dotenv import load_dotenv # 
import argparse # arge-parse - pass argument or audio path from terminal
import tkinter as tk
from tkinter import filedialog, messagebox


#2 Configuration - define the settings the program needs - control panel 
##like setting up a google map gps before a trip to coventry
openai.api_key = "sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl" #permission to use gpt from openAi
whisper_model_size = "base"
audio_file_path = "" #what audio file to transcribe
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"transcription_{TIMESTAMP}.txt"

def input_audio(file_path):
    import torchaudio 
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
        return waveform

def preprocess_audio(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform / waveform.abs().max()

    mel_spec = mel_spectrogram(waveform)

    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return log_mel_spec


model = whisper.load_model("base")

def transcribe_audio(log_mel_spec):
    if isinstance(log_mel_spec, torch.tensor):
        audio_np = log_mel_spec.squeeze().numpy()

    result = model.transcribe(audio_np) 
    return result['textt']

    def main():
        ## file_path = ""
        file_path = input("Type in audio file path: ")
        try:
            waveform = input_audio(file_path)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return 

        waveform = input_audio(file_path)
        processed_audio = preprocess_audio(waveform)
        transcript = transcribe_audio(processed_audio)
        print(transcript)
    
    if __name__ == "__main__":
        main()



def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    if file_path:
        file_label.config(text=file_path)
        # Store the path somewhere accessible, e.g., global variable
        global selected_file
        selected_file = file_path

def transcribe():
    if not selected_file:
        messagebox.showwarning("No file", "Please select an audio file first.")
        return
    try:
        waveform = input_audio(selected_file)
        processed_audio = preprocess_audio(waveform)
        transcript = transcribe_audio(processed_audio)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, transcript)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to transcribe: {e}")

# Tkinter setup
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
