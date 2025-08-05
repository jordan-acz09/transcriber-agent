# 🎧 Whisper WAV/MP3 Audio Transcription Tool

This is a Python-based command-line tool that uses OpenAI's Whisper model to transcribe audio files (WAV, MP3, M4A) into text.

---

## 🧠 Features

- Converts any audio format into standardized WAV
- Uses Whisper model (`small`) to transcribe audio
- Automatically saves transcription to a timestamped `.txt` file
- Fully CLI-based with minimal dependencies

---

## ⚙️ Requirements

- Python 3.8+
- FFmpeg installed and added to system path

### Python Libraries

Install dependencies:

```bash
pip install torch openai-whisper pydub numpy
