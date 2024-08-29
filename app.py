import streamlit as st
from TTS.api import TTS
import torch
import speech_recognition as sr
import io
from pydub import AudioSegment

# Initialize TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

st.title("Voice Cloning Web App")

# Real-time text input
text = st.text_input("Enter text to synthesize:")

# Upload voice input
st.header("Upload Your Voice")
uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_audio:
    audio_segment = AudioSegment.from_file(uploaded_audio)
    audio_bytes = io.BytesIO()
    audio_segment.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    st.audio(audio_bytes, format="audio/wav")
    recognizer = sr.Recognizer()
    recorded_audio = sr.AudioFile(audio_bytes)

    # Generate speech if both text and voice are provided
    if text:
        output_path = "output_audio.wav"
        with recorded_audio as source:
            audio_data = recognizer.record(source)
            tts.tts_to_file(
                text=text,
                speaker_wav=io.BytesIO(audio_data.get_wav_data()),
                language="en",
                file_path=output_path,
                split_sentences=True
            )
        st.audio(output_path, format="audio/wav")
    else:
        st.warning("Please enter text to synthesize.")
else:
    st.warning("Please upload an audio file.")
