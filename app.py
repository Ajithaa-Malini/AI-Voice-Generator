import streamlit as st
from TTS.api import TTS
import torch
import PyPDF2
import pytesseract
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment

# Initialize TTS model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

# Streamlit app setup
st.title("Interactive Voice Cloning Web App")

# Step 1: Upload Document for Text Extraction
st.header("Upload a Document")
uploaded_file = st.file_uploader("Choose a file (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)
    st.write("Extracted Text:")
    st.text_area("Text", text)

# Step 2: Record Real-time Audio
st.header("Record Your Voice")
recognizer = sr.Recognizer()

def record_audio():
    with sr.Microphone() as source:
        st.write("Recording...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        st.write("Recording complete.")
        return audio

if st.button("Start Recording"):
    recorded_audio = record_audio()

if 'recorded_audio' in locals():
    audio_path = "user_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.get_wav_data())
    st.audio(audio_path)

# Step 3: Generate Speech Using TTS
st.header("Generate Speech")
if 'text' in locals() and 'recorded_audio' in locals():
    output_path = "output_audio.wav"
    tts.tts_to_file(
        text=text,
        speaker_wav=audio_path,
        language="en",  # Or "es" based on the extracted text's language
        file_path=output_path,
        split_sentences=True
    )
    st.audio(output_path)
else:
    st.warning("Please upload a document and record your voice first.")