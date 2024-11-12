import streamlit as st
import yt_dlp
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
import os


# Cache PEGASUS model and tokenizer to avoid reloading every time
@st.cache_resource
def load_pegasus_model():
    model_name = "google/pegasus-xsum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Cache Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Initialize the models
pegasus_model, pegasus_tokenizer = load_pegasus_model()
whisper_model = load_whisper_model()

# Function to download audio from YouTube
def download_audio(url, filename='audio.mp3'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return filename


# Transcription function using Whisper
def transcribe_audio(filename):
    result = whisper_model.transcribe(filename)
    transcript = result['text']
    return transcript

# Chunking and summarizing function
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def summarize_long_text(text, max_length=150, min_length=50):
    chunks = chunk_text(text, max_tokens=pegasus_tokenizer.model_max_length - 10)
    chunk_summaries = []

    for chunk in chunks:
        inputs = pegasus_tokenizer(chunk, return_tensors="pt", truncation=True)
        with torch.no_grad():
            summary_ids = pegasus_model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, no_repeat_ngram_size=2)
        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary)

    # Combine summaries of chunks into a final summary
    combined_summary = " ".join(chunk_summaries)
    return combined_summary

# Streamlit interface
st.title("YouTube Video Summarizer")
st.write("Provide a YouTube video link, and get a summary of the video.")

# User input for YouTube URL
url = st.text_input("Enter YouTube URL:")

if st.button("Generate Summary") and url:
    with st.spinner("Downloading audio..."):
        audio_file = download_audio(url)
        st.write("Audio downloaded successfully.")

    with st.spinner("Transcribing audio..."):
        transcript = transcribe_audio(audio_file)
        st.write("Transcription completed.")

    with st.spinner("Summarizing transcript..."):
        summary = summarize_long_text(transcript)
        st.write("Summary generated.")

    # Display the summary
    st.subheader("Summary:")
    st.write(summary)
