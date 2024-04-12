from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import soundfile as sf
import librosa
import transformers
import torch
import os

app = FastAPI()

@app.post("/processfile/")
async def process_file(file: UploadFile = File(...)):
    """
    Submit an audio file

    This endpoint will take an audio file, generate a transcription, determine sentiment and return both
    """
    ########################################
    # Process audio file and run transcription
    ########################################
    login(token="hf_AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    # Convert MP3 bytes to WAV for processing
    audio_data, sampling_rate = sf.read(file.file, dtype='float32')

    # Adjust the sample rate to what Whisper expects, 16000
    audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)

    # Load the Whisper model and processor just once when the app starts
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    # Process the audio waveform with the Whisper model
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    ########################################
    # Process sentiment and return
    ########################################
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Tokenize the input text
    inputs = tokenizer(transcription[0], return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the highest probability to determine the sentiment
    _, predicted_class = torch.max(probabilities, dim=1)

    # Map the predicted class to label
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[predicted_class.item()]

    print(f"Transcription: {transcription[0]}")
    print(f"Sentiment: {sentiment}")
    print(f"Probabilities: {probabilities[0].numpy()}")

    return {"Transcription": transcription[0], "Sentiment": sentiment}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    details = f"{request.method} {request.url.path} - {response.status_code}"
    print(details)  # Use a proper logger in production
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
