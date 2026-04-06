import runpod
import base64
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os


MODEL_ID = "BadiniAI/BadiniW2VBert"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

print("Model loaded.")
print("HF_TOKEN:", HF_TOKEN)
def process_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    speech, sr = librosa.load("temp.wav", sr=16000)

    input_values = processor(
        speech,
        return_tensors="pt",
        sampling_rate=16000
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription


def handler(job):
    try:
        audio_base64 = job["input"]["audio"]
        audio_bytes = base64.b64decode(audio_base64)

        result = process_audio(audio_bytes)

        return {"text": result}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

print("New build trigger")