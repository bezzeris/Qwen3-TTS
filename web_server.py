import os
import torch
import soundfile as sf
import io
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

# Configuración del modelo
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Forzamos la detección de la Nvidia
if torch.cuda.is_available():
    device = "cuda:0" # Forzamos la primera GPU (Nvidia)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 ¡ÉXITO! Usando GPU Nvidia: {gpu_name}")
else:
    device = "cpu"
    print("⚠️ ADVERTENCIA: No se detectó CUDA. Usando CPU (será muy lento).")

print(f"Cargando modelo en {device}... (esto puede tardar un momento)")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    dtype=torch.bfloat16 if "cuda" in device else torch.float32,
)

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    language: str = "Spanish"
    speaker: str = "Vivian"
    instruct: str = ""
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05

@app.post("/generate")
async def generate_audio(req: TTSRequest):
    try:
        # Generar audio con los parámetros del usuario
        wavs, sr = model.generate_custom_voice(
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=True
        )

        # Convertir a buffer de memoria para enviar al navegador
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Servir la interfaz HTML
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
