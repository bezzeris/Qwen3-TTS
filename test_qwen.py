import torch
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel

# Cargamos el modelo pequeño (0.6B) para que sea rápido
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda",
    dtype=torch.bfloat16,
)

# Lista de voces disponibles según la documentación
speakers = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
    "Ryan", "Aiden", "Ono_Anna", "Sohee"
]

# Creamos una carpeta para los resultados si no existe
output_dir = "muestras_voces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Generando muestras para {len(speakers)} voces...")

for speaker in speakers:
    print(f" -> Generando voz de: {speaker}...")
    try:
        wavs, sr = model.generate_custom_voice(
            text=f"Hola, mi nombre es {speaker}. Esta es una prueba de mi voz en español.",
            language="Spanish",
            speaker=speaker,
        )
        
        filename = f"{output_dir}/prueba_{speaker}.wav"
        sf.write(filename, wavs[0], sr)
    except Exception as e:
        print(f"Error con {speaker}: {e}")

print(f"\n¡Hecho! Puedes encontrar todas las voces en la carpeta: {output_dir}")