import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def main():
    # Configuración de rutas y archivos
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" # Usamos 0.6B para rapidez, cambia a 1.7B si prefieres
    OUT_DIR = "TTSWAV"
    ensure_dir(OUT_DIR)

    print(f"Cargando modelo {MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Nota: No usamos flash_attention_2 porque falló la instalación antes
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Audios de referencia (pueden ser URLs o rutas locales)
    ref_audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

    # Texto que queremos generar con la voz clonada
    textos_a_generar = [
        "Esta es una prueba de clonación de voz usando el modelo base de Qwen 3.",
        "Puedo hablar cualquier cosa manteniendo el timbre de la voz de referencia."
    ]

    print("Iniciando proceso de clonación...")

    for i, texto in enumerate(textos_a_generar):
        print(f" -> Generando audio {i+1}...")
        
        # Generar la clonación
        # x_vector_only_mode=False usa tanto el audio como el texto de referencia para mejor calidad
        wavs, sr = tts.generate_voice_clone(
            text=texto,
            language="Spanish",
            ref_audio=ref_audio_url,
            ref_text=ref_text,
            x_vector_only_mode=False
        )

        # Guardar en la carpeta TTSWAV
        filename = os.path.join(OUT_DIR, f"clonacion_prueba_{i+1}.wav")
        sf.write(filename, wavs[0], sr)
        print(f"    Guardado en: {filename}")

    print(f"\n¡Proceso finalizado! Los archivos están en la carpeta '{OUT_DIR}'")

if __name__ == "__main__":
    main()
