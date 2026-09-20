from pathlib import Path
from datetime import datetime
import subprocess
import wave
import numpy as np


def gravar_audio(config_audio):
    dispositivo = config_audio.get("dispositivo", "plughw:CARD=sndrpigooglevoi,DEV=0")
    taxa = int(config_audio.get("taxa", 48000))
    canais = int(config_audio.get("canais", 2))
    formato = config_audio.get("formato", "S32_LE")
    duracao = int(config_audio.get("janela_segundos", 3))

    pasta = Path("captures/audio")
    pasta.mkdir(parents=True, exist_ok=True)

    nome = datetime.now().strftime("audio_%Y%m%d_%H%M%S.wav")
    caminho = pasta / nome

    comando = [
        "arecord",
        "-D", dispositivo,
        "-c", str(canais),
        "-r", str(taxa),
        "-f", formato,
        "-t", "wav",
        "-d", str(duracao),
        str(caminho),
    ]

    subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return caminho


def calcular_rms_audio(caminho_audio):
    with wave.open(str(caminho_audio), "rb") as wav:
        canais = wav.getnchannels()
        largura = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if largura == 4:
        dados = np.frombuffer(frames, dtype=np.int32)
    elif largura == 2:
        dados = np.frombuffer(frames, dtype=np.int16)
    else:
        raise RuntimeError(f"Largura de amostra nao suportada: {largura}")

    if canais > 1:
        dados = dados.reshape(-1, canais)

    dados = dados.astype(np.float32)
    dados = dados - np.mean(dados, axis=0)

    rms_por_canal = np.sqrt(np.mean(dados ** 2, axis=0))
    rms_max = float(np.max(rms_por_canal))

    return {
        "rms_max": rms_max,
        "rms_por_canal": rms_por_canal.tolist()
    }


def detectar_vad_por_energia(metricas, limiar):
    return metricas["rms_max"] >= limiar
