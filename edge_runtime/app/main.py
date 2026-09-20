import argparse
import time
from pathlib import Path
import yaml

from alerta_gpio import AlertaGPIO
from audio import gravar_audio, calcular_rms_audio, detectar_vad_por_energia
from camera import capturar_frame
from telegram_bot import TelegramBot
from tflite_model import ModeloTFLite


def carregar_config():
    with open("configs/edge.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_modelos(cfg):
    modelos_cfg = cfg["modelos"]

    modelos = {
        "violencia": None,
        "armas": None,
        "aed": None,
    }

    for nome in modelos.keys():
        try:
            modelos[nome] = ModeloTFLite(modelos_cfg[nome])
        except Exception as erro:
            print(f"[MODELO] Nao consegui carregar {nome}: {erro}")

    return modelos


def montar_alerta(cfg):
    gpio = cfg.get("gpio", {})

    return AlertaGPIO(
        gpio_verde=gpio.get("led_verde", 17),
        gpio_amarelo=gpio.get("led_amarelo", 22),
        gpio_vermelho=gpio.get("led_vermelho", None),
    )


def executar_ciclo(cfg, alerta, telegram, modelos):
    decisao_cfg = cfg.get("decisao", {})

    limiar_vad = float(decisao_cfg.get("limiar_vad_rms", 5000000))
    limiar_violencia = float(decisao_cfg.get("limiar_violencia", 0.70))
    limiar_arma = float(decisao_cfg.get("limiar_arma", 0.50))

    print("\n========================================")
    print("[PIPELINE] Capturando audio...")

    caminho_audio = gravar_audio(cfg["audio"])
    metricas = calcular_rms_audio(caminho_audio)

    print(f"[AUDIO] Arquivo: {caminho_audio}")
    print(f"[AUDIO] RMS max: {metricas['rms_max']:.2f}")
    print(f"[AUDIO] RMS canais: {metricas['rms_por_canal']}")

    vad_positivo = detectar_vad_por_energia(metricas, limiar_vad)

    if not vad_positivo:
        print("[VAD] Negativo. Evento descartado.")
        alerta.normal()
        return

    print("[VAD] Positivo. Passando para analise visual.")
    alerta.suspeito()

    # AED TFLite ja esta no projeto, mas o preprocess real ainda precisa ser plugado.
    print("[AED] Modelo enviado, mas neste runtime o gatilho ainda usa VAD por energia.")

    print("[CAMERA] Capturando imagem...")
    alerta.camera_acionada()

    frame, caminho_imagem = capturar_frame(cfg["camera"])
    print(f"[CAMERA] Imagem: {caminho_imagem}")

    score_violencia = 0.0
    score_arma = 0.0

    if modelos["violencia"] is not None:
        score_violencia = modelos["violencia"].predizer_classificador_imagem(frame)

    if modelos["armas"] is not None:
        score_arma = modelos["armas"].predizer_detector_yolo_generico(frame)

    violencia_detectada = score_violencia >= limiar_violencia
    arma_detectada = score_arma >= limiar_arma

    print(f"[CV] Score violencia: {score_violencia:.4f} | limiar={limiar_violencia}")
    print(f"[CV] Score arma/faca: {score_arma:.4f} | limiar={limiar_arma}")

    if violencia_detectada or arma_detectada:
        print("[DECISAO] ALERTA: provavel situacao suspeita/violencia.")
        alerta.critico()

        texto = (
            "ALERTA DO SISTEMA MULTIMODAL\n"
            "Possivel situacao de violencia detectada.\n\n"
            f"Audio RMS: {metricas['rms_max']:.2f}\n"
            f"Score violencia: {score_violencia:.4f}\n"
            f"Score arma/faca: {score_arma:.4f}\n"
            f"Imagem: {caminho_imagem}"
        )

        telegram.enviar_foto(caminho_imagem, texto)
    else:
        print("[DECISAO] Sem confirmacao visual forte. Descartado.")
        alerta.normal()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Executa apenas um ciclo")
    parser.add_argument("--intervalo", type=float, default=2.0, help="Intervalo entre ciclos")
    args = parser.parse_args()

    cfg = carregar_config()

    alerta = montar_alerta(cfg)
    alerta.sistema_ligado()

    telegram = TelegramBot(ativado=cfg.get("telegram", {}).get("ativado", False))
    modelos = carregar_modelos(cfg)

    while True:
        try:
            executar_ciclo(cfg, alerta, telegram, modelos)
        except KeyboardInterrupt:
            print("\nEncerrando por teclado.")
            break
        except Exception as erro:
            print(f"[ERRO] {erro}")
            alerta.critico()

        if args.once:
            break

        time.sleep(args.intervalo)

    alerta.desligar_todos()


if __name__ == "__main__":
    main()
