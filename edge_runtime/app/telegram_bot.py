import os
import requests
from dotenv import load_dotenv


class TelegramBot:
    def __init__(self, ativado=False):
        load_dotenv()

        self.ativado = bool(ativado)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if self.ativado and (not self.token or not self.chat_id):
            print("[TELEGRAM] Ativado no config, mas faltam TOKEN ou CHAT_ID no .env")
            self.ativado = False

    def enviar_mensagem(self, texto):
        if not self.ativado:
            print("[TELEGRAM] Desativado. Mensagem seria:")
            print(texto)
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        requests.post(
            url,
            data={
                "chat_id": self.chat_id,
                "text": texto
            },
            timeout=10
        )

    def enviar_foto(self, caminho_foto, legenda):
        if not self.ativado:
            print("[TELEGRAM] Desativado. Foto seria enviada:", caminho_foto)
            print(legenda)
            return

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"

        with open(caminho_foto, "rb") as arquivo:
            requests.post(
                url,
                data={
                    "chat_id": self.chat_id,
                    "caption": legenda
                },
                files={
                    "photo": arquivo
                },
                timeout=20
            )
