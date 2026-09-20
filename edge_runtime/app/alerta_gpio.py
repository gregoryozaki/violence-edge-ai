import time


class AlertaGPIO:
    def __init__(self, gpio_verde=17, gpio_amarelo=22, gpio_vermelho=None):
        self.disponivel = False
        self.verde = None
        self.amarelo = None
        self.vermelho = None

        try:
            from gpiozero import LED

            self.verde = LED(gpio_verde) if gpio_verde is not None else None
            self.amarelo = LED(gpio_amarelo) if gpio_amarelo is not None else None
            self.vermelho = LED(gpio_vermelho) if gpio_vermelho is not None else None
            self.disponivel = True
            print("[GPIO] LEDs inicializados")
        except Exception as erro:
            print(f"[GPIO] Rodando sem GPIO real: {erro}")

    def desligar_todos(self):
        for led in [self.verde, self.amarelo, self.vermelho]:
            if led is not None:
                led.off()

    def piscar(self, led, nome, vezes=1, duracao=0.2, pausa=0.2):
        print(f"[LED] {nome}: {vezes}x")

        if led is None:
            time.sleep(0.2)
            return

        self.desligar_todos()

        for _ in range(vezes):
            led.on()
            time.sleep(duracao)
            led.off()
            time.sleep(pausa)

    def sistema_ligado(self):
        self.piscar(self.verde, "sistema ligado / verde", vezes=1, duracao=1.0)

    def normal(self):
        self.piscar(self.verde, "normal / descartado", vezes=1)

    def suspeito(self):
        self.piscar(self.amarelo, "suspeito", vezes=2)

    def camera_acionada(self):
        self.piscar(self.amarelo, "camera acionada", vezes=1, duracao=1.0)

    def critico(self):
        led = self.vermelho if self.vermelho is not None else self.amarelo
        self.piscar(led, "alerta critico", vezes=5, duracao=0.08, pausa=0.08)
