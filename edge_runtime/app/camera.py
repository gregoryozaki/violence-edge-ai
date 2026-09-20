from pathlib import Path
from datetime import datetime
import time
import cv2


def capturar_frame(config_camera):
    camera_id = int(config_camera.get("dispositivo", 0))
    largura = int(config_camera.get("largura", 640))
    altura = int(config_camera.get("altura", 480))

    pasta = Path("captures/images")
    pasta.mkdir(parents=True, exist_ok=True)

    nome = datetime.now().strftime("imagem_%Y%m%d_%H%M%S.jpg")
    caminho = pasta / nome

    camera = cv2.VideoCapture(camera_id)

    if not camera.isOpened():
        raise RuntimeError("Nao consegui abrir a camera")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    for _ in range(10):
        camera.read()
        time.sleep(0.03)

    ok, frame = camera.read()
    camera.release()

    if not ok:
        raise RuntimeError("Nao consegui capturar imagem")

    cv2.imwrite(str(caminho), frame)

    return frame, caminho
