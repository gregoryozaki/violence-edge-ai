from pathlib import Path
import numpy as np
import cv2


def obter_interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except Exception:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except Exception:
        pass

    try:
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter
    except Exception:
        pass

    raise RuntimeError("Nenhum interpretador TFLite encontrado")


class ModeloTFLite:
    def __init__(self, caminho_modelo):
        self.caminho_modelo = Path(caminho_modelo)

        if not self.caminho_modelo.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {self.caminho_modelo}")

        Interpreter = obter_interpreter()
        self.interpreter = Interpreter(model_path=str(self.caminho_modelo))
        self.interpreter.allocate_tensors()

        self.entradas = self.interpreter.get_input_details()
        self.saidas = self.interpreter.get_output_details()

        print(f"[TFLITE] Carregado: {self.caminho_modelo}")
        for entrada in self.entradas:
            print(f"  entrada: {entrada['name']} shape={entrada['shape']} dtype={entrada['dtype']}")
        for saida in self.saidas:
            print(f"  saida: {saida['name']} shape={saida['shape']} dtype={saida['dtype']}")

    def _quantizar_entrada(self, entrada, array_float):
        dtype = entrada["dtype"]

        if dtype == np.float32:
            return array_float.astype(np.float32)

        escala, zero = entrada.get("quantization", (0.0, 0))

        if escala and escala > 0:
            array = array_float / escala + zero
            return np.clip(array, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

        return array_float.astype(dtype)

    def _desquantizar_saida(self, saida, array):
        escala, zero = saida.get("quantization", (0.0, 0))

        if escala and escala > 0:
            return (array.astype(np.float32) - zero) * escala

        return array.astype(np.float32)

    def _preparar_imagem(self, frame_bgr):
        entrada = self.entradas[0]
        shape = entrada["shape"]

        altura = int(shape[1])
        largura = int(shape[2])

        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (largura, altura))

        img_float = img.astype(np.float32) / 255.0
        img_float = np.expand_dims(img_float, axis=0)

        return self._quantizar_entrada(entrada, img_float)

    def predizer_classificador_imagem(self, frame_bgr):
        entrada = self.entradas[0]
        tensor = self._preparar_imagem(frame_bgr)

        self.interpreter.set_tensor(entrada["index"], tensor)
        self.interpreter.invoke()

        saida = self.saidas[0]
        pred = self.interpreter.get_tensor(saida["index"])
        pred = self._desquantizar_saida(saida, pred)
        pred = np.squeeze(pred)

        if pred.size == 1:
            return float(pred)

        pred = pred.reshape(-1)
        return float(np.max(pred))

    def predizer_detector_yolo_generico(self, frame_bgr):
        entrada = self.entradas[0]
        tensor = self._preparar_imagem(frame_bgr)

        self.interpreter.set_tensor(entrada["index"], tensor)
        self.interpreter.invoke()

        saida = self.saidas[0]
        pred = self.interpreter.get_tensor(saida["index"])
        pred = self._desquantizar_saida(saida, pred)
        pred = np.squeeze(pred)

        if pred.ndim == 2:
            # Caso YOLO: [6, 8400] ou [8400, 6]
            if pred.shape[0] < pred.shape[1]:
                pred = pred.T

            if pred.shape[1] >= 5:
                scores = pred[:, 4:]
                return float(np.max(scores))

        return float(np.max(pred))

    def teste_dummy(self):
        for entrada in self.entradas:
            shape = entrada["shape"]
            dtype = entrada["dtype"]
            dado = np.random.random_sample(shape).astype(np.float32)
            dado = self._quantizar_entrada(entrada, dado)
            self.interpreter.set_tensor(entrada["index"], dado)

        self.interpreter.invoke()

        resultados = []

        for saida in self.saidas:
            pred = self.interpreter.get_tensor(saida["index"])
            pred = self._desquantizar_saida(saida, pred)
            resultados.append(pred)

        return resultados
