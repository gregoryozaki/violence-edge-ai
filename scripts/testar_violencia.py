import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf


def encontrar_raiz_projeto() -> Path:
    caminho_script = Path(__file__).resolve()

    for candidato in [caminho_script.parent, *caminho_script.parents]:
        if (
            (candidato / "models").exists()
            and (candidato / "scripts").exists()
        ):
            return candidato

    raise FileNotFoundError(
        "Não foi possível localizar a raiz do projeto violence-edge-ai."
    )


def configurar_gpu() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def redimensionar_com_corte_central(
    quadro: np.ndarray,
    tamanho: int,
) -> np.ndarray:
    altura, largura = quadro.shape[:2]
    escala = max(tamanho / largura, tamanho / altura)

    nova_largura = max(tamanho, int(round(largura * escala)))
    nova_altura = max(tamanho, int(round(altura * escala)))

    quadro = cv2.resize(
        quadro,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA,
    )

    inicio_x = (nova_largura - tamanho) // 2
    inicio_y = (nova_altura - tamanho) // 2

    return quadro[
        inicio_y : inicio_y + tamanho,
        inicio_x : inicio_x + tamanho,
    ]


def extrair_quadros(
    caminho_video: Path,
    quantidade_quadros: int,
    tamanho_imagem: int,
) -> np.ndarray:
    captura = cv2.VideoCapture(str(caminho_video))

    if not captura.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {caminho_video}")

    total_quadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_quadros <= 0:
        captura.release()
        raise ValueError(f"Vídeo sem quadros válidos: {caminho_video}")

    indices = np.linspace(
        0,
        total_quadros - 1,
        quantidade_quadros,
        dtype=int,
    )

    quadros: list[np.ndarray] = []

    for indice in indices:
        captura.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
        sucesso, quadro = captura.read()

        if not sucesso:
            if quadros:
                quadros.append(quadros[-1].copy())
                continue

            captura.release()
            raise ValueError(
                f"Não foi possível ler o quadro {indice}: {caminho_video}"
            )

        quadro = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        quadro = redimensionar_com_corte_central(
            quadro,
            tamanho_imagem,
        )
        quadros.append(quadro)

    captura.release()

    video = np.asarray(quadros, dtype=np.float32)
    return np.expand_dims(video, axis=0)


def classificar(probabilidade: float, limiar: float) -> str:
    return "violencia" if probabilidade >= limiar else "nao_violencia"


def prever_keras(
    caminho_modelo: Path,
    video: np.ndarray,
) -> tuple[float, tuple[Any, ...]]:
    modelo = tf.keras.models.load_model(
        caminho_modelo,
        compile=False,
    )

    formato_entrada = tuple(modelo.input_shape)

    if len(formato_entrada) != 5:
        raise ValueError(
            f"Formato inesperado do modelo Keras: {formato_entrada}"
        )

    probabilidade = float(
        np.asarray(modelo.predict(video, verbose=0)).reshape(-1)[0]
    )

    return probabilidade, formato_entrada


def quantizar_entrada(
    entrada: np.ndarray,
    detalhes: dict[str, Any],
) -> np.ndarray:
    dtype = detalhes["dtype"]

    if np.issubdtype(dtype, np.floating):
        return entrada.astype(dtype)

    escala, ponto_zero = detalhes["quantization"]

    if escala == 0:
        raise ValueError("O modelo quantizado não informou escala válida.")

    quantizada = np.round(entrada / escala + ponto_zero)
    limites = np.iinfo(dtype)

    return np.clip(
        quantizada,
        limites.min,
        limites.max,
    ).astype(dtype)


def desquantizar_saida(
    saida: np.ndarray,
    detalhes: dict[str, Any],
) -> np.ndarray:
    if np.issubdtype(saida.dtype, np.floating):
        return saida.astype(np.float32)

    escala, ponto_zero = detalhes["quantization"]

    if escala == 0:
        return saida.astype(np.float32)

    return (saida.astype(np.float32) - ponto_zero) * escala


def prever_tflite(
    caminho_modelo: Path,
    video: np.ndarray,
) -> tuple[float, list[int], str]:
    interpretador = tf.lite.Interpreter(
        model_path=str(caminho_modelo),
        num_threads=max(1, (os.cpu_count() or 2) // 2),
    )
    interpretador.allocate_tensors()

    entrada = interpretador.get_input_details()[0]
    saida = interpretador.get_output_details()[0]

    formato = entrada["shape"].tolist()

    if list(video.shape) != formato:
        raise ValueError(
            f"Formato do vídeo {list(video.shape)} diferente do "
            f"modelo TFLite {formato}."
        )

    interpretador.set_tensor(
        entrada["index"],
        quantizar_entrada(video, entrada),
    )
    interpretador.invoke()

    valores = interpretador.get_tensor(saida["index"])
    valores = desquantizar_saida(valores, saida)
    probabilidade = float(valores.reshape(-1)[0])

    return probabilidade, formato, np.dtype(entrada["dtype"]).name


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Testa o classificador de violência Keras e suas exportações "
            "TFLite usando um vídeo externo."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Caminho do vídeo que será analisado.",
    )
    parser.add_argument(
        "--limiar",
        type=float,
        default=0.50,
        help="Limiar para classificar violência. Padrão: 0.50.",
    )
    parser.add_argument(
        "--esperado",
        choices=["violencia", "nao_violencia"],
        help="Classe esperada para conferir manualmente o resultado.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = criar_argumentos()
    configurar_gpu()

    raiz = encontrar_raiz_projeto()
    caminho_video = argumentos.video.expanduser().resolve()

    if not caminho_video.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    pasta_modelos = raiz / "models" / "violence"
    caminho_keras = pasta_modelos / "violencia_final.keras"
    caminho_float32 = pasta_modelos / "tflite" / "violencia_float32.tflite"
    caminho_float16 = pasta_modelos / "tflite" / "violencia_float16.tflite"

    for caminho in (caminho_keras, caminho_float32, caminho_float16):
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {caminho}")

    modelo_temporario = tf.keras.models.load_model(
        caminho_keras,
        compile=False,
    )
    formato = tuple(modelo_temporario.input_shape)
    del modelo_temporario

    if len(formato) != 5:
        raise ValueError(f"Formato inesperado: {formato}")

    quantidade_quadros = int(formato[1])
    tamanho_imagem = int(formato[2])

    video = extrair_quadros(
        caminho_video,
        quantidade_quadros,
        tamanho_imagem,
    )

    resultados: dict[str, dict[str, Any]] = {}

    prob_keras, entrada_keras = prever_keras(
        caminho_keras,
        video,
    )
    resultados["keras"] = {
        "probabilidade": prob_keras,
        "classe": classificar(prob_keras, argumentos.limiar),
        "entrada": list(entrada_keras),
    }

    for nome, caminho in (
        ("tflite_float32", caminho_float32),
        ("tflite_float16", caminho_float16),
    ):
        probabilidade, entrada, dtype = prever_tflite(
            caminho,
            video,
        )
        resultados[nome] = {
            "probabilidade": probabilidade,
            "classe": classificar(
                probabilidade,
                argumentos.limiar,
            ),
            "entrada": entrada,
            "dtype_entrada": dtype,
            "diferenca_keras": abs(probabilidade - prob_keras),
        }

    print("\n" + "=" * 72)
    print("Vídeo:", caminho_video)
    print("Quadros usados:", quantidade_quadros)
    print("Tamanho:", f"{tamanho_imagem}x{tamanho_imagem}")
    print("TensorFlow GPU:", bool(tf.config.list_physical_devices("GPU")))
    print("Limiar:", argumentos.limiar)

    if argumentos.esperado:
        print("Classe esperada:", argumentos.esperado)

    print("-" * 72)

    for nome, dados in resultados.items():
        correto = (
            argumentos.esperado is None
            or dados["classe"] == argumentos.esperado
        )
        marcador = "OK" if correto else "DIVERGIU"

        print(
            f"{nome:18s} "
            f"prob={dados['probabilidade']:.6f} "
            f"classe={dados['classe']:14s} "
            f"{marcador}"
        )

    relatorio = {
        "video": str(caminho_video),
        "classe_esperada": argumentos.esperado,
        "limiar": argumentos.limiar,
        "quantidade_quadros": quantidade_quadros,
        "tamanho_imagem": tamanho_imagem,
        "resultados": resultados,
    }

    caminho_saida = (
        raiz
        / "outputs"
        / "metrics"
        / "violence"
        / "ultima_inferencia_externa.json"
    )
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_saida.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 72)
    print("Relatório salvo em:", caminho_saida)


if __name__ == "__main__":
    main()
