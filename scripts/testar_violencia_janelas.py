import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf


NOMES_MODELOS = {
    "cabeca": "violencia_cabeca_classificadora.keras",
    "ajuste": "violencia_ajuste_fino.keras",
    "final": "violencia_final.keras",
}


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


def obter_metadados_video(caminho: Path) -> dict[str, float]:
    captura = cv2.VideoCapture(str(caminho))

    if not captura.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {caminho}")

    fps = float(captura.get(cv2.CAP_PROP_FPS))
    total_quadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    captura.release()

    if fps <= 0 or total_quadros <= 0:
        raise ValueError("O vídeo não possui FPS ou quadros válidos.")

    return {
        "fps": fps,
        "total_quadros": total_quadros,
        "duracao": total_quadros / fps,
    }


def extrair_janela(
    caminho_video: Path,
    inicio_segundos: float,
    fim_segundos: float,
    quantidade_quadros: int,
    tamanho_imagem: int,
    fps: float,
    total_quadros: int,
) -> np.ndarray:
    captura = cv2.VideoCapture(str(caminho_video))

    inicio_quadro = max(0, int(round(inicio_segundos * fps)))
    fim_quadro = min(
        total_quadros - 1,
        int(round(fim_segundos * fps)),
    )

    indices = np.linspace(
        inicio_quadro,
        fim_quadro,
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
                f"Falha ao ler o quadro {indice} do vídeo."
            )

        quadro = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        quadro = redimensionar_com_corte_central(
            quadro,
            tamanho_imagem,
        )
        quadros.append(quadro)

    captura.release()

    janela = np.asarray(quadros, dtype=np.float32)
    return np.expand_dims(janela, axis=0)


def criar_intervalos(
    duracao: float,
    tamanho_janela: float,
    passo: float,
) -> list[tuple[float, float]]:
    if duracao <= tamanho_janela:
        return [(0.0, duracao)]

    intervalos: list[tuple[float, float]] = []
    inicio = 0.0

    while inicio < duracao:
        fim = min(inicio + tamanho_janela, duracao)

        if fim - inicio < tamanho_janela * 0.50:
            inicio = max(0.0, duracao - tamanho_janela)
            fim = duracao

        intervalo = (round(inicio, 3), round(fim, 3))

        if not intervalos or intervalo != intervalos[-1]:
            intervalos.append(intervalo)

        if fim >= duracao:
            break

        inicio += passo

    return intervalos


def testar_modelo(
    caminho_modelo: Path,
    caminho_video: Path,
    intervalos: list[tuple[float, float]],
    metadados: dict[str, float],
    limiar: float,
) -> dict[str, Any]:
    tf.keras.backend.clear_session()

    modelo = tf.keras.models.load_model(
        caminho_modelo,
        compile=False,
    )

    formato = tuple(modelo.input_shape)

    if len(formato) != 5:
        raise ValueError(
            f"Formato inesperado em {caminho_modelo.name}: {formato}"
        )

    quantidade_quadros = int(formato[1])
    tamanho_imagem = int(formato[2])
    resultados = []

    for indice, (inicio, fim) in enumerate(intervalos, start=1):
        janela = extrair_janela(
            caminho_video=caminho_video,
            inicio_segundos=inicio,
            fim_segundos=fim,
            quantidade_quadros=quantidade_quadros,
            tamanho_imagem=tamanho_imagem,
            fps=metadados["fps"],
            total_quadros=int(metadados["total_quadros"]),
        )

        probabilidade = float(
            np.asarray(
                modelo.predict(janela, verbose=0)
            ).reshape(-1)[0]
        )

        resultados.append(
            {
                "janela": indice,
                "inicio": inicio,
                "fim": fim,
                "probabilidade": probabilidade,
                "violencia": probabilidade >= limiar,
            }
        )

    probabilidades = [
        item["probabilidade"]
        for item in resultados
    ]
    quantidade_positivas = sum(
        item["violencia"]
        for item in resultados
    )

    return {
        "modelo": caminho_modelo.name,
        "entrada": list(formato),
        "resultados": resultados,
        "maximo": max(probabilidades),
        "media": float(np.mean(probabilidades)),
        "mediana": float(np.median(probabilidades)),
        "janelas_positivas": quantidade_positivas,
        "total_janelas": len(resultados),
    }


def salvar_csv(
    caminho_saida: Path,
    resultados_modelos: dict[str, dict[str, Any]],
) -> None:
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with caminho_saida.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            [
                "modelo",
                "janela",
                "inicio_segundos",
                "fim_segundos",
                "probabilidade_violencia",
                "violencia",
            ]
        )

        for nome, dados in resultados_modelos.items():
            for resultado in dados["resultados"]:
                escritor.writerow(
                    [
                        nome,
                        resultado["janela"],
                        resultado["inicio"],
                        resultado["fim"],
                        resultado["probabilidade"],
                        resultado["violencia"],
                    ]
                )


def mostrar_resultados(
    resultados_modelos: dict[str, dict[str, Any]],
    limiar: float,
    esperado: str | None,
) -> None:
    for nome, dados in resultados_modelos.items():
        classe = (
            "violencia"
            if dados["maximo"] >= limiar
            else "nao_violencia"
        )

        correto = esperado is None or classe == esperado
        marcador = "OK" if correto else "DIVERGIU"

        print("\n" + "=" * 72)
        print("Modelo:", nome)
        print(
            "Resumo:",
            f"máximo={dados['maximo']:.6f}",
            f"média={dados['media']:.6f}",
            f"mediana={dados['mediana']:.6f}",
        )
        print(
            "Janelas positivas:",
            f"{dados['janelas_positivas']}/{dados['total_janelas']}",
        )
        print("Decisão pelo máximo:", classe, marcador)
        print("-" * 72)
        print("Cinco janelas com maior probabilidade:")

        melhores = sorted(
            dados["resultados"],
            key=lambda item: item["probabilidade"],
            reverse=True,
        )[:5]

        for item in melhores:
            print(
                f"{item['inicio']:7.2f}s–{item['fim']:7.2f}s  "
                f"prob={item['probabilidade']:.6f}  "
                f"{'VIOLÊNCIA' if item['violencia'] else 'não violência'}"
            )


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa um vídeo em janelas temporais e compara os modelos "
            "de violência antes e depois do ajuste fino."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Caminho do vídeo que será analisado.",
    )
    parser.add_argument(
        "--janela",
        type=float,
        default=3.0,
        help="Tamanho de cada janela em segundos. Padrão: 3.",
    )
    parser.add_argument(
        "--passo",
        type=float,
        default=1.5,
        help="Avanço entre janelas em segundos. Padrão: 1.5.",
    )
    parser.add_argument(
        "--limiar",
        type=float,
        default=0.70,
        help="Limiar de violência. Padrão: 0.50.",
    )
    parser.add_argument(
        "--esperado",
        choices=["violencia", "nao_violencia"],
        help="Classe esperada para conferir o teste.",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        choices=list(NOMES_MODELOS),
        default=list(NOMES_MODELOS),
        help="Modelos que serão comparados.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = criar_argumentos()
    configurar_gpu()

    if argumentos.janela <= 0 or argumentos.passo <= 0:
        raise ValueError("Janela e passo devem ser maiores que zero.")

    raiz = encontrar_raiz_projeto()
    caminho_video = argumentos.video.expanduser().resolve()

    if not caminho_video.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    metadados = obter_metadados_video(caminho_video)
    intervalos = criar_intervalos(
        duracao=metadados["duracao"],
        tamanho_janela=argumentos.janela,
        passo=argumentos.passo,
    )

    print("\nVídeo:", caminho_video)
    print(f"Duração: {metadados['duracao']:.2f}s")
    print(f"FPS: {metadados['fps']:.2f}")
    print("Janelas:", len(intervalos))
    print("TensorFlow GPU:", bool(tf.config.list_physical_devices("GPU")))
    print("Limiar:", argumentos.limiar)

    resultados_modelos: dict[str, dict[str, Any]] = {}

    for nome in argumentos.modelos:
        caminho_modelo = (
            raiz
            / "models"
            / "violence"
            / NOMES_MODELOS[nome]
        )

        if not caminho_modelo.exists():
            print(f"Aviso: modelo ausente, ignorado: {caminho_modelo}")
            continue

        resultados_modelos[nome] = testar_modelo(
            caminho_modelo=caminho_modelo,
            caminho_video=caminho_video,
            intervalos=intervalos,
            metadados=metadados,
            limiar=argumentos.limiar,
        )

    if not resultados_modelos:
        raise RuntimeError("Nenhum modelo pôde ser testado.")

    mostrar_resultados(
        resultados_modelos,
        argumentos.limiar,
        argumentos.esperado,
    )

    caminho_saida = (
        raiz
        / "outputs"
        / "metrics"
        / "violence"
        / "inferencia_janelas.csv"
    )
    salvar_csv(caminho_saida, resultados_modelos)

    print("\nRelatório por janela salvo em:", caminho_saida)


if __name__ == "__main__":
    main()
