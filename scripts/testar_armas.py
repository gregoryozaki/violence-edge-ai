import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


CLASSES_ESPERADAS = {
    "machete",
    "knife",
    "baseball_bat",
    "rifle",
    "gun",
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


def validar_modelo(modelo: YOLO) -> None:
    classes_modelo = set(modelo.names.values())

    if classes_modelo != CLASSES_ESPERADAS:
        raise RuntimeError(
            "As classes do modelo não correspondem às esperadas. "
            f"Encontradas: {sorted(classes_modelo)}"
        )


def serializar_deteccoes(
    resultado: Any,
) -> list[dict[str, Any]]:
    deteccoes: list[dict[str, Any]] = []

    if resultado.boxes is None:
        return deteccoes

    caixas = resultado.boxes.xyxy.cpu().tolist()
    confiancas = resultado.boxes.conf.cpu().tolist()
    classes = resultado.boxes.cls.cpu().tolist()

    for caixa, confianca, indice_classe in zip(
        caixas,
        confiancas,
        classes,
        strict=True,
    ):
        indice = int(indice_classe)

        deteccoes.append(
            {
                "classe": resultado.names[indice],
                "confianca": float(confianca),
                "bounding_box": {
                    "x1": float(caixa[0]),
                    "y1": float(caixa[1]),
                    "x2": float(caixa[2]),
                    "y2": float(caixa[3]),
                },
            }
        )

    return deteccoes


def mostrar_deteccoes(
    deteccoes: list[dict[str, Any]],
) -> None:
    if not deteccoes:
        print("Nenhuma arma detectada.")
        return

    print(f"Detecções: {len(deteccoes)}")

    for indice, deteccao in enumerate(deteccoes, start=1):
        caixa = deteccao["bounding_box"]

        print(
            f"[{indice}] "
            f"{deteccao['classe']} "
            f"conf={deteccao['confianca']:.4f} "
            f"box=({caixa['x1']:.1f}, {caixa['y1']:.1f}, "
            f"{caixa['x2']:.1f}, {caixa['y2']:.1f})"
        )


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Testa o detector de armas YOLO em uma imagem externa."
        )
    )
    parser.add_argument(
        "imagem",
        type=Path,
        help="Caminho da imagem que será analisada.",
    )
    parser.add_argument(
        "--confianca",
        type=float,
        default=0.50,
        help="Limiar mínimo de confiança. Padrão: 0.50.",
    )
    parser.add_argument(
        "--esperado",
        choices=[
            "machete",
            "knife",
            "baseball_bat",
            "rifle",
            "gun",
            "nenhuma",
        ],
        help="Classe esperada para conferir manualmente o resultado.",
    )
    parser.add_argument(
        "--dispositivo",
        default=None,
        help=(
            "Dispositivo Ultralytics, por exemplo 0 para GPU ou cpu. "
            "Se omitido, será escolhido automaticamente."
        ),
    )
    return parser.parse_args()


def main() -> None:
    argumentos = criar_argumentos()

    if not 0.0 <= argumentos.confianca <= 1.0:
        raise ValueError("A confiança deve estar entre 0 e 1.")

    raiz = encontrar_raiz_projeto()
    caminho_imagem = argumentos.imagem.expanduser().resolve()

    if not caminho_imagem.exists():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho_imagem}"
        )

    imagem = cv2.imread(str(caminho_imagem))

    if imagem is None:
        raise ValueError(
            f"Não foi possível abrir a imagem: {caminho_imagem}"
        )

    caminho_modelo = (
        raiz
        / "models"
        / "weapons"
        / "pytorch"
        / "best.pt"
    )

    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}"
        )

    modelo = YOLO(str(caminho_modelo))
    validar_modelo(modelo)

    inicio = time.perf_counter()

    resultados = modelo.predict(
        source=str(caminho_imagem),
        conf=argumentos.confianca,
        imgsz=960,
        device=argumentos.dispositivo,
        verbose=False,
    )

    tempo_inferencia = time.perf_counter() - inicio

    if not resultados:
        raise RuntimeError("O modelo não retornou resultados.")

    resultado = resultados[0]
    deteccoes = serializar_deteccoes(resultado)

    classes_detectadas = {
        deteccao["classe"]
        for deteccao in deteccoes
    }

    if argumentos.esperado == "nenhuma":
        correto = not deteccoes
    elif argumentos.esperado:
        correto = argumentos.esperado in classes_detectadas
    else:
        correto = None

    pasta_saida = (
        raiz
        / "outputs"
        / "predictions"
        / "weapons"
    )
    pasta_saida.mkdir(parents=True, exist_ok=True)

    nome_base = caminho_imagem.stem
    caminho_imagem_saida = (
        pasta_saida / f"{nome_base}_detectado.jpg"
    )
    caminho_json_saida = (
        pasta_saida / f"{nome_base}_resultado.json"
    )

    imagem_anotada = resultado.plot()
    cv2.imwrite(
        str(caminho_imagem_saida),
        imagem_anotada,
    )

    relatorio = {
        "imagem": str(caminho_imagem),
        "modelo": str(caminho_modelo),
        "confianca_minima": argumentos.confianca,
        "tempo_inferencia_segundos": tempo_inferencia,
        "classe_esperada": argumentos.esperado,
        "resultado_correto": correto,
        "deteccoes": deteccoes,
    }

    caminho_json_saida.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Imagem:", caminho_imagem)
    print("Modelo:", caminho_modelo)
    print("Classes:", modelo.names)
    print(f"Confiança mínima: {argumentos.confianca:.2f}")
    print(f"Tempo de inferência: {tempo_inferencia:.4f}s")
    print("-" * 72)

    mostrar_deteccoes(deteccoes)

    if correto is not None:
        print(
            "Resultado esperado:",
            argumentos.esperado,
            "OK" if correto else "DIVERGIU",
        )

    print("-" * 72)
    print("Imagem anotada:", caminho_imagem_saida)
    print("Relatório JSON:", caminho_json_saida)


if __name__ == "__main__":
    main()
