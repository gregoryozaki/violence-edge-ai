import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROTULOS_PERIGO = {
    "pedido_socorro",
    "ameaca",
    "agressao",
    "referencia_arma",
}

FRASES_DEMO = [
    "Socorro, ele está me batendo!",
    "Se você sair daqui eu vou te matar.",
    "Ele puxou uma arma e apontou para mim.",
    "Tem uma briga acontecendo aqui agora.",
    "A gente vai assistir um filme de ação hoje.",
    "A faca está guardada na cozinha.",
    "Me ajuda por favor, estou com medo.",
    "Está tudo bem, foi apenas uma discussão sobre futebol.",
]


def encontrar_raiz_projeto() -> Path:
    caminho_script = Path(__file__).resolve()

    for candidato in [caminho_script.parent, *caminho_script.parents]:
        if (
            (candidato / "models").exists()
            and (candidato / "datasets").exists()
            and (candidato / "scripts").exists()
        ):
            return candidato

    raise FileNotFoundError(
        "Não foi possível localizar a raiz do projeto violence-edge-ai."
    )


def extrair_limiares(conteudo: Any) -> dict[str, float]:
    if not isinstance(conteudo, dict):
        return {}

    for chave in ("limiares", "thresholds", "limiares_por_classe"):
        valor = conteudo.get(chave)

        if isinstance(valor, dict):
            conteudo = valor
            break

    limiares: dict[str, float] = {}

    for rotulo, valor in conteudo.items():
        try:
            limiares[str(rotulo)] = float(valor)
        except (TypeError, ValueError):
            continue

    return limiares


def carregar_limiares(
    caminho: Path,
    rotulos: list[str],
    limiar_padrao: float,
) -> dict[str, float]:
    limiares = {rotulo: limiar_padrao for rotulo in rotulos}

    if not caminho.exists():
        print(
            f"Aviso: {caminho.name} não encontrado. "
            f"Usando limiar padrão {limiar_padrao:.2f}."
        )
        return limiares

    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    salvos = extrair_limiares(conteudo)

    for rotulo in rotulos:
        if rotulo in salvos:
            limiares[rotulo] = salvos[rotulo]

    return limiares


class ClassificadorNLP:
    def __init__(
        self,
        pasta_modelo: Path,
        limiar_padrao: float = 0.50,
        comprimento_maximo: int = 64,
    ) -> None:
        if not pasta_modelo.exists():
            raise FileNotFoundError(
                f"Modelo NLP não encontrado: {pasta_modelo}"
            )

        self.dispositivo = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.comprimento_maximo = comprimento_maximo

        self.tokenizador = AutoTokenizer.from_pretrained(
            pasta_modelo,
            local_files_only=True,
        )
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            pasta_modelo,
            local_files_only=True,
        )
        self.modelo.to(self.dispositivo)
        self.modelo.eval()

        self.rotulos = [
            str(self.modelo.config.id2label[indice])
            for indice in range(self.modelo.config.num_labels)
        ]

        self.limiares = carregar_limiares(
            pasta_modelo / "limiares.json",
            self.rotulos,
            limiar_padrao,
        )

    @torch.inference_mode()
    def prever(self, texto: str) -> dict[str, Any]:
        texto = texto.strip()

        if not texto:
            raise ValueError("O texto não pode estar vazio.")

        entradas = self.tokenizador(
            texto,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.comprimento_maximo,
        )
        entradas = {
            chave: valor.to(self.dispositivo)
            for chave, valor in entradas.items()
        }

        logits = self.modelo(**entradas).logits[0]
        probabilidades = torch.sigmoid(logits).cpu().tolist()

        resultados = {
            rotulo: {
                "probabilidade": float(probabilidade),
                "limiar": float(self.limiares[rotulo]),
                "ativo": bool(
                    probabilidade >= self.limiares[rotulo]
                ),
            }
            for rotulo, probabilidade in zip(
                self.rotulos,
                probabilidades,
                strict=True,
            )
        }

        rotulos_ativos = [
            rotulo
            for rotulo, dados in resultados.items()
            if dados["ativo"]
        ]

        existe_perigo = any(
            rotulo in ROTULOS_PERIGO
            for rotulo in rotulos_ativos
        )

        if existe_perigo and "nao_violento" in rotulos_ativos:
            resultados["nao_violento"]["ativo"] = False
            rotulos_ativos.remove("nao_violento")

        return {
            "texto": texto,
            "dispositivo": str(self.dispositivo),
            "rotulos_ativos": rotulos_ativos,
            "resultados": resultados,
        }


def mostrar_resultado(resultado: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("Texto:", resultado["texto"])
    print("Dispositivo:", resultado["dispositivo"])

    ativos = resultado["rotulos_ativos"]
    print("Classificação:", ", ".join(ativos) if ativos else "nenhum rótulo")

    print("-" * 72)

    ordenados = sorted(
        resultado["resultados"].items(),
        key=lambda item: item[1]["probabilidade"],
        reverse=True,
    )

    for rotulo, dados in ordenados:
        marcador = "ATIVO" if dados["ativo"] else "inativo"

        print(
            f"{rotulo:20s} "
            f"prob={dados['probabilidade']:.4f} "
            f"limiar={dados['limiar']:.4f} "
            f"{marcador}"
        )


def executar_demo(classificador: ClassificadorNLP) -> None:
    for frase in FRASES_DEMO:
        mostrar_resultado(classificador.prever(frase))


def executar_interativo(classificador: ClassificadorNLP) -> None:
    print("\nDigite uma frase para classificar.")
    print("Pressione Enter sem texto para encerrar.")

    while True:
        texto = input("\nFrase: ").strip()

        if not texto:
            break

        mostrar_resultado(classificador.prever(texto))


def criar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa o modelo BERTimbau multirrótulo do projeto."
    )
    parser.add_argument(
        "texto",
        nargs="?",
        help="Frase que será classificada.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Executa um conjunto curto de frases de teste.",
    )
    parser.add_argument(
        "--limiar-padrao",
        type=float,
        default=0.50,
        help="Limiar usado caso limiares.json não tenha uma classe.",
    )
    return parser.parse_args()


def main() -> None:
    argumentos = criar_argumentos()
    raiz = encontrar_raiz_projeto()

    pasta_modelo = (
        raiz
        / "models"
        / "nlp"
        / "bertimbau_multirrotulo_regularizado"
    )

    classificador = ClassificadorNLP(
        pasta_modelo=pasta_modelo,
        limiar_padrao=argumentos.limiar_padrao,
    )

    print("Modelo:", pasta_modelo)
    print("Rótulos:", classificador.rotulos)
    print("GPU disponível:", torch.cuda.is_available())

    if argumentos.demo:
        executar_demo(classificador)
    elif argumentos.texto:
        mostrar_resultado(classificador.prever(argumentos.texto))
    else:
        executar_interativo(classificador)


if __name__ == "__main__":
    main()
