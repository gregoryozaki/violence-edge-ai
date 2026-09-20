import yaml
from pathlib import Path
from tflite_model import ModeloTFLite


def main():
    with open("configs/edge.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    modelos = cfg["modelos"]

    for nome in ["aed", "violencia", "armas"]:
        caminho = Path(modelos[nome])
        print("\n========================================")
        print(f"Testando modelo: {nome}")
        print(f"Caminho: {caminho}")

        modelo = ModeloTFLite(caminho)
        saidas = modelo.teste_dummy()

        for i, saida in enumerate(saidas):
            print(f"Saida {i}: shape={saida.shape}, min={saida.min():.4f}, max={saida.max():.4f}")

    print("\nTodos os modelos TFLite carregaram e executaram.")


if __name__ == "__main__":
    main()
