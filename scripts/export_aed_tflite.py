from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf


MODELO_KERAS = Path("models/aed/aed_multiclasse_v2_final.keras")

SAIDA_DIR = Path("edge_runtime/models/aed")
SAVEDMODEL_DIR = SAIDA_DIR / "aed_savedmodel_tmp"

SAIDA_FLOAT32 = SAIDA_DIR / "aed_float32.tflite"
SAIDA_FLOAT16 = SAIDA_DIR / "aed_float16.tflite"
SAIDA_DYNAMIC = SAIDA_DIR / "aed_dynamic.tflite"


def salvar_tflite(caminho: Path, conteudo: bytes):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(conteudo)
    tamanho_mb = caminho.stat().st_size / 1024 / 1024
    print(f"Salvo: {caminho} ({tamanho_mb:.2f} MB)")


def exportar_savedmodel(modelo):
    if SAVEDMODEL_DIR.exists():
        shutil.rmtree(SAVEDMODEL_DIR)

    print(f"\nExportando SavedModel temporário em: {SAVEDMODEL_DIR}")

    # Keras 3
    if hasattr(modelo, "export"):
        modelo.export(str(SAVEDMODEL_DIR))
    else:
        # fallback
        tf.saved_model.save(modelo, str(SAVEDMODEL_DIR))

    print("SavedModel exportado.")


def converter_float32():
    print("\nConvertendo AED para TFLite float32...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))
    tflite = converter.convert()
    salvar_tflite(SAIDA_FLOAT32, tflite)


def converter_float16():
    print("\nConvertendo AED para TFLite float16...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite = converter.convert()
    salvar_tflite(SAIDA_FLOAT16, tflite)


def converter_dynamic():
    print("\nConvertendo AED para TFLite dynamic range...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVEDMODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    salvar_tflite(SAIDA_DYNAMIC, tflite)


def testar_tflite(caminho_modelo: Path):
    print(f"\nTestando modelo TFLite: {caminho_modelo}")

    interpreter = tf.lite.Interpreter(model_path=str(caminho_modelo))
    interpreter.allocate_tensors()

    entradas = interpreter.get_input_details()
    saidas = interpreter.get_output_details()

    print("Entradas:")
    for entrada in entradas:
        print(
            f"  nome={entrada['name']} "
            f"shape={entrada['shape']} "
            f"dtype={entrada['dtype']}"
        )

    print("Saídas:")
    for saida in saidas:
        print(
            f"  nome={saida['name']} "
            f"shape={saida['shape']} "
            f"dtype={saida['dtype']}"
        )

    # Modelo AED tem 3 entradas:
    # log_mel:      (1, 64, 126, 1)
    # mfcc:         (1, 40, 126, 3)
    # estatisticas: (1, 8)
    for entrada in entradas:
        shape = entrada["shape"]
        dtype = entrada["dtype"]

        dado = np.random.random_sample(shape).astype(dtype)
        interpreter.set_tensor(entrada["index"], dado)

    interpreter.invoke()

    for saida in saidas:
        pred = interpreter.get_tensor(saida["index"])
        print("Predição teste:", pred)
        print("Shape saída:", pred.shape)


def main():
    if not MODELO_KERAS.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODELO_KERAS}")

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Carregando modelo AED: {MODELO_KERAS}")

    modelo = tf.keras.models.load_model(MODELO_KERAS, compile=False)

    print("\nResumo do modelo:")
    modelo.summary()

    exportar_savedmodel(modelo)

    converter_float32()
    converter_float16()
    converter_dynamic()

    testar_tflite(SAIDA_FLOAT32)
    testar_tflite(SAIDA_FLOAT16)
    testar_tflite(SAIDA_DYNAMIC)

    print("\nExportação AED finalizada com sucesso.")


if __name__ == "__main__":
    main()
