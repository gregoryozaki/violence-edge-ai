from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def detectar_frase_secreta(
    transcricao: str,
    frase_secreta: str | None,
    variantes_asr: Iterable[str] = (),
) -> bool:
    """Detecta a frase configurada sem depender do classificador NLP."""
    if not frase_secreta:
        return False

    texto = normalizar_texto(transcricao)
    candidatos = [frase_secreta, *variantes_asr]

    return any(
        normalizar_texto(candidato) in texto
        for candidato in candidatos
        if candidato.strip()
    )
