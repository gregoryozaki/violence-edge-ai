from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROTULOS = (
    "pedido_socorro",
    "ameaca",
    "agressao",
    "referencia_arma",
    "nao_violento",
)

PERFIS_PADRAO = {
    "nao_violento": 14_000,
    "pedido_socorro": 4_000,
    "ameaca": 4_000,
    "agressao": 4_000,
    "referencia_arma": 3_000,
    "pedido_ameaca": 2_500,
    "pedido_agressao": 2_500,
    "pedido_arma": 2_000,
    "ameaca_arma": 1_500,
    "agressao_arma": 1_500,
    "multirrotulo": 1_000,
}


@dataclass(frozen=True)
class Configuracao:
    quantidade_total: int = 40_000
    proporcao_treino: float = 0.70
    proporcao_validacao: float = 0.15
    proporcao_teste: float = 0.15
    semente: int = 42
    maximo_tentativas_por_exemplo: int = 300


VOCATIVOS = [
    "", "mano", "mana", "cara", "moço", "moça", "meu deus",
    "pelo amor de deus", "gente", "alguém", "vizinho", "mãe",
    "pai", "amiga", "amigo", "senhora", "senhor",
]

MARCADORES_REGIONAIS = {
    "sem_marcador": [""],
    "norte": ["mana", "mano", "égua", "bora"],
    "nordeste": ["oxe", "oxente", "visse", "eita"],
    "sul": ["bah", "guri", "guria"],
    "urbano_geral": ["mano", "véi", "cara", "meu"],
}

ABERTURAS_NEUTRAS = [
    "", "olha", "tipo assim", "só comentando", "então", "pra explicar",
    "eu tava dizendo", "na real", "só pra avisar", "seguinte",
]

ABERTURAS_URGENCIA = [
    "", "escuta", "olha", "por favor", "presta atenção", "agora",
    "rápido", "eu tô falando sério", "não desliga", "fica na linha",
    "pelo amor de deus", "tem alguém aí", "ouve isso", "me escuta",
]

FECHAMENTOS_POSITIVOS = [
    "", "aqui em casa", "aqui na rua", "perto da porta", "no quarto",
    "na cozinha", "no quintal", "agora", "nesse momento", "vem logo",
    "não demora", "antes que piore", "eu tô com medo", "ninguém tá vendo",
    "não consigo sair", "a porta tá trancada", "eu tô sozinha",
]

PALAVROES = [
    "porra", "caralho", "merda", "desgraçado", "filho da puta",
    "puta que pariu",
]

PEDIDOS = [
    "socorro", "me ajuda", "me acode", "chama alguém", "chama a polícia",
    "liga pra polícia", "vem rápido", "não me deixa aqui", "me tira daqui",
    "abre a porta pra mim", "fica comigo", "preciso de ajuda",
    "manda alguém pra cá", "avisa meu irmão", "chama o vizinho",
]

SITUACOES_PEDIDO = [
    "eu tô presa aqui", "não consigo sair", "tem alguém me seguindo",
    "eu tô com muito medo", "não sei onde eu tô", "não consigo falar direito",
    "fica na linha comigo", "não desliga", "eu preciso sair daqui",
    "tem alguém na porta", "eu não tô segura", "vem me buscar",
]

AMEACAS = [
    "eu vou te matar", "tu vai se arrepender", "vou acabar contigo",
    "ninguém vai te encontrar", "eu vou quebrar tua cara",
    "se tu sair eu te pego", "vou fazer tu pagar", "vou te pegar lá fora",
    "isso não vai ficar assim", "eu vou atrás de você",
    "se falar com alguém vai piorar", "você não vai escapar",
    "eu vou acabar com tua vida", "quando eu te encontrar tu vai ver",
]

AMEACAS_RELATADAS = [
    "ele disse que vai me matar", "ela falou que vai acabar comigo",
    "o cara disse que vai me pegar", "ele jurou que vai me machucar",
    "ela ameaçou me encontrar na saída", "ele disse que ninguém vai me achar",
    "o homem falou que vai me fazer pagar", "ela disse que vai quebrar minha cara",
    "ele falou que se eu sair vai piorar", "ela disse que eu não vou escapar",
]

AGRESSOES = [
    "ele tá me batendo", "ela me deu um soco", "ele me empurrou no chão",
    "ela tá puxando meu cabelo", "ele não solta meu braço", "ela me chutou",
    "ele me deu um tapa", "ela tá me segurando à força",
    "ele me jogou contra a parede", "ela apertou meu pescoço",
    "ele me derrubou", "ela tá me arrastando", "ele me acertou de novo",
    "ela tá me prendendo no quarto", "ele segurou minha boca",
]

ARMAS = [
    "faca", "facão", "revólver", "pistola", "arma", "espingarda", "rifle",
    "taco", "bastão", "pedaço de pau", "martelo", "machado",
    "chave de fenda", "canivete", "barra de ferro", "cano de metal",
]

ARTIGO_ARMAS = {
    "faca": "uma", "facão": "um", "revólver": "um", "pistola": "uma",
    "arma": "uma", "espingarda": "uma", "rifle": "um", "taco": "um",
    "bastão": "um", "pedaço de pau": "um", "martelo": "um",
    "machado": "um", "chave de fenda": "uma", "canivete": "um",
    "barra de ferro": "uma", "cano de metal": "um",
}

REFERENCIAS_ARMA = [
    "ele tá com {objeto} na mão", "ela escondeu {objeto} na bolsa",
    "tem {objeto} em cima da mesa", "o cara mostrou {objeto}",
    "ele entrou aqui armado com {objeto}", "ela levantou {objeto} pra mim",
    "vi {objeto} no carro dele", "o homem tá segurando {objeto}",
    "ela trouxe {objeto} pra dentro de casa", "ele tá apontando {objeto}",
    "tem {objeto} atrás da porta", "o sujeito puxou {objeto}",
]


def formatar_arma(arma: str) -> str:
    return f"{ARTIGO_ARMAS[arma]} {arma}"

OBJETOS_BENIGNOS = [
    ("uma faca", "cortar legumes"), ("um facão", "limpar o terreno"),
    ("um martelo", "consertar a cadeira"),
    ("uma chave de fenda", "arrumar o ventilador"),
    ("um taco", "jogar beisebol"), ("um bastão", "fazer exercício"),
    ("um canivete", "abrir uma embalagem"), ("um machado", "cortar lenha"),
    ("um rifle antigo", "ficar em exposição no museu"),
    ("uma arma de brinquedo", "usar numa fantasia"),
]

MIDIAS = ["o filme", "a série", "a novela", "o documentário", "o jogo", "a notícia", "o livro"]
ACOES_FICCIONAIS = [
    "uma cena de tiro", "uma briga", "um personagem pedindo socorro",
    "uma ameaça", "uma perseguição", "uma faca cenográfica",
    "uma arma falsa", "uma discussão pesada",
]

FRASES_FIGURADAS = [
    "vou matar a saudade quando chegar", "esse trabalho tá me matando",
    "essa piada vai me matar de rir", "morri de vergonha nessa apresentação",
    "esse calor tá me matando", "vou acabar com esse prato de comida",
    "o time matou o jogo no segundo tempo", "a prova acabou comigo",
]

FALAS_COTIDIANAS = [
    "esse jogo tá bom pra caralho", "que porra de tarefa difícil",
    "tô puto porque a internet caiu", "perdi o ônibus e fiquei com raiva",
    "o vizinho tá fazendo barulho de novo", "minha irmã bateu a porta forte",
    "a panela caiu no chão", "o cachorro começou a latir",
    "o pessoal tá discutindo futebol", "a criança gritou brincando",
]

CONTEXTOS_SEGUROS = [
    "mas tá tudo bem", "foi só uma história", "era parte do filme",
    "não aconteceu nada aqui", "foi só no jogo", "isso já passou",
    "ninguém está em perigo", "era uma encenação", "foi uma aula",
    "foi ontem e já foi resolvido", "era uma demonstração", "era de brinquedo",
]

SUBSTITUICOES_INFORMAIS = {
    r"\bvocê\b": ["vc", "cê"], r"\bestá\b": ["tá"], r"\bestou\b": ["tô"],
    r"\bpara\b": ["pra"], r"\bporque\b": ["pq"], r"\bque\b": ["q"],
    r"\btambém\b": ["tbm"], r"\bnão\b": ["nao", "n"],
    r"\bcomigo\b": ["cmg"], r"\balguém\b": ["algm"],
    r"\bpolícia\b": ["policia"],
}

SUBSTITUICOES_ASR = {
    "socorro": ["só corro", "socoro"], "me ajuda": ["me ajude", "m ajuda"],
    "me acode": ["me acorde", "me acodi"],
    "chama a polícia": ["chama polícia", "chama a policia"],
    "ele vai me matar": ["ele vai mim matar", "ele vai me mata"],
    "tá me batendo": ["ta me batendo", "tá mim batendo"],
    "não me deixa": ["nao me deixa", "num me deixa"],
    "revólver": ["revolver"], "facão": ["facao"], "bastão": ["bastao"],
}


def normalizar_espacos(texto: str) -> str:
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+([,.;!?])", r"\1", texto)
    return texto.strip(" ,.;")


def remover_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def juntar(*partes: str) -> str:
    return normalizar_espacos(" ".join(p for p in partes if p))


def rotulos_ativos(*ativos: str) -> dict[str, int]:
    rotulos = {rotulo: 0 for rotulo in ROTULOS}
    for rotulo in ativos:
        rotulos[rotulo] = 1
    if ativos != ("nao_violento",):
        rotulos["nao_violento"] = 0
    return rotulos


def escolher_marcador(gerador: random.Random) -> tuple[str, str]:
    regioes = list(MARCADORES_REGIONAIS)
    regiao = gerador.choices(regioes, weights=[45, 15, 15, 10, 15], k=1)[0]
    return regiao, gerador.choice(MARCADORES_REGIONAIS[regiao])


def aplicar_informalidade(texto: str, gerador: random.Random) -> tuple[str, bool]:
    alterou = False
    for padrao, opcoes in SUBSTITUICOES_INFORMAIS.items():
        if re.search(padrao, texto, re.IGNORECASE) and gerador.random() < 0.45:
            texto = re.sub(padrao, gerador.choice(opcoes), texto, flags=re.IGNORECASE)
            alterou = True
    if gerador.random() < 0.15:
        texto = remover_acentos(texto)
        alterou = True
    return normalizar_espacos(texto.lower()), alterou


def aplicar_ruido_asr(texto: str, gerador: random.Random) -> tuple[str, bool]:
    if gerador.random() >= 0.22:
        return texto, False
    alterou = False
    candidatos = [x for x in SUBSTITUICOES_ASR if x in texto.lower()]
    if candidatos:
        original = gerador.choice(candidatos)
        texto = re.sub(
            re.escape(original), gerador.choice(SUBSTITUICOES_ASR[original]),
            texto, count=1, flags=re.IGNORECASE,
        )
        alterou = True
    palavras = texto.split()
    if len(palavras) > 6 and gerador.random() < 0.35:
        palavras.pop(gerador.randrange(1, len(palavras) - 1))
        texto = " ".join(palavras)
        alterou = True
    if gerador.random() < 0.55:
        texto = re.sub(r"[,.!?;:]", "", texto)
        alterou = True
    return normalizar_espacos(texto), alterou


def estilizar(
    texto_base: str,
    gerador: random.Random,
    chance_palavrao: float,
    positivo: bool,
) -> tuple[str, dict]:
    regiao, marcador = escolher_marcador(gerador)
    aberturas = ABERTURAS_URGENCIA if positivo else ABERTURAS_NEUTRAS
    abertura = gerador.choice(aberturas)
    texto = juntar(marcador if gerador.random() < 0.65 else "", abertura, texto_base)
    possui_palavrao = gerador.random() < chance_palavrao
    if possui_palavrao:
        palavrao = gerador.choice(PALAVROES)
        texto = juntar(palavrao, texto) if gerador.random() < 0.5 else juntar(texto, palavrao)
    texto, informal = aplicar_informalidade(texto, gerador)
    texto, ruido_asr = aplicar_ruido_asr(texto, gerador)
    registro = "asr_ruidoso" if ruido_asr else (
        "regional" if regiao != "sem_marcador" else (
            "informal" if informal or possui_palavrao else "formal"
        )
    )
    return texto, {
        "registro": registro,
        "regiao": regiao,
        "possui_palavrao": possui_palavrao,
        "ruido_asr": ruido_asr,
    }


def gerar_nao_violento(g: random.Random):
    tipo = g.choice(["objeto", "midia", "figurado", "cotidiano", "academico"])
    if tipo == "objeto":
        objeto, uso = g.choice(OBJETOS_BENIGNOS)
        base = g.choice([
            f"comprei {objeto} para {uso}", f"ele usou {objeto} para {uso}",
            f"{objeto.capitalize()} ficou guardado depois de {uso}",
        ])
    elif tipo == "midia":
        base = f"{g.choice(MIDIAS)} mostrou {g.choice(ACOES_FICCIONAIS)}"
    elif tipo == "figurado":
        base = g.choice(FRASES_FIGURADAS)
    elif tipo == "academico":
        base = g.choice([
            "estou estudando violência doméstica", "o professor explicou o que é ameaça",
            "a palavra socorro apareceu no exercício", "a notícia falou de uma agressão antiga",
            "o relatório cita armas apreendidas", "a aula discutiu prevenção de conflitos",
        ])
    else:
        base = g.choice(FALAS_COTIDIANAS)
    base = juntar(base, g.choice(CONTEXTOS_SEGUROS))
    return base, rotulos_ativos("nao_violento"), tipo


def gerar_pedido(g: random.Random):
    base = juntar(g.choice(VOCATIVOS), g.choice(PEDIDOS), g.choice(SITUACOES_PEDIDO), g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("pedido_socorro"), "pedido"


def gerar_ameaca(g: random.Random):
    return juntar(g.choice(AMEACAS + AMEACAS_RELATADAS), g.choice(FECHAMENTOS_POSITIVOS)), rotulos_ativos("ameaca"), "ameaca"


def gerar_agressao(g: random.Random):
    return juntar(g.choice(AGRESSOES), g.choice(FECHAMENTOS_POSITIVOS)), rotulos_ativos("agressao"), "agressao"


def gerar_arma(g: random.Random):
    arma = g.choice(ARMAS)
    return juntar(g.choice(REFERENCIAS_ARMA).format(objeto=formatar_arma(arma)), g.choice(FECHAMENTOS_POSITIVOS)), rotulos_ativos("referencia_arma"), "arma"


def gerar_pedido_ameaca(g: random.Random):
    base = juntar(g.choice(PEDIDOS), g.choice(AMEACAS_RELATADAS), g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("pedido_socorro", "ameaca"), "pedido_ameaca"


def gerar_pedido_agressao(g: random.Random):
    base = juntar(g.choice(PEDIDOS), g.choice(AGRESSOES), g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("pedido_socorro", "agressao"), "pedido_agressao"


def gerar_pedido_arma(g: random.Random):
    arma = g.choice(ARMAS)
    base = juntar(g.choice(PEDIDOS), g.choice(REFERENCIAS_ARMA).format(objeto=formatar_arma(arma)), g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("pedido_socorro", "referencia_arma"), "pedido_arma"


def gerar_ameaca_arma(g: random.Random):
    arma = g.choice(ARMAS)
    base = juntar(g.choice(REFERENCIAS_ARMA).format(objeto=formatar_arma(arma)), g.choice(AMEACAS_RELATADAS), g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("ameaca", "referencia_arma"), "ameaca_arma"


def gerar_agressao_arma(g: random.Random):
    arma = g.choice(ARMAS)
    objeto = formatar_arma(arma)
    complemento = g.choice([
        f"ele usou {objeto}", f"ela tá com {objeto}",
        f"tem {objeto} aqui", f"ele veio com {objeto}",
    ])
    base = juntar(g.choice(AGRESSOES), complemento, g.choice(FECHAMENTOS_POSITIVOS))
    return base, rotulos_ativos("agressao", "referencia_arma"), "agressao_arma"


def gerar_multirrotulo(g: random.Random):
    pedido, agressao, ameaca = g.choice(PEDIDOS), g.choice(AGRESSOES), g.choice(AMEACAS_RELATADAS)
    referencia = g.choice(REFERENCIAS_ARMA).format(objeto=formatar_arma(g.choice(ARMAS)))
    opcoes = [
        (juntar(pedido, agressao, referencia), ("pedido_socorro", "agressao", "referencia_arma")),
        (juntar(pedido, ameaca, referencia), ("pedido_socorro", "ameaca", "referencia_arma")),
        (juntar(pedido, agressao, ameaca), ("pedido_socorro", "agressao", "ameaca")),
        (juntar(pedido, agressao, ameaca, referencia), ("pedido_socorro", "agressao", "ameaca", "referencia_arma")),
    ]
    base, ativos = g.choice(opcoes)
    return juntar(base, g.choice(FECHAMENTOS_POSITIVOS)), rotulos_ativos(*ativos), "multirrotulo"


GERADORES: dict[str, Callable] = {
    "nao_violento": gerar_nao_violento,
    "pedido_socorro": gerar_pedido,
    "ameaca": gerar_ameaca,
    "agressao": gerar_agressao,
    "referencia_arma": gerar_arma,
    "pedido_ameaca": gerar_pedido_ameaca,
    "pedido_agressao": gerar_pedido_agressao,
    "pedido_arma": gerar_pedido_arma,
    "ameaca_arma": gerar_ameaca_arma,
    "agressao_arma": gerar_agressao_arma,
    "multirrotulo": gerar_multirrotulo,
}


def criar_hash(*partes: str, tamanho: int = 16) -> str:
    return hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:tamanho]


def ajustar_quantidades(total: int) -> dict[str, int]:
    total_padrao = sum(PERFIS_PADRAO.values())
    quantidades = {p: int(total * q / total_padrao) for p, q in PERFIS_PADRAO.items()}
    faltam = total - sum(quantidades.values())
    perfis = list(quantidades)
    for i in range(faltam):
        quantidades[perfis[i % len(perfis)]] += 1
    return quantidades


def gerar_exemplos(config: Configuracao) -> list[dict]:
    g = random.Random(config.semente)
    quantidades = ajustar_quantidades(config.quantidade_total)
    exemplos, textos_usados = [], set()

    for perfil, quantidade in quantidades.items():
        criados, tentativas = 0, 0
        limite = quantidade * config.maximo_tentativas_por_exemplo

        while criados < quantidade:
            tentativas += 1
            if tentativas > limite:
                raise RuntimeError(f"Falha ao gerar exemplos únicos para {perfil}: {criados}/{quantidade}")

            texto_base, rotulos, subtipo = GERADORES[perfil](g)
            chance_palavrao = 0.05 if perfil == "nao_violento" else 0.20
            texto, meta = estilizar(
                texto_base,
                g,
                chance_palavrao,
                positivo=(perfil != "nao_violento"),
            )
            chave = texto.casefold()

            if len(texto.split()) < 2 or chave in textos_usados:
                continue

            textos_usados.add(chave)
            exemplos.append({
                "id": criar_hash(perfil, texto),
                "texto": texto,
                "rotulos": rotulos,
                "perfil": perfil,
                "subtipo": subtipo,
                "registro": meta["registro"],
                "regiao": meta["regiao"],
                "possui_palavrao": meta["possui_palavrao"],
                "ruido_asr": meta["ruido_asr"],
                "origem": "sintetico",
                "grupo_id": criar_hash(perfil, texto_base.lower(), tamanho=12),
            })
            criados += 1

        print(f"{perfil:20s}: {criados:6d}")

    g.shuffle(exemplos)
    return exemplos


def dividir(exemplos: list[dict], config: Configuracao) -> dict[str, list[dict]]:
    """Divide por grupo sem colocar variações da mesma frase-base em splits diferentes."""
    grupos_por_perfil: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for exemplo in exemplos:
        grupos_por_perfil[exemplo["perfil"]][exemplo["grupo_id"]].append(exemplo)

    splits = {"treino": [], "validacao": [], "teste": []}
    g = random.Random(config.semente)

    for grupos in grupos_por_perfil.values():
        lista_grupos = list(grupos.values())
        g.shuffle(lista_grupos)
        total = sum(len(grupo) for grupo in lista_grupos)
        alvo_treino = total * config.proporcao_treino
        alvo_validacao = total * config.proporcao_validacao
        acumulado_treino = 0
        acumulado_validacao = 0

        for grupo in lista_grupos:
            if acumulado_treino < alvo_treino:
                splits["treino"].extend(grupo)
                acumulado_treino += len(grupo)
            elif acumulado_validacao < alvo_validacao:
                splits["validacao"].extend(grupo)
                acumulado_validacao += len(grupo)
            else:
                splits["teste"].extend(grupo)

    for itens in splits.values():
        g.shuffle(itens)
    return splits


def salvar_jsonl(caminho: Path, registros: list[dict]) -> None:
    with caminho.open("w", encoding="utf-8") as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def contar_rotulos(registros: list[dict]) -> dict[str, int]:
    contagem = Counter()
    for registro in registros:
        for rotulo, ativo in registro["rotulos"].items():
            if ativo:
                contagem[rotulo] += 1
    return {rotulo: contagem[rotulo] for rotulo in ROTULOS}


def validar(completo: list[dict], splits: dict[str, list[dict]]) -> dict:
    textos = [x["texto"].casefold() for x in completo]
    if len(textos) != len(set(textos)):
        raise RuntimeError("Foram encontrados textos duplicados.")

    ids = {nome: {x["id"] for x in dados} for nome, dados in splits.items()}
    grupos = {nome: {x["grupo_id"] for x in dados} for nome, dados in splits.items()}
    sobreposicoes = {
        "treino_validacao": len(ids["treino"] & ids["validacao"]),
        "treino_teste": len(ids["treino"] & ids["teste"]),
        "validacao_teste": len(ids["validacao"] & ids["teste"]),
    }
    sobreposicoes_grupos = {
        "treino_validacao": len(grupos["treino"] & grupos["validacao"]),
        "treino_teste": len(grupos["treino"] & grupos["teste"]),
        "validacao_teste": len(grupos["validacao"] & grupos["teste"]),
    }
    if any(sobreposicoes.values()) or any(sobreposicoes_grupos.values()):
        raise RuntimeError(
            f"Há sobreposição entre splits: ids={sobreposicoes}, grupos={sobreposicoes_grupos}"
        )

    conflitos = 0
    for registro in completo:
        r = registro["rotulos"]
        positivos = sum(r[x] for x in ROTULOS if x != "nao_violento")
        if (r["nao_violento"] and positivos) or (not r["nao_violento"] and positivos == 0):
            conflitos += 1
    if conflitos:
        raise RuntimeError(f"Há {conflitos} conflitos de rótulos.")

    return {
        "quantidade_total": len(completo),
        "duplicatas": 0,
        "conflitos_rotulos": 0,
        "sobreposicoes": sobreposicoes,
        "sobreposicoes_grupos": sobreposicoes_grupos,
        "splits": {
            nome: {
                "quantidade": len(dados),
                "rotulos": contar_rotulos(dados),
                "perfis": dict(Counter(x["perfil"] for x in dados)),
                "registros": dict(Counter(x["registro"] for x in dados)),
                "regioes": dict(Counter(x["regiao"] for x in dados)),
                "com_palavrao": sum(x["possui_palavrao"] for x in dados),
                "com_ruido_asr": sum(x["ruido_asr"] for x in dados),
            }
            for nome, dados in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintético multirrótulo para NLP de violência.")
    parser.add_argument("--saida", type=Path, default=Path("datasets/nlp"))
    parser.add_argument("--quantidade", type=int, default=40_000)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    config = Configuracao(quantidade_total=args.quantidade, semente=args.semente)
    if abs(config.proporcao_treino + config.proporcao_validacao + config.proporcao_teste - 1.0) > 1e-9:
        raise ValueError("As proporções devem somar 1.")

    print("Gerando dataset...")
    exemplos = gerar_exemplos(config)
    print("\nDividindo...")
    splits = dividir(exemplos, config)
    print("Validando...")
    relatorio = validar(exemplos, splits)

    args.saida.mkdir(parents=True, exist_ok=True)
    pasta_splits = args.saida / "splits"
    pasta_splits.mkdir(parents=True, exist_ok=True)

    salvar_jsonl(args.saida / "dataset_nlp_sintetico.jsonl", exemplos)
    for nome, dados in splits.items():
        salvar_jsonl(pasta_splits / f"{nome}.jsonl", dados)

    (args.saida / "rotulos.json").write_text(json.dumps({
        "rotulos": list(ROTULOS),
        "tipo": "classificacao_multirrotulo",
        "observacao": "A frase secreta deve ser tratada por regra configurável separada do modelo NLP.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.saida / "relatorio_dataset.json").write_text(json.dumps({
        "configuracao": asdict(config), **relatorio,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.saida / "amostras.json").write_text(
        json.dumps(exemplos[:100], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDataset criado com sucesso.")
    print("Saída:", args.saida.resolve())
    for nome, dados in splits.items():
        print(f"{nome:10s}: {len(dados)}")
    print("Duplicatas:", relatorio["duplicatas"])
    print("Conflitos:", relatorio["conflitos_rotulos"])


if __name__ == "__main__":
    main()
