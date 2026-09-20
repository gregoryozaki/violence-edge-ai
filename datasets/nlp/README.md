# Dataset sintético NLP para detecção de violência

## Conteúdo

- 40.000 textos únicos em português.
- Classificação multirrótulo.
- Gírias, linguagem informal, palavrões e marcadores regionais.
- Erros simulados de transcrição do Whisper.
- Negativos difíceis com filmes, notícias, aulas, esportes, ferramentas e linguagem figurada.
- Divisão por grupo sem variações da mesma frase-base entre treino, validação e teste.

## Rótulos

- `pedido_socorro`
- `ameaca`
- `agressao`
- `referencia_arma`
- `nao_violento`

## Arquivos

- `dataset_nlp_sintetico.jsonl`: dataset completo.
- `splits/treino.jsonl`: conjunto de treino.
- `splits/validacao.jsonl`: conjunto de validação.
- `splits/teste.jsonl`: conjunto de teste.
- `relatorio_dataset.json`: distribuição e validações.
- `rotulos.json`: definição das classes.
- `amostras.json`: 100 exemplos para inspeção.
- `gerar_dataset_nlp.py`: script reproduzível.
- `detectar_frase_secreta.py`: regra separada do modelo NLP.
- `configuracao_frase_secreta.json`: configuração ainda sem frase definida.

## Executar novamente

```bash
python gerar_dataset_nlp.py \
  --saida datasets/nlp \
  --quantidade 40000 \
  --semente 42
```

## Observação crítica

O dataset é sintético. Ele serve para prototipagem e treinamento inicial, mas as métricas finais devem incluir textos reais ou transcrições reais anonimizadas. A frase secreta não é um rótulo do modelo: ela é verificada por uma regra independente e configurável.
