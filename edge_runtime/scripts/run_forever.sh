#!/usr/bin/env bash

cd /home/gregory/projeto_ia_embarcada || exit 1
source .venv/bin/activate

mkdir -p logs

echo "Rotina de reinicio automatico iniciada."
echo "Logs em: logs/app.log"

while true; do
    echo "========================================" >> logs/app.log
    echo "Inicio: $(date)" >> logs/app.log

    python app/main.py >> logs/app.log 2>&1

    echo "Aplicacao caiu/terminou: $(date)" >> logs/app.log
    echo "Reiniciando em 5 segundos..." >> logs/app.log

    sleep 5
done
