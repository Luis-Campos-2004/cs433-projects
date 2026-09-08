#!/usr/bin/env bash
# Inicializa o repositório e liga-o a um remoto (GitHub ou GitLab da EPFL).
# Uso:  ./scripts/setup_repo.sh git@github.com:teu-user/ml-cs433.git
set -euo pipefail

REMOTE="${1:-}"

if [ ! -d .git ]; then
  git init -b main
  echo "repositório git inicializado"
fi

git add .
git commit -m "chore: estrutura inicial do repositório CS-433" || echo "nada para commitar"

if [ -n "$REMOTE" ]; then
  git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
  git push -u origin main
  echo "ligado a $REMOTE"
else
  echo "sem remoto. Cria o repo (PRIVADO) na web e corre:"
  echo "  git remote add origin <url>"
  echo "  git push -u origin main"
fi
