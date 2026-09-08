# CS-433 — Projetos (equipa)

Repositório **partilhado pelos três**. Privado.

| | Peso | Prazo |
|---|---|---|
| Projeto 1 | 10% | 31 Out |
| Projeto 2 | 30% | 18 Dez |

## Estrutura

```
projects/project1/
├── data/          # NÃO versionado
├── src/           # código: funções puras, testáveis
├── notebooks/     # exploração — a lógica não vive aqui
├── tests/         # testes unitários
├── report/        # relatório LaTeX
└── run.py         # reproduz a submissão de uma vez
```

## Arranque (cada pessoa faz isto uma vez)

```bash
git clone <url-do-repo>
cd cs433-projetos
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Regras

Ver `CONTRIBUTING.md`. Lê antes do primeiro commit.
