# Como trabalhamos (combinar entre os três)

## As cinco regras

1. **A `main` está sempre a funcionar.** Ninguém commita direto na `main`.
2. **Uma branch por tarefa.** `feat/ridge-regression`, `fix/data-split`, `report/intro`.
3. **Pull Request + 1 revisão** antes de fazer merge.
4. **Dados nunca entram no git.** Ficam em `data/`, que está no `.gitignore`.
5. **`pytest -q` verde** antes de abrir o PR.

## O ciclo, sempre igual

```bash
git checkout main          # 1. voltar à main
git pull                   # 2. trazer o trabalho dos outros
git checkout -b feat/nome  # 3. criar a minha branch
# ... trabalhar ...
git add .
git commit -m "add cross-validation for ridge"
git push -u origin feat/nome
# 4. abrir o Pull Request no site do GitHub
```

Depois do merge, apagar a branch e recomeçar do passo 1.

## Divisão de trabalho

Para não haver conflitos, cada pessoa fica dona de ficheiros diferentes.
Exemplo para o Projeto 1:

| Pessoa | Responsável por |
|---|---|
| A | `src/implementations.py` + `tests/` |
| B | `src/preprocessing.py` + feature engineering |
| C | `src/cross_validation.py` + `run.py` |

O relatório escreve-se no fim, em Overleaf, e só o PDF final entra no repo.

## Se der conflito

Não entres em pânico e não apagues nada:

```bash
git status                 # mostra que ficheiros estão em conflito
```

Abre o ficheiro. Vais ver marcas assim:

```
<<<<<<< HEAD
o meu código
=======
o código da outra pessoa
>>>>>>> main
```

Escolhe o que fica (às vezes é uma mistura dos dois), apaga as três linhas de
marcas, e depois:

```bash
git add <ficheiro>
git commit
```

Se estiver mesmo mau: `git merge --abort` desfaz tudo e voltas ao início.
