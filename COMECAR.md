# Arrancar — instruções para o grupo

Lê tudo antes de escrever o primeiro comando. Demora uns 15 minutos.

---

## 1. Preparar o computador (uma vez na vida)

### Instalar o Git

Abre o Terminal e escreve:

```bash
git --version
```

Se aparecer um número, já tens. Se não:

- **macOS** — aparece uma janela a propor instalar as "Command Line Tools". Aceita.
- **Linux** — `sudo apt install git`
- **Windows** — instala de https://git-scm.com e usa sempre o **Git Bash**
  (que vem incluído), nunca o `cmd` nem o PowerShell.

### Dizer ao Git quem és

```bash
git config --global user.name "O Teu Nome"
git config --global user.email "o.teu.email@epfl.ch"
git config --global init.defaultBranch main
```

### Conta no GitHub

Cria conta em https://github.com com o email da EPFL.
Depois ativa o Student Pack: https://education.github.com/pack
(grátis com email da escola, dá repositórios privados sem limite).

**Manda o teu nome de utilizador do GitHub ao grupo.** Sem isso ninguém
te pode dar acesso.

### Instalar o `gh`

É a ferramenta que evita a chatice das chaves SSH.

- **macOS** — `brew install gh`
- **Linux** — `sudo apt install gh`
- **Windows** — `winget install GitHub.cli`

E autentica:

```bash
gh auth login
```

Responde: `GitHub.com` → `HTTPS` → `Y` → `Login with a web browser`.
Aparece um código de 8 caracteres. Copia, carrega Enter, cola no browser.
Nunca mais te pede a palavra-passe.

---

## 2. Descarregar os dois repositórios

Escolhe uma pasta onde vais guardar tudo o resto do semestre:

```bash
mkdir -p ~/epfl/cs433
cd ~/epfl/cs433

gh repo clone SUBSTITUIR-PELO-USER/cs433-labs
gh repo clone SUBSTITUIR-PELO-USER/cs433-projetos
```

Se der erro de permissão, ainda não te aceitaram o convite — vai ao email
e aceita, ou pede ao grupo.

---

## 3. Preparar o repositório dos labs

```bash
cd ~/epfl/cs433/cs433-labs

python -m venv .venv
source .venv/bin/activate          # Windows: source .venv/Scripts/activate
pip install -r requirements.txt

echo "o-teu-nome" > .eu            # exatamente o nome da tua pasta!
```

O `.eu` diz aos scripts quem tu és. Cada pessoa tem o seu, e não vai para
o git. Se escreveres o nome errado, os scripts vão pôr o teu trabalho na
pasta de outra pessoa.

Confirma que a tua pasta existe:

```bash
ls
```

Deves ver o teu nome na lista. Se não vires, avisa o grupo.

---

## 4. Preparar o repositório dos projetos

```bash
cd ~/epfl/cs433/cs433-projetos

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q
```

Devem passar 5 testes. Se passarem, está tudo bem instalado.

---

## 5. Como trabalhar nos LABS

Regra única: **cada um só escreve na sua pasta.** Podes ler as dos outros
à vontade, mas não editas.

Como não há dois a mexer no mesmo ficheiro, não precisas de branches.

### Começar um lab

```bash
cd ~/epfl/cs433/cs433-labs
source .venv/bin/activate
git pull                          # SEMPRE primeiro
./scripts/novo_lab.sh 04          # copia o enunciado para a tua pasta
```

Trabalha dentro de `o-teu-nome/lab04/`.

### Guardar o trabalho

```bash
git pull                          # outra vez, antes de enviar
git add o-teu-nome/
git commit -m "lab04: cross-validation"
git push
```

Faz isto no fim de cada sessão, mesmo com o trabalho a meio.

### Ver o trabalho dos outros

```bash
git pull
ls nome-do-colega/lab04/                       # espreitar
./scripts/comparar.sh 04 nome-do-colega        # o que difere da minha versão
./scripts/comparar.sh 04 nome-do-colega ex.py  # ver linha a linha
```

---

## 6. Como trabalhar nos PROJETOS

Aqui é diferente: os três mexem nos mesmos ficheiros. Por isso há regras.

**A `main` está sempre a funcionar. Ninguém commita direto na `main`.**

O ciclo é sempre este, do princípio ao fim:

```bash
cd ~/epfl/cs433/cs433-projetos
source .venv/bin/activate

git checkout main                  # 1. voltar ao tronco
git pull                           # 2. trazer o trabalho dos outros
git checkout -b feat/ridge         # 3. criar a minha linha paralela

# ... trabalhar ...

pytest -q                          # 4. testes verdes antes de enviar
git add .
git commit -m "add ridge regression"
git push -u origin feat/ridge      # 5. enviar
gh pr create --fill                # 6. pedir revisão
```

Depois um colega lê no GitHub, aprova e faz merge. Tu voltas ao passo 1.

### O que cada passo faz

| Comando | Em português |
|---|---|
| `git pull` | descarrega o que os outros fizeram |
| `git checkout -b nome` | cria uma cópia onde posso estragar sem afetar ninguém |
| `git add .` | marcar o que quero guardar |
| `git commit -m "..."` | tirar uma fotografia do trabalho |
| `git push` | enviar para a nuvem |
| `gh pr create` | pedir que alguém confira antes de entrar no oficial |

### Divisão de ficheiros

Para não haver conflitos, cada pessoa é dona de ficheiros diferentes.
Preencher na primeira reunião:

| Pessoa | Ficheiros |
|---|---|
| | `src/implementations.py` e `tests/` |
| | `src/preprocessing.py` |
| | `src/cross_validation.py` e `run.py` |

O relatório escreve-se em Overleaf. Só o PDF final entra no repositório.

---

## 7. Coisas que nunca se fazem

- **Nunca** metas ficheiros de dados no git (`.csv`, `.npy`, datasets).
  O `.gitignore` já bloqueia, mas não forces.
- **Nunca** commites direto na `main` do repositório dos projetos.
- **Nunca** edites a pasta de outra pessoa no repositório dos labs.
- **Nunca** tornes os repositórios públicos.

---

## 8. Quando corre mal

### "Estraguei tudo, quero voltar atrás"

```bash
git status                  # primeiro percebe onde estás
git restore <ficheiro>      # desfaz alterações não guardadas num ficheiro
git restore .               # desfaz tudo o que não está commitado
```

### Conflito ao fazer pull ou merge

O Git avisa e mostra o ficheiro. Abre-o e vais ver:

```
<<<<<<< HEAD
o meu código
=======
o código do outro
>>>>>>> main
```

Decides o que fica (às vezes é uma mistura dos dois), apagas as três linhas
de marcas (`<<<`, `===`, `>>>`), e depois:

```bash
git add <ficheiro>
git commit
```

Se estiver mesmo mau, `git merge --abort` desfaz tudo e voltas ao início.

### "Não sei o que se passa"

`git status` responde a 90% das perguntas. Lê o que ele diz — normalmente
sugere o comando seguinte.
