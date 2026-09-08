# How we work

## The five rules

1. **`main` always works.** Nobody commits directly to `main`.
2. **One branch per task.** `feat/ridge-regression`, `fix/data-split`, `report/intro`.
3. **Pull request + 1 review** before merging.
4. **Data never goes into git.** It lives in `data/`, which is gitignored.
5. **`pytest -q` green** before opening the PR.

## The cycle, always the same

```bash
git checkout main          # 1. back to the trunk
git pull                   # 2. fetch everyone else's work
git checkout -b feat/name  # 3. my own branch
# ... work ...
pytest -q
git add .
git commit -m "add cross-validation for ridge"
git push -u origin feat/name
gh pr create --fill        # 4. ask for review
```

After the merge, delete the branch and start again from step 1.

## What each command does

| Command | In plain words |
|---|---|
| `git pull` | download what the others did |
| `git checkout -b name` | make a copy where I can break things safely |
| `git add .` | mark what I want to save |
| `git commit -m "..."` | take a snapshot of the work |
| `git push` | send it to the cloud |
| `gh pr create` | ask someone to check it before it goes official |

## Splitting the work

To avoid conflicts, each person owns different files.
Fill this in at the first meeting:

| Person | Owns |
|---|---|
| | `src/implementations.py` and `tests/` |
| | `src/preprocessing.py` and feature engineering |
| | `src/cross_validation.py` and `run.py` |

The report gets written in Overleaf. Only the final PDF goes in the repo.

## When you hit a conflict

Don't panic and don't delete anything:

```bash
git status                 # shows which files are conflicted
```

Open the file. You'll see markers like:

```
<<<<<<< HEAD
my code
=======
the other person's code
>>>>>>> main
```

Pick what stays (sometimes a mix of both), delete the three marker lines, then:

```bash
git add <file>
git commit
```

If it's really bad: `git merge --abort` undoes everything and puts you back.
