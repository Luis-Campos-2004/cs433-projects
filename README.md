# CS-433 — Projects (team repo)

Shared by the three of us. Private.

| | Weight | Due |
|---|---|---|
| Project 1 | 0% | 29 Oct |
| Project 2 | 30% | 18 Dec |

## Layout

```
projects/project1/
├── data/          # NOT versioned
├── src/           # code: pure, testable functions
├── notebooks/     # exploration — logic doesn't live here
├── tests/         # unit tests
├── report/        # LaTeX report
└── run.py         # reproduces the submission in one go
```

## Setup (once per person)

```bash
git clone <repo-url>
cd cs433-projects
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q                        # 5 tests should pass
```

## Rules

See `CONTRIBUTING.md`. Read it before your first commit.
