# Prompt Eval Workflow

A small custom evaluation pipeline for prompts, built while following the
[Claude with the Anthropic API](https://anthropic-partners.skilljar.com/claude-with-the-anthropic-api/287736) course.

The prompt under test asks Claude to answer AWS-related tasks with **only** Python, JSON or a regular
expression, with no commentary. The pipeline measures how well it does that.

## How it works

1. `dataset_generator.py` asks Claude to generate test cases (`task`, `format`, `solution_criteria`) and writes `dataset.json`.
2. `main.py` runs the prompt on every test case, then scores each output twice:
   - **Model grade**: a grader prompt returns strengths, weaknesses, reasoning and a 1-10 score against the case's criteria.
   - **Syntax grade**: 10 if the output parses as the expected format (`json`, `ast`, `re.compile`), otherwise 0.
3. The final score per case is the mean of the two; the run prints the average and writes `results.json`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in a .env file
```

## Usage

```sh
uv run dataset_generator.py   # regenerate dataset.json (optional, one is included)
uv run main.py                # run the eval
```

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Anthropic client, message helpers and the `chat` wrapper |
| `dataset_generator.py` | Generates the test dataset |
| `main.py` | Prompt runner, graders and eval loop |
| `dataset.json` | Generated test cases |

## Notes

- Runs on Claude Haiku (set in `client.py`).
- The API no longer accepts `temperature`, `top_p` or `top_k`, so `chat` does not expose them.

## License

Public domain ([Unlicense](LICENSE)). Use it however you like.
