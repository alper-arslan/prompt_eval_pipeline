"""
Our prompt needs to assist users in writing three specific types of output for AWS use cases:

Python code
JSON configuration files
Regular expressions
The key requirement is that when a user requests help with a task,
we return clean output in one of these formats without any extra explanations, headers, or footers.
"""
import json
import re
import ast
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import mean

from anthropic import AuthenticationError, PermissionDeniedError

from client import add_user_message, chat, add_assistant_message


DEFAULT_PROMPT_TEMPLATE = """Please solve the following task:
{task}

* The required output format is: {format}
* Output exactly one code block containing only the answer, nothing else
* For regex, the answer is a single-line pattern only: no code, no function, no imports
* Do not add any comments, docstrings, commentary or explanations
"""

RUNS_DIR = "runs"


def run_prompt(test_case, template=DEFAULT_PROMPT_TEMPLATE):
    """Run the prompt under test on one test case and return the raw model output.

    The template's {task} and {format} placeholders are filled from the test case.
    The reply is prefilled with a code fence for the case's format and stopped at
    the closing fence, so the output holds only the answer itself.
    """
    prompt = template.replace("{task}", test_case["task"]).replace("{format}", test_case["format"])
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, f"```{test_case['format']}")
    output = chat(messages, stop_sequences=["```"])
    return output

def grade_by_model(test_case, output):
    """Have the model grade an output against the case's solution criteria.

    Returns the parsed JSON grade with strengths, weaknesses, reasoning and a
    1-10 score. Retries up to 3 times if the reply is not valid JSON and raises
    ValueError if every attempt fails.
    """
    eval_prompt = f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Solution to Evaluate:
<solution>
{output}
</solution>

Criteria you should use to evaluate the solution:
<criteria>
{test_case["solution_criteria"]}
</criteria>

Output Format
Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with JSON. Keep your response concise and direct.
Your response must be valid JSON. Escape every backslash as \\\\ (for example, write a regex \\d as \\\\d).
Example response shape:
{{
    "strengths": string[],
    "weaknesses": string[],
    "reasoning": string,
    "score": number
}}
    """

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    for _ in range(3):
        eval_text = chat(messages, stop_sequences=["```"])
        try:
            return json.loads(eval_text)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Grader returned invalid JSON after 3 attempts:\n{eval_text}")

def validate_json(text):
    """Return 10 if text parses as JSON, otherwise 0."""
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    """Return 10 if text is syntactically valid Python, otherwise 0."""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    """Return 10 if text is a single-line pattern that compiles as a regex, otherwise 0."""
    pattern = text.strip()
    # re.compile accepts almost any string, so also require a bare single-line pattern
    if "\n" in pattern:
        return 0
    try:
        re.compile(pattern)
        return 10
    except re.error:
        return 0

def grade_syntax(resp, test_case):
    """Score the output with the validator matching the case's format (0 or 10).

    Raises ValueError for a format other than json, python or regex.
    """
    fmt = test_case["format"]
    if fmt == "json":
        return validate_json(resp)
    elif fmt == "python":
        return validate_python(resp)
    elif fmt == "regex":
        return validate_regex(resp)
    raise ValueError(f"Unknown format in test case: {fmt!r}")

def run_test_case(test_case, template=DEFAULT_PROMPT_TEMPLATE):
    """Run one test case and return its output, grades and final score.

    The final score is the mean of the model grade and the syntax grade.
    """
    output = run_prompt(test_case, template)
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)
    score = (model_score + syntax_score) / 2
    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "model_score": model_score,
        "syntax_score": syntax_score,
        "reasoning": reasoning,
        "strengths": model_grade.get("strengths", []),
        "weaknesses": model_grade.get("weaknesses", []),
    }

def run_eval(dataset, template=DEFAULT_PROMPT_TEMPLATE, on_progress=None, max_workers=4):
    """Run every test case in the dataset and return the results in dataset order.

    Cases run in parallel on max_workers threads. on_progress(done, total, result)
    is called after each case finishes, with result None if that case failed.
    A failed case is skipped with a message, except authentication errors,
    which stop the whole run.
    """
    slots = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_test_case, case, template): i for i, case in enumerate(dataset)}
        for done, future in enumerate(as_completed(futures), start=1):
            i = futures[future]
            try:
                slots[i] = future.result()
            except (AuthenticationError, PermissionDeniedError):
                for pending in futures:
                    pending.cancel()
                raise
            except Exception as e:
                print(f"Skipping failed test case {dataset[i]['task'][:60]!r}: {e}")
            if on_progress:
                on_progress(done, len(dataset), slots[i])

    results = [r for r in slots if r is not None]
    if results:
        average_score = mean([result["score"] for result in results])
        print(f"Average score: {average_score} ({len(results)}/{len(dataset)} cases scored)")
    return results

def save_run(results, template):
    """Write a run to runs/<timestamp>.json and return the file path."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "template": template,
        "average": mean([r["score"] for r in results]),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    return path


if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        dataset = json.load(f)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Export it or add it to a .env file.")

    results = run_eval(dataset)
    if not results:
        sys.exit("No test cases were scored; nothing saved.")

    print(f"Saved to {save_run(results, DEFAULT_PROMPT_TEMPLATE)}")