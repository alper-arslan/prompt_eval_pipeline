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
from statistics import mean

from anthropic import AuthenticationError, PermissionDeniedError

from client import add_user_message, chat, add_assistant_message


def run_prompt(test_case):
    """Run the prompt under test on one test case and return the raw model output.

    The reply is prefilled with a code fence for the case's format and stopped at
    the closing fence, so the output holds only the answer itself.
    """
    prompt = f"""
    please solve the following task:
    {test_case["task"]}

    * The required output format is: {test_case["format"]}
    * Output exactly one code block containing only the answer, nothing else
    * For regex, the answer is a single-line pattern only: no code, no function, no imports
    * Do not add any comments, docstrings, commentary or explanations
"""
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

def run_test_case(test_case):
    """Run one test case and return its output, reasoning and final score.

    The final score is the mean of the model grade and the syntax grade.
    """
    output = run_prompt(test_case)
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    grade_score = grade_syntax(output, test_case)
    score =  (model_score + grade_score) / 2
    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning,
    }

def run_eval(dataset):
    """Run every test case in the dataset, print the average score and return the results.

    A case that fails is skipped with a message, except authentication errors,
    which stop the whole run.
    """
    results = []
    for test_case in dataset:
        try:
            results.append(run_test_case(test_case))
        except (AuthenticationError, PermissionDeniedError):
            raise
        except Exception as e:
            print(f"Skipping failed test case {test_case['task'][:60]!r}: {e}")

    if results:
        average_score = mean([result["score"] for result in results])
        print(f"Average score: {average_score} ({len(results)}/{len(dataset)} cases scored)")
    return results


if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        dataset = json.load(f)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Export it or add it to a .env file.")

    results = run_eval(dataset)
    if not results:
        sys.exit("No test cases were scored; leaving results.json untouched.")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))