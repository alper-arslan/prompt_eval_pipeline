import json

from client import add_user_message, add_assistant_message, chat


DEFAULT_DATASET_PROMPT = """Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
    {
        "task": "Description of task",
        "format": "json" or "python" or "regex",
        "solution_criteria": "Key criteria for evaluating the solution"
    },
    ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
* Focus on tasks that do not require writing much code

Please generate {count} objects, with a roughly even mix of python, json and regex tasks.
"""


def generate_dataset(prompt=DEFAULT_DATASET_PROMPT, count=15):
    """Ask the model for evaluation test cases and return them as a list of dicts.

    The prompt's {count} placeholder is replaced with the number of cases wanted.
    Each case has a "task", an expected "format" (json, python or regex) and
    "solution_criteria" for the grader. The reply is prefilled with a json code
    fence and stopped at the closing fence so only the JSON array comes back.
    """
    messages = []
    add_user_message(messages, prompt.replace("{count}", str(count)))
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)

if __name__ == "__main__":
    dataset = generate_dataset()
    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)