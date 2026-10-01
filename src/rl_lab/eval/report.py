"""Readable per-example benchmark reports, with wrong answers first."""

from datetime import datetime, timezone
from pathlib import Path
import re


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        part["text"] if part["type"] == "text"
        else f"[{part['type']}]\n{part['text']}"
        for part in content
    )


def _block(text):
    # A generated code block must not close the report's enclosing fence.
    width = max((len(match[0]) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, width + 1)
    return f"{fence}text\n{text}\n{fence}\n\n"


class EvaluationReport:
    """Collect scored examples and write one Markdown file after a successful run."""

    def __init__(self, task_name, model_name, settings):
        self.task_name = task_name
        self.model_name = model_name
        self.settings = settings
        self.examples = []

    def record(self, index, example, completion, score):
        self.examples.append((index, example, completion, score))

    def write(self, result):
        timestamp = datetime.now(timezone.utc)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{self.task_name}-{self.model_name}")
        directory = Path("report")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{slug}-{timestamp:%Y%m%dT%H%M%S%fZ}.md"
        wrong = [record for record in self.examples if record[3] != 1]
        correct = [record for record in self.examples if record[3] == 1]
        with path.open("x", encoding="utf-8") as output:
            output.write(f"# {self.task_name} evaluation\n\n")
            output.write(f"Model: {self.model_name}\n\n")
            output.write(f"Timestamp (UTC): {timestamp.isoformat()}\n\n")
            output.write(
                f"Accuracy: {result.score:.4f} | Examples: {result.num_examples} | "
                f"Wrong: {len(wrong)} | Correct: {len(correct)}\n\n"
            )
            output.write("## Run settings\n\n")
            for key, value in self.settings.items():
                output.write(f"- {key}: {value}\n")
            output.write("\n")
            for title, records in (("Wrong answers", wrong), ("Correct answers", correct)):
                output.write(f"## {title} ({len(records)})\n\n")
                for index, example, completion, score in records:
                    output.write(f"### Example {index + 1} — score {score:g}\n\n")
                    question = "\n\n".join(
                        f"{message['role']}:\n{_text(message['content'])}"
                        for message in example["messages"][:-1]
                    )
                    output.write("**Question / conversation**\n\n" + _block(question))
                    reference = _text(example["messages"][-1]["content"])
                    output.write("**Expected answer**\n\n" + _block(reference))
                    output.write("**Model answer**\n\n" + _block(completion))
        return path
