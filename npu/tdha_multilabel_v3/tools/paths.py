"""Path contracts for explicit, read-only inputs and separate outputs."""
from pathlib import Path


def output_separate(output, *inputs):
    output = Path(output).resolve()
    for item in inputs:
        source = Path(item).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from every input directory")
    return output


def task_path(root, task):
    root = Path(root).resolve()
    path = (root / task).resolve()
    if not isinstance(task, str) or not task or path.parent != root:
        raise ValueError("Task path must be a direct child of its root")
    return path
