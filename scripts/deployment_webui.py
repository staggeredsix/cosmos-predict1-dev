#!/usr/bin/env python3
"""Simple Gradio launcher for Cosmos-Predict1 pipelines."""

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Dict, List

import gradio as gr


def _read_models() -> List[str]:
    """Parse README.md and return a list of available models."""
    models = []
    collect = False
    for line in Path("README.md").read_text().splitlines():
        if line.strip() == "## Cosmos-Predict1 Models":
            collect = True
            continue
        if collect:
            if line.startswith("* [Cosmos"):
                models.append(line.strip().lstrip("* "))
            elif line.strip() == "":
                if models:
                    break
    return models


def _parse_example_scripts() -> Dict[str, str]:
    """Return a mapping from use case name to script path."""
    mapping = {}
    examples_dir = Path("examples")
    pattern_py = re.compile(r"python\s+(cosmos_predict1/[\w/]+\.py)")
    pattern_m = re.compile(r"-m\s+(cosmos_predict1[\w\.]+)")

    for md in sorted(examples_dir.glob("*.md")):
        name = md.stem.replace("_", " ")
        script = None
        for line in md.read_text().splitlines():
            m = pattern_py.search(line)
            if m:
                script = m.group(1)
                break
            m = pattern_m.search(line)
            if m:
                script = m.group(1)
                if not script.endswith(".py"):
                    script = script.replace(".", "/") + ".py"
                break
        mapping[name] = script
    return mapping


def _get_free_port(start: int = 7860) -> int:
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


USE_CASES = _parse_example_scripts()
MODELS = _read_models()


def launch_use_case(use_case: str) -> str:
    script = USE_CASES.get(use_case)
    if not script:
        return f"No script found for '{use_case}'."

    port = _get_free_port()
    env = os.environ.copy()
    env.setdefault("GRADIO_SERVER_PORT", str(port))

    if script.endswith(".py"):
        cmd = ["python3", script]
    else:
        cmd = ["python3", "-m", script]

    subprocess.Popen(cmd, env=env)
    return f"Started {script} on port {port}."


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Cosmos-Predict1 Deployment")
        gr.Markdown("## Available Models")
        gr.Markdown("\n".join(MODELS))

        gr.Markdown("## Use Cases")
        use_case_dd = gr.Dropdown(choices=list(USE_CASES.keys()), label="Select Use Case")
        out = gr.Textbox(label="Status")
        gr.Button("Launch").click(fn=launch_use_case, inputs=use_case_dd, outputs=out)

    demo.launch()


if __name__ == "__main__":
    main()

