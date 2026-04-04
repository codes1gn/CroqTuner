import re

import pytest

from app.agent import (
    BASELINE_PATTERN,
    ITER_PATTERN,
    build_command,
    build_prompt,
    extract_session_id,
    is_fatal_agent_error,
)
from app.config import settings
from app.models import Task


def _make_task(**kwargs):
    defaults = dict(
        shape_key="f16_768x768x768",
        dtype="f16",
        m=768, n=768, k=768,
        mode="from_current_best",
        max_iterations=30,
        status="running",
        current_iteration=0,
    )
    defaults.update(kwargs)
    return Task(**defaults)


def test_iter_pattern_keep():
    line = "iter005: unroll factor 4 — 89.7 TFLOPS (KEEP)"
    m = ITER_PATTERN.search(line)
    assert m is not None
    assert m.group(1) == "005"
    assert float(m.group(2)) == 89.7
    assert m.group(3) == "KEEP"


def test_iter_pattern_discard():
    line = "iter_023: barrier depth — 45.2 TFLOPS (DISCARD)"
    m = ITER_PATTERN.search(line)
    assert m is not None
    assert m.group(1) == "023"
    assert m.group(3) == "DISCARD"


def test_iter_pattern_no_match():
    line = "Compiling kernel..."
    m = ITER_PATTERN.search(line)
    assert m is None


def test_baseline_pattern():
    line = "Baseline measurement: 123.4 TFLOPS"
    m = BASELINE_PATTERN.search(line)
    assert m is not None
    assert float(m.group(1)) == 123.4


def test_build_prompt_from_best():
    task = _make_task(mode="from_current_best")
    prompt = build_prompt(task)
    assert "fsm-engine" in prompt
    assert "ai-tune-from-current-best" in prompt
    assert "M=768" in prompt
    assert "max 30 iterations" in prompt


def test_build_prompt_from_scratch():
    task = _make_task(mode="from_scratch", shape_key="f16_768x768x768_fs", max_iterations=150)
    prompt = build_prompt(task)
    assert "ai-tune-from-scratch" in prompt
    assert "max 150 iterations" in prompt


def test_build_command_includes_model_override():
    task = _make_task()
    command = build_command(task)
    assert command[:3] == [settings.opencode_bin, "run", "--print-logs"]
    assert "--model" in command
    model_index = command.index("--model")
    assert command[model_index + 1] == settings.opencode_model


def test_build_command_prefers_task_model():
    task = _make_task(model="opencode/big-pickle")
    command = build_command(task)
    model_index = command.index("--model")
    assert command[model_index + 1] == "opencode/big-pickle"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Error: Model not found: nvidia/nemotron-3-super-free.", True),
        ("ProviderModelNotFoundError: ProviderModelNotFoundError", True),
        ("iter_003: candidate A -- 91.2 TFLOPS (KEEP)", False),
    ],
)
def test_is_fatal_agent_error(line, expected):
    assert is_fatal_agent_error(line) is expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("INFO service=llm sessionID=ses_abc123 stream", "ses_abc123"),
        ("GET /session/ses_xyz789/message request", "ses_xyz789"),
        ("iter_003: candidate A -- 91.2 TFLOPS (KEEP)", None),
    ],
)
def test_extract_session_id(line, expected):
    assert extract_session_id(line) == expected
