import re

import pytest

from app.agent import ITER_PATTERN, BASELINE_PATTERN, build_prompt
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
