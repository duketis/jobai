"""Tests for the structured JSON logging configuration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
import structlog

from jobai.observability.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Reset structlog and root-logger handlers between tests.

    structlog and stdlib logging are global state. Without this fixture
    a test that configures logging would leak its handlers into the next
    test and mask real failures.
    """
    yield
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)


def test_configure_logging_emits_json_with_expected_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="INFO")
    log = get_logger("jobai.test")

    log.info("hello", user_id=42, source="greenhouse")

    captured = capsys.readouterr()
    parsed = json.loads(captured.err.strip())

    assert parsed["event"] == "hello"
    assert parsed["user_id"] == 42
    assert parsed["source"] == "greenhouse"
    assert parsed["level"] == "info"
    assert "timestamp" in parsed


def test_configure_logging_creates_rotating_file_when_dir_given(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(level="INFO", log_dir=log_dir)
    log = get_logger("jobai.test")

    log.info("file_test", x=1)

    # Force handlers to flush so the file write is visible.
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = log_dir / "jobai.log"
    assert log_file.exists()

    contents = log_file.read_text(encoding="utf-8").strip()
    parsed = json.loads(contents)
    assert parsed["event"] == "file_test"
    assert parsed["x"] == 1


def test_configure_logging_does_not_create_file_when_dir_is_none() -> None:
    configure_logging(level="INFO", log_dir=None)

    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers == []


def test_log_level_filters_messages_below_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(level="WARNING")
    log = get_logger("jobai.test")

    log.info("should be filtered")
    log.warning("should appear")

    captured = capsys.readouterr().err
    assert "should be filtered" not in captured
    assert "should appear" in captured


def test_unknown_level_falls_back_to_info(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="NOT_A_REAL_LEVEL")
    log = get_logger("jobai.test")

    log.info("info_message")
    log.debug("debug_message")

    captured = capsys.readouterr().err
    assert "info_message" in captured
    assert "debug_message" not in captured


def test_get_logger_returns_a_structlog_bound_logger() -> None:
    configure_logging(level="INFO")
    log = get_logger("jobai.test")

    # BoundLogger isn't directly importable in a way mypy likes across versions,
    # but it always exposes the standard logging methods.
    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    assert hasattr(log, "bind")


def test_configure_logging_replaces_existing_handlers(
    tmp_path: Path,
) -> None:
    """Re-configuring must not leave stale handlers from the previous run."""
    configure_logging(level="INFO", log_dir=tmp_path / "first")
    first_handler_count = len(logging.getLogger().handlers)

    configure_logging(level="INFO", log_dir=tmp_path / "second")
    second_handler_count = len(logging.getLogger().handlers)

    assert second_handler_count == first_handler_count
    # Only the second log_dir's file should exist as a handler target.
    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert "second" in file_handlers[0].baseFilename
