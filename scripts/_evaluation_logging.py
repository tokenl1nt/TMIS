from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationContext:
    workload_index: int | None = None
    workload_total: int | None = None
    scenario_id: str | None = None
    query_id: str | None = None
    guard_mode: str | None = None
    turn_index: int | None = None


_CONTEXT: ContextVar[EvaluationContext] = ContextVar(
    "tmsi_evaluation_context",
    default=EvaluationContext(),
)
_ACTIVE_LOGGER: EvaluationLogger | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def current_context() -> EvaluationContext:
    return _CONTEXT.get()


@contextmanager
def evaluation_context(**values: Any) -> Iterator[EvaluationContext]:
    context = replace(current_context(), **values)
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def set_active_logger(logger: EvaluationLogger | None) -> None:
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def active_logger() -> EvaluationLogger | None:
    return _ACTIVE_LOGGER


def log_event(event: str, *, level: str = "info", cli: bool = False, **details: Any) -> None:
    logger = active_logger()
    if logger is not None:
        logger.event(event, level=level, cli=cli, **details)


class EvaluationLogger:
    """Thread-safe structured run logger with concise stderr progress output."""

    def __init__(self, path: Path):
        self.path = path
        self._write_lock = Lock()
        self._counter_lock = Lock()
        self._counters: dict[str, int | float] = {}

    @classmethod
    def start(cls, path: Path, *, metadata: Mapping[str, Any]) -> EvaluationLogger:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        logger = cls(path)
        logger.event("run_started", metadata=_compact_mapping(metadata))
        return logger

    def resume(self, *, output_dir: Path, result_path: Path, existing_results: int) -> None:
        self.event(
            "run_resumed",
            output_dir=str(output_dir),
            results_path=str(result_path),
            existing_results=existing_results,
        )

    def increment(self, name: str, amount: int | float = 1) -> None:
        with self._counter_lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def counters(self) -> dict[str, int | float]:
        with self._counter_lock:
            return dict(sorted(self._counters.items()))

    def event(self, event: str, *, level: str = "info", cli: bool = False, **details: Any) -> None:
        record: dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "level": level,
            "event": event,
        }
        context = _compact_mapping(asdict(current_context()))
        if context:
            record["context"] = context
        clean_details = _compact_mapping(details)
        if clean_details:
            record["details"] = clean_details
        self._append_record(record)
        if cli:
            self._print_cli(event, level=level, details=clean_details)

    def query_completed(self, *, result_count: int, elapsed_seconds: float) -> None:
        self.increment("queries_completed")
        self.event(
            "query_completed",
            cli=True,
            result_count=result_count,
            elapsed_seconds=round(elapsed_seconds, 3),
        )

    def run_completed(self, *, summary: Mapping[str, Any]) -> None:
        self.event("run_completed", cli=True, counters=self.counters(), summary=dict(summary))

    def run_failed(self, error: BaseException) -> None:
        self.event(
            "run_failed",
            level="error",
            cli=True,
            error_type=type(error).__name__,
            error=str(error),
            counters=self.counters(),
        )

    def _append_record(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{line}\n")

    def _print_cli(self, event: str, *, level: str, details: Mapping[str, Any]) -> None:
        context = current_context()
        progress = ""
        if context.workload_index is not None and context.workload_total is not None:
            progress = f"[{context.workload_index:>3}/{context.workload_total}] "
        scenario_id = context.scenario_id or details.get("scenario_id")
        query_id = context.query_id or details.get("query_id")
        target = "/".join(str(value) for value in (scenario_id, query_id) if value)
        if target:
            target = f"{target:<13} "
        step = str(details.get("step") or _event_step(event)).upper()
        status = _event_status(event, level)
        qualifiers: list[str] = []
        if context.guard_mode:
            qualifiers.append(f"guard={context.guard_mode}")
        if context.turn_index is not None:
            qualifiers.append(f"turn={context.turn_index}")
        if details.get("attempt") is not None and details.get("max_attempts") is not None:
            qualifiers.append(f"attempt={details['attempt']}/{details['max_attempts']}")
        if details.get("elapsed_seconds") is not None:
            qualifiers.append(f"{float(details['elapsed_seconds']):.2f}s")
        if details.get("message"):
            qualifiers.append(str(details["message"]))
        suffix = f" | {' | '.join(qualifiers)}" if qualifiers else ""
        with self._write_lock:
            print(f"{progress}{target}{step:<7} {status}{suffix}", file=sys.stderr, flush=True)


def append_jsonl(path: Path, record: Mapping[str, Any], *, lock: Lock | None = None) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
    if lock is None:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(f"{line}\n")
        return
    with lock:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(f"{line}\n")


def read_run_metadata(path: Path) -> dict[str, str]:
    """Read current JSON logs and legacy key/value logs for resume compatibility."""
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record.get("event") != "run_started":
            continue
        metadata = record.get("details", {}).get("metadata", {})
        return {str(key): _metadata_value(value) for key, value in metadata.items()}

    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Running ") or line.startswith("Skipping "):
            break
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        metadata.setdefault(key.strip(), value.strip())
    return metadata


def selected_http_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    allowed = {
        "x-request-id",
        "x-openrouter-generation-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "retry-after",
    }
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if str(key).lower() in allowed
    }


def response_diagnostics(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        usage = response.get("usage", {})
        metadata = response.get("ResponseMetadata", {})
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, Mapping) else {}
        content = message.get("content", []) if isinstance(message, Mapping) else []
        return _compact_mapping(
            {
                "response_id": metadata.get("RequestId") if isinstance(metadata, Mapping) else None,
                "finish_reason": response.get("stopReason"),
                "prompt_tokens": usage.get("inputTokens") if isinstance(usage, Mapping) else None,
                "completion_tokens": usage.get("outputTokens") if isinstance(usage, Mapping) else None,
                "total_tokens": usage.get("totalTokens") if isinstance(usage, Mapping) else None,
                "content_present": bool(content),
                **_mapping_content_diagnostics(message, content),
            }
        )

    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    return _compact_mapping(
        {
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "upstream_provider": getattr(response, "provider", None),
            "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
            "native_finish_reason": getattr(choice, "native_finish_reason", None) if choice is not None else None,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details is not None else None,
            "cached_tokens": getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None,
            "cost": getattr(usage, "cost", None) if usage is not None else None,
            "content_present": bool(getattr(message, "content", None)) if message is not None else None,
        }
    )


def _mapping_content_diagnostics(message: Any, content: Any) -> dict[str, Any]:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return {"content_type": type(content).__name__}

    block_types: list[str] = []
    block_keys: list[str] = []
    tool_names: list[str] = []
    text_blocks = 0
    text_chars = 0
    for block in content:
        if not isinstance(block, Mapping):
            block_types.append(type(block).__name__)
            continue

        keys = sorted(str(key) for key in block.keys())
        if keys:
            block_keys.append(",".join(keys))
        matched_type = False
        for key in (
            "text",
            "toolUse",
            "toolResult",
            "reasoningContent",
            "citationsContent",
            "guardContent",
            "cachePoint",
            "image",
            "document",
            "video",
        ):
            if key in block:
                block_types.append(key)
                matched_type = True
        if not matched_type:
            block_types.append("unknown")

        text = block.get("text")
        if isinstance(text, str):
            text_blocks += 1
            text_chars += len(text)

        tool_use = block.get("toolUse")
        if isinstance(tool_use, Mapping) and tool_use.get("name"):
            tool_names.append(str(tool_use["name"]))

    role = message.get("role") if isinstance(message, Mapping) else None
    return _compact_mapping(
        {
            "message_role": role,
            "content_block_count": len(content),
            "content_block_types": _dedupe_preserving_order(block_types),
            "content_block_keys": _dedupe_preserving_order(block_keys),
            "content_text_blocks": text_blocks,
            "content_text_chars": text_chars,
            "tool_use_names": _dedupe_preserving_order(tool_names),
        }
    )


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def exception_diagnostics(error: BaseException) -> dict[str, Any]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    body = getattr(error, "body", None)
    response_body = _response_body_diagnostics(response) if body is None else None
    return _compact_mapping(
        {
            "error_type": type(error).__name__,
            "error": str(error)[:2_000],
            "http_status": getattr(error, "status_code", None),
            "request_id": getattr(error, "request_id", None),
            "http_headers": selected_http_headers(headers),
            "api_error_code": getattr(error, "code", None),
            "api_error_param": getattr(error, "param", None),
            "api_error_type": getattr(error, "type", None),
            "api_error_body": _truncate_diagnostic_value(body),
            "api_response_body": response_body,
        }
    )


def _response_body_diagnostics(response: Any) -> Any:
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return _truncate_diagnostic_value(json_method())
        except Exception:
            pass
    text = getattr(response, "text", None)
    if text is None:
        return None
    return _truncate_diagnostic_value(text)


def _truncate_diagnostic_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:2_000]
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)[:2_000]
    if len(encoded) <= 2_000:
        return value
    return encoded[:2_000]


def _compact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in values.items()
        if value is not None and value != {} and value != [] and value != () and value != ""
    }


def _metadata_value(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(item) for item in value)
    if value is None:
        return "n/a"
    return str(value)


def _event_step(event: str) -> str:
    if event.startswith("rag_") or event.startswith("index_"):
        return "rag"
    if event.startswith("model_"):
        return "llm"
    if event.startswith("query_"):
        return "query"
    if event.startswith("run_"):
        return "run"
    return event.split("_", 1)[0]


def _event_status(event: str, level: str) -> str:
    if level == "error" or event.endswith("failed"):
        return "FAILED"
    if event.endswith("started"):
        return "START"
    if event.endswith("retry"):
        return "RETRY"
    if event.endswith("completed"):
        return "DONE"
    if event.endswith("cache_hit"):
        return "CACHED"
    if event.endswith("skipped"):
        return "SKIP"
    if event.endswith("persisted"):
        return "SAVED"
    if event.endswith("ready"):
        return "READY"
    return event.upper()
