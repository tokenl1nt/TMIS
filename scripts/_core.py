from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
INVENTORY_FILE = "inventory.json"
SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".markdown",
    ".txt",
}


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    uploading_member: str
    uploading_time: str
    source_path: str

    def metadata(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "uploading_member": self.uploading_member,
            "uploading_time": self.uploading_time,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class LoadedDocument:
    source: SourceDocument
    text: str


@dataclass(frozen=True)
class Evidence:
    text: str
    metadata: dict[str, str]
    score: float | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class CoverageReport:
    expected_members: tuple[str, ...]
    covered_members: tuple[str, ...]
    missing_members: tuple[str, ...]
    retrieved_documents: tuple[str, ...]
    evidence_count: int

    @property
    def is_sufficient(self) -> bool:
        return self.evidence_count > 0

    @property
    def has_complete_member_coverage(self) -> bool:
        return not self.missing_members

    def summary(self) -> str:
        missing = ", ".join(self.missing_members) or "none"
        covered = ", ".join(self.covered_members) or "none"
        return (
            f"retrieval_sufficient={str(self.is_sufficient).lower()}; "
            f"covered_members={covered}; missing_members={missing}; "
            f"retrieved_documents={len(self.retrieved_documents)}; "
            f"evidence_items={self.evidence_count}"
        )

    @classmethod
    def from_evidence(
        cls,
        inventory: Sequence[SourceDocument],
        evidence: Sequence[Evidence],
        selected_members: Sequence[str] | None = None,
    ) -> CoverageReport:
        if selected_members:
            expected = set(selected_members)
        else:
            expected = {doc.uploading_member for doc in inventory}

        covered = {
            item.metadata.get("uploading_member", "")
            for item in evidence
            if item.metadata.get("uploading_member") in expected
        }
        retrieved_documents = {
            item.metadata.get("document_id", "")
            for item in evidence
            if item.metadata.get("document_id")
        }

        return cls(
            expected_members=tuple(sorted(expected)),
            covered_members=tuple(sorted(covered)),
            missing_members=tuple(sorted(expected - covered)),
            retrieved_documents=tuple(sorted(retrieved_documents)),
            evidence_count=len(evidence),
        )


@dataclass(frozen=True)
class ModelCallTrace:
    name: str
    prompt: str
    response: str


@dataclass(frozen=True)
class DebugTrace:
    question: str
    policy: str
    draft_answer: str
    final_answer: str
    allowed: bool
    guard_reason: str
    coverage_summary: str
    model_calls: tuple[ModelCallTrace, ...]


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    draft_answer: str
    allowed: bool
    guard_reason: str
    coverage: CoverageReport
    evidence: tuple[Evidence, ...]
    debug_trace: DebugTrace | None = None


@dataclass(frozen=True)
class _GuardDecision:
    allowed: bool
    final_answer: str
    reason: str
    prompt: str
    raw_response: str


def load_input_documents(documents_dir: Path) -> list[LoadedDocument]:
    documents_dir = documents_dir.expanduser().resolve()
    if not documents_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {documents_dir}")

    loaded: list[LoadedDocument] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _read_text(path).strip()
        if not text:
            continue

        source = _source_from_path(documents_dir, path)
        loaded.append(LoadedDocument(source=source, text=text))

    if not loaded:
        raise ValueError(f"No supported text documents found under {documents_dir}")
    return loaded


def write_inventory(persist_dir: Path, sources: Sequence[SourceDocument]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(source) for source in sorted(sources, key=lambda item: item.document_id)]
    (persist_dir / INVENTORY_FILE).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def read_inventory(persist_dir: Path) -> list[SourceDocument]:
    path = persist_dir / INVENTORY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing provenance inventory at {path}. Run `tmsi ingest` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SourceDocument(**item) for item in data]


def build_index(
    documents_dir: Path,
    persist_dir: Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_max_retries: int = 10,
    embedding_timeout: float = 60.0,
) -> int:
    loaded_documents = load_input_documents(documents_dir)
    return build_index_from_loaded_documents(
        loaded_documents,
        persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_max_retries=embedding_max_retries,
        embedding_timeout=embedding_timeout,
        show_progress=True,
    )


def build_index_from_loaded_documents(
    loaded_documents: Sequence[LoadedDocument],
    persist_dir: Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_max_retries: int = 10,
    embedding_timeout: float = 60.0,
    show_progress: bool = False,
) -> int:
    if not loaded_documents:
        raise ValueError("No documents provided for indexing.")

    li = _configure_llama_index(
        embedding_model=embedding_model,
        embedding_max_retries=embedding_max_retries,
        embedding_timeout=embedding_timeout,
    )
    documents = [
        li.Document(text=item.text, metadata=item.source.metadata())
        for item in loaded_documents
    ]
    splitter = li.SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = li.VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=show_progress,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    write_inventory(persist_dir, [item.source for item in loaded_documents])
    return len(loaded_documents)


def load_index(
    persist_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_max_retries: int = 10,
    embedding_timeout: float = 60.0,
):
    li = _configure_llama_index(
        embedding_model=embedding_model,
        embedding_max_retries=embedding_max_retries,
        embedding_timeout=embedding_timeout,
    )
    storage_context = li.StorageContext.from_defaults(persist_dir=str(persist_dir))
    return li.load_index_from_storage(storage_context)


def retrieve_evidence(
    index,
    question: str,
    inventory: Sequence[SourceDocument],
    *,
    selected_members: Sequence[str] | None = None,
    top_k: int = 6,
    trace: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[Evidence], CoverageReport]:
    """Retrieve one relevance-ranked evidence set with one query embedding."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    li = _require_llama_index()
    if trace is not None:
        trace(
            "embedding_query",
            {
                "scope": "global",
                "strategy": "single_pass",
                "top_k": top_k,
                "selected_members": list(selected_members or ()),
            },
        )
    evidence = _retrieve(
        index,
        question,
        top_k=top_k,
        li=li,
        filters=_member_filters(selected_members, li),
    )
    evidence = _deduplicate_evidence(evidence)
    coverage = CoverageReport.from_evidence(inventory, evidence, selected_members)
    return evidence, coverage


def answer_question(
    index,
    question: str,
    policy: str,
    inventory: Sequence[SourceDocument],
    *,
    selected_members: Sequence[str] | None = None,
    top_k: int = 6,
    include_debug_trace: bool = False,
) -> AnswerResult:
    openai_client = _require_openai_client()
    model_calls: list[ModelCallTrace] = []
    evidence, coverage = retrieve_evidence(
        index,
        question,
        inventory,
        selected_members=selected_members,
        top_k=top_k,
    )

    if not coverage.is_sufficient:
        draft = (
            "Insufficient evidence to answer from the retrieved material. "
            f"Coverage check: {coverage.summary()}."
        )
    else:
        reasoning_prompt = _reasoning_prompt(
            question=question,
            policy=policy,
            evidence=evidence,
            coverage=coverage,
        )
        draft = _complete(openai_client, reasoning_prompt)
        if include_debug_trace:
            model_calls.append(
                ModelCallTrace(
                    name="answer_generation",
                    prompt=reasoning_prompt,
                    response=draft,
                )
            )

    guard = _guard_answer(
        openai_client=openai_client,
        policy=policy,
        evidence=evidence,
        draft_answer=draft,
    )
    if include_debug_trace:
        model_calls.append(
            ModelCallTrace(
                name="policy_guard",
                prompt=guard.prompt,
                response=guard.raw_response,
            )
        )

    debug_trace = None
    if include_debug_trace:
        debug_trace = DebugTrace(
            question=question,
            policy=policy,
            draft_answer=draft,
            final_answer=guard.final_answer,
            allowed=guard.allowed,
            guard_reason=guard.reason,
            coverage_summary=coverage.summary(),
            model_calls=tuple(model_calls),
        )

    return AnswerResult(
        answer=guard.final_answer,
        draft_answer=draft,
        allowed=guard.allowed,
        guard_reason=guard.reason,
        coverage=coverage,
        evidence=tuple(evidence),
        debug_trace=debug_trace,
    )


def format_evidence(evidence: Sequence[Evidence], *, max_chars: int = 16_000) -> str:
    parts: list[str] = []
    remaining = max_chars

    for index, item in enumerate(evidence, start=1):
        score = f"{item.score:.4f}" if item.score is not None else "n/a"
        header = (
            f"[E{index}] member={item.metadata.get('uploading_member', 'unknown')} "
            f"document={item.metadata.get('document_id', 'unknown')} "
            f"time={item.metadata.get('uploading_time', 'unknown')} "
            f"score={score}"
        )
        body = item.text.strip()
        entry = f"{header}\n{body}"
        if len(entry) > remaining:
            if remaining > 200:
                parts.append(entry[:remaining].rstrip() + "\n[truncated]")
            break
        parts.append(entry)
        remaining -= len(entry)

    return "\n\n".join(parts) if parts else "No evidence retrieved."


def format_debug_trace(trace: DebugTrace) -> str:
    lines = [
        "Debug trace",
        "=" * 11,
        "",
        "Question",
        "-" * 8,
        trace.question.strip(),
        "",
        "Coverage",
        "-" * 8,
        trace.coverage_summary,
        "",
        "Draft answer",
        "-" * 12,
        trace.draft_answer.strip(),
        "",
        "Guard decision",
        "-" * 14,
        f"allowed={str(trace.allowed).lower()}",
        f"reason={trace.guard_reason}",
        "",
        "Final CLI answer",
        "-" * 16,
        trace.final_answer.strip(),
    ]

    for call in trace.model_calls:
        title = call.name.replace("_", " ").title()
        lines.extend(
            [
                "",
                f"{title} Prompt",
                "-" * (len(title) + 7),
                call.prompt.strip(),
                "",
                f"{title} Raw Response",
                "-" * (len(title) + 13),
                call.response.strip(),
            ]
        )

    return "\n".join(lines)


def _source_from_path(root: Path, path: Path) -> SourceDocument:
    relative = path.relative_to(root)
    member = relative.parts[0] if len(relative.parts) > 1 else "default"
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return SourceDocument(
        document_id=relative.as_posix(),
        uploading_member=member,
        uploading_time=modified.isoformat(),
        source_path=relative.as_posix(),
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _configure_llama_index(
    *,
    embedding_model: str,
    embedding_max_retries: int = 10,
    embedding_timeout: float = 60.0,
) -> SimpleNamespace:
    li = _require_llama_index()
    li.Settings.llm = li.MockLLM()
    li.Settings.embed_model = li.OpenAIEmbedding(
        model=embedding_model,
        max_retries=embedding_max_retries,
        timeout=embedding_timeout,
    )
    return li


def _require_llama_index() -> SimpleNamespace:
    try:
        from llama_index.core import (  # type: ignore[import-not-found]
            Document,
            Settings,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
        )
        from llama_index.core.llms.mock import MockLLM  # type: ignore[import-not-found]
        from llama_index.core.node_parser import SentenceSplitter  # type: ignore[import-not-found]
        from llama_index.core.schema import MetadataMode  # type: ignore[import-not-found]
        from llama_index.core.vector_stores import (  # type: ignore[import-not-found]
            FilterCondition,
            MetadataFilter,
            MetadataFilters,
        )
        from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Missing LlamaIndex dependencies. Install them with `uv sync` or "
            "`python -m pip install -e .`."
        ) from exc

    return SimpleNamespace(
        Document=Document,
        FilterCondition=FilterCondition,
        MetadataFilter=MetadataFilter,
        MetadataFilters=MetadataFilters,
        MetadataMode=MetadataMode,
        OpenAIEmbedding=OpenAIEmbedding,
        MockLLM=MockLLM,
        SentenceSplitter=SentenceSplitter,
        Settings=Settings,
        StorageContext=StorageContext,
        VectorStoreIndex=VectorStoreIndex,
        load_index_from_storage=load_index_from_storage,
    )


def _member_filters(selected_members: Sequence[str] | None, li: SimpleNamespace) -> Any | None:
    members = tuple(dict.fromkeys(selected_members or ()))
    if not members:
        return None

    filters = [
        li.MetadataFilter(key="uploading_member", value=member)
        for member in members
    ]
    if len(filters) == 1:
        return li.MetadataFilters(filters=filters)
    return li.MetadataFilters(filters=filters, condition=li.FilterCondition.OR)


def _retrieve(index, question: str, *, top_k: int, li: SimpleNamespace, filters=None) -> list[Evidence]:
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    results = retriever.retrieve(question)

    evidence: list[Evidence] = []
    for result in results:
        node = result.node
        metadata = {str(key): str(value) for key, value in node.metadata.items()}
        evidence.append(
            Evidence(
                text=_node_text(node, li),
                metadata=metadata,
                score=getattr(result, "score", None),
                node_id=getattr(node, "node_id", None) or getattr(node, "id_", None),
            )
        )
    return evidence


def _node_text(node, li: SimpleNamespace) -> str:
    try:
        return node.get_content(metadata_mode=li.MetadataMode.NONE).strip()
    except TypeError:
        return node.get_content().strip()
    except AttributeError:
        return str(getattr(node, "text", "")).strip()


def _deduplicate_evidence(evidence: Iterable[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    unique: list[Evidence] = []
    for item in evidence:
        key = item.node_id or f"{item.metadata.get('document_id', '')}:{hash(item.text)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _reasoning_prompt(
    *,
    question: str,
    policy: str,
    evidence: Sequence[Evidence],
    coverage: CoverageReport,
) -> str:
    return f"""You are the answer generator for a policy-constrained RAG system.

Shared policy:
{policy.strip()}

Rules:
- Answer only from the retrieved evidence below.
- Do not use outside knowledge.
- If the retrieved evidence is absent or ambiguous, say that clearly.
- If the policy prohibits specific details, omit only those details and answer
  the remaining allowed parts when possible.
- Cite evidence ids like [E1] when making factual claims.
- Keep the answer concise.

Coverage check:
{coverage.summary()}

User question:
{question.strip()}

Retrieved evidence:
{format_evidence(evidence)}
"""


def _guard_answer(
    *,
    openai_client,
    policy: str,
    evidence: Sequence[Evidence],
    draft_answer: str,
) -> _GuardDecision:
    prompt = f"""You are an independent policy guard.

You receive the shared policy, the retrieved evidence, and a draft answer.
You do not receive the user's original prompt.

Shared policy:
{policy.strip()}

Retrieved evidence:
{format_evidence(evidence)}

Draft answer:
{draft_answer.strip()}

Return JSON only with this shape:
{{"allowed": true, "reason": "short reason"}}

Policy interpretation rules:
- If the policy lists forbidden information, allow answers that do not disclose
  or directly reconstruct that forbidden information.
- If the policy lists allowed information, allow only answers within that
  permission.
- Judge the draft answer itself. The retrieved evidence may contain forbidden
  information; that alone is not a reason to refuse.
- Do not refuse allowed facts merely because they appear near forbidden facts
  in the evidence.
- Set allowed to false only when the draft answer violates a prohibition or,
  for allow-list policies, falls outside the listed permissions.
"""
    raw = _complete(openai_client, prompt)
    payload = _parse_guard_json(raw)

    if payload.get("allowed") is True:
        return _GuardDecision(
            allowed=True,
            final_answer=draft_answer,
            reason=str(payload.get("reason", "Allowed by policy.")),
            prompt=prompt,
            raw_response=raw,
        )

    reason = str(payload.get("reason", "The policy guard did not allow the draft answer."))
    return _GuardDecision(
        allowed=False,
        final_answer=f"Refusal: {reason}",
        reason=reason,
        prompt=prompt,
        raw_response=raw,
    )


def _parse_guard_json(raw: str) -> dict[str, object]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _require_openai_client():
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenAI dependency. Install it with `uv sync` or "
            "`python -m pip install -e .`."
        ) from exc

    return OpenAI()


def _complete(openai_client, prompt: str) -> str:
    request = {
        "model": DEFAULT_LLM_MODEL,
        "input": [{"role": "user", "content": prompt}],
    }
    if _supports_reasoning(DEFAULT_LLM_MODEL):
        request["reasoning"] = {"effort": DEFAULT_REASONING_EFFORT}

    response = openai_client.responses.create(**request)
    text = _extract_response_text(response)
    if not text:
        raise RuntimeError("OpenAI response did not contain output text.")
    return text.strip()


def _supports_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o"))


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
