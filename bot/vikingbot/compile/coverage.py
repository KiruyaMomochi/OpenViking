"""Deterministic, tool-independent coverage accounting for Compile."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, cast

_VERSION = 2
_SUPPORTED_VERSIONS = frozenset({1, 2})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORDS_RE = re.compile(r"\w+", re.UNICODE)
_GENERIC_SKIP_PHRASES = {
    "不相关",
    "不重要",
    "无需",
    "无关",
    "与当前任务无关",
    "范围外",
    "跳过",
}
_GENERIC_SKIP_WORDS = {
    "a",
    "because",
    "current",
    "duplicate",
    "ignore",
    "ignored",
    "irrelevant",
    "is",
    "item",
    "low",
    "n",
    "na",
    "need",
    "needed",
    "no",
    "not",
    "of",
    "out",
    "priority",
    "reason",
    "relevant",
    "scope",
    "simply",
    "skip",
    "skipped",
    "source",
    "task",
    "the",
    "this",
    "to",
    "unit",
    "unimportant",
    "unnecessary",
    "useful",
    "value",
}
_SKIP_EVIDENCE_RE = re.compile(
    r"(?:viking://|[/\\]|\.[a-z0-9]{1,12}\b|\b(?:generated|duplicate|empty|control|"
    r"metadata|derived|binary|archive|cache|lockfile|manifest|schema|snapshot|summary|"
    r"构建|生成|重复|空|控制|元数据|派生|二进制|归档|缓存|锁文件|清单|摘要)\b)",
    re.IGNORECASE,
)


class CoverageError(ValueError):
    """Invalid evidence, snapshot or coverage transition."""


class CoverageStatus(str, Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    COVERED = "covered"
    SKIPPED = "skipped"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoverageError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CoverageError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_range(
    start: Any,
    end: Any,
    *,
    start_field: str = "range_start",
    end_field: str = "range_end",
) -> tuple[int | None, int | None]:
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise CoverageError(f"{start_field} and {end_field} must be provided together")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise CoverageError("source range must contain non-negative integers with start < end")
    return start, end


@dataclass(frozen=True, slots=True)
class ReviewUnit:
    unit_id: str
    source_id: str
    locator: str
    kind: str = "document"
    substantive: bool = True
    content_sha256: str | None = None
    discovered_bytes: int | None = None
    parent_unit_id: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    source_revision: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        for field in ("unit_id", "source_id", "locator", "kind"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if not isinstance(self.substantive, bool):
            raise CoverageError("substantive must be a boolean")
        if self.content_sha256 is not None:
            object.__setattr__(
                self, "content_sha256", _digest(self.content_sha256, "content_sha256")
            )
        if self.discovered_bytes is not None and (
            isinstance(self.discovered_bytes, bool)
            or not isinstance(self.discovered_bytes, int)
            or self.discovered_bytes < 0
        ):
            raise CoverageError("discovered_bytes must be a non-negative integer")
        if self.parent_unit_id is not None:
            object.__setattr__(self, "parent_unit_id", _text(self.parent_unit_id, "parent_unit_id"))
            if self.parent_unit_id == self.unit_id:
                raise CoverageError("review unit hierarchy contains a cycle")
        start, end = _source_range(self.range_start, self.range_end)
        object.__setattr__(self, "range_start", start)
        object.__setattr__(self, "range_end", end)
        if self.source_revision is not None:
            object.__setattr__(
                self, "source_revision", _text(self.source_revision, "source_revision")
            )
        if not isinstance(self.required, bool):
            raise CoverageError("required must be a boolean")


@dataclass(frozen=True, slots=True)
class VerifiedSummary:
    summary_unit_id: str
    covered_unit_ids: tuple[str, ...]
    verifier: str
    membership_sha256: str = ""

    def __post_init__(self) -> None:
        summary = _text(self.summary_unit_id, "summary_unit_id")
        members = tuple(_text(item, "covered_unit_id") for item in self.covered_unit_ids)
        verifier = _text(self.verifier, "verifier")
        if not members or len(members) != len(set(members)) or summary in members:
            raise CoverageError("verified summary requires unique, non-self members")
        object.__setattr__(self, "summary_unit_id", summary)
        object.__setattr__(self, "covered_unit_ids", members)
        object.__setattr__(self, "verifier", verifier)
        expected = _sha256([summary, list(members), verifier])
        if self.membership_sha256 and self.membership_sha256 != expected:
            raise CoverageError("verified summary membership digest does not match")
        object.__setattr__(self, "membership_sha256", expected)


def _unit_payload(unit: ReviewUnit) -> dict[str, Any]:
    """Return the stable unit payload, omitting additive fields at their defaults."""

    payload = {
        "unit_id": unit.unit_id,
        "source_id": unit.source_id,
        "locator": unit.locator,
        "kind": unit.kind,
        "substantive": unit.substantive,
        "content_sha256": unit.content_sha256,
        "discovered_bytes": unit.discovered_bytes,
    }
    if unit.parent_unit_id is not None:
        payload["parent_unit_id"] = unit.parent_unit_id
    if unit.range_start is not None:
        payload["range_start"] = unit.range_start
        payload["range_end"] = unit.range_end
    if unit.source_revision is not None:
        payload["source_revision"] = unit.source_revision
    if not unit.required:
        payload["required"] = False
    return payload


def _snapshot_payload(snapshot: "CollectionSnapshot") -> dict[str, Any]:
    return {
        "source_id": snapshot.source_id,
        "units": [_unit_payload(unit) for unit in snapshot.units],
        "verified_summaries": [
            {**asdict(summary), "covered_unit_ids": list(summary.covered_unit_ids)}
            for summary in snapshot.verified_summaries
        ],
    }


@dataclass(frozen=True, slots=True)
class CollectionSnapshot:
    source_id: str
    units: tuple[ReviewUnit, ...]
    verified_summaries: tuple[VerifiedSummary, ...] = ()
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        source = _text(self.source_id, "source_id")
        units = tuple(self.units)
        summaries = tuple(self.verified_summaries)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "verified_summaries", summaries)
        by_id = {unit.unit_id: unit for unit in units}
        if len(by_id) != len(units):
            raise CoverageError("collection snapshot contains duplicate unit IDs")
        if any(unit.source_id != source for unit in units):
            raise CoverageError("collection unit belongs to a different source")
        for unit in units:
            if unit.parent_unit_id is None:
                continue
            parent = by_id.get(unit.parent_unit_id)
            if parent is None or parent.source_id != unit.source_id:
                raise CoverageError(
                    "parent and child units must belong to the same collection snapshot and source"
                )
            if parent.source_revision != unit.source_revision:
                raise CoverageError("parent and child units must use the same source revision")
            if (
                parent.range_start is not None
                and unit.range_start is not None
                and (
                    unit.range_start < parent.range_start
                    or cast(int, unit.range_end) > cast(int, parent.range_end)
                )
            ):
                raise CoverageError("child source range must be contained by its parent range")
        for unit in units:
            ancestors: set[str] = set()
            current = unit
            while current.parent_unit_id is not None:
                if current.unit_id in ancestors:
                    raise CoverageError("review unit hierarchy contains a cycle")
                ancestors.add(current.unit_id)
                current = by_id[current.parent_unit_id]
        seen: set[str] = set()
        for summary in summaries:
            unit = by_id.get(summary.summary_unit_id)
            if summary.summary_unit_id in seen or unit is None or unit.kind != "summary":
                raise CoverageError("verified summary does not name a unique summary unit")
            if any(member not in by_id for member in summary.covered_unit_ids):
                raise CoverageError("verified summary references an unknown unit")
            seen.add(summary.summary_unit_id)
        expected = _sha256(_snapshot_payload(self))
        if self.snapshot_id and self.snapshot_id != expected:
            raise CoverageError("collection snapshot digest does not match")
        object.__setattr__(self, "snapshot_id", expected)


def _receipt_payload(receipt: "ReadReceipt") -> dict[str, Any]:
    payload = {
        "snapshot_id": receipt.snapshot_id,
        "unit_id": receipt.unit_id,
        "locator": receipt.locator,
        "content_sha256": receipt.content_sha256,
        "byte_count": receipt.byte_count,
        "complete": receipt.complete,
    }
    if receipt.range_start is not None:
        payload["range_start"] = receipt.range_start
        payload["range_end"] = receipt.range_end
    if receipt.source_revision is not None:
        payload["source_revision"] = receipt.source_revision
    return payload


@dataclass(frozen=True, slots=True)
class ReadReceipt:
    snapshot_id: str
    unit_id: str
    locator: str
    content_sha256: str
    byte_count: int
    complete: bool
    receipt_id: str = ""
    range_start: int | None = None
    range_end: int | None = None
    source_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _digest(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "unit_id", _text(self.unit_id, "unit_id"))
        object.__setattr__(self, "locator", _text(self.locator, "locator"))
        object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256"))
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise CoverageError("byte_count must be an integer")
        if self.byte_count < 0 or not isinstance(self.complete, bool):
            raise CoverageError("invalid read receipt length or completeness")
        start, end = _source_range(self.range_start, self.range_end)
        object.__setattr__(self, "range_start", start)
        object.__setattr__(self, "range_end", end)
        if self.source_revision is not None:
            object.__setattr__(
                self, "source_revision", _text(self.source_revision, "source_revision")
            )
        expected = _sha256(_receipt_payload(self))
        if self.receipt_id and self.receipt_id != expected:
            raise CoverageError("read receipt digest does not match")
        object.__setattr__(self, "receipt_id", expected)

    @classmethod
    def for_content(
        cls,
        snapshot: CollectionSnapshot,
        unit_id: str,
        content: str | bytes,
        *,
        complete: bool = True,
    ) -> "ReadReceipt":
        unit = next((item for item in snapshot.units if item.unit_id == unit_id), None)
        if unit is None:
            raise CoverageError(f"unknown unit_id {unit_id!r}")
        raw = content.encode() if isinstance(content, str) else bytes(content)
        return cls(
            snapshot.snapshot_id,
            unit.unit_id,
            unit.locator,
            hashlib.sha256(raw).hexdigest(),
            len(raw),
            complete,
            range_start=unit.range_start,
            range_end=unit.range_end,
            source_revision=unit.source_revision,
        )


@dataclass(slots=True)
class _State:
    status: CoverageStatus = CoverageStatus.PENDING
    receipt: ReadReceipt | None = None
    covered_by: str | None = None
    skip_reason: str | None = None
    transition_iteration: int | None = None
    harness_non_substantive: bool = False


def _skip_reason(reason: Any) -> str:
    reason = _text(reason, "skip reason")
    words = _WORDS_RE.findall(reason.casefold())
    normalized = " ".join(words)
    if (
        not words
        or normalized in _GENERIC_SKIP_PHRASES
        or all(word in _GENERIC_SKIP_WORDS for word in words)
        or (len(reason) < 12 and not _SKIP_EVIDENCE_RE.search(reason))
    ):
        raise CoverageError("skip reason must identify a concrete source-specific reason")
    return reason


class CoverageLedger:
    """Coverage state machine; list/search have no state-changing entry point."""

    def __init__(self, snapshots: Iterable[CollectionSnapshot]):
        self._snapshots = tuple(snapshots)
        self._by_source = {snapshot.source_id: snapshot for snapshot in self._snapshots}
        if len(self._by_source) != len(self._snapshots):
            raise CoverageError("duplicate source_id")
        self._units = {
            unit.unit_id: unit for snapshot in self._snapshots for unit in snapshot.units
        }
        if len(self._units) != sum(len(snapshot.units) for snapshot in self._snapshots):
            raise CoverageError("duplicate unit_id across sources")
        self._snapshot_by_unit = {
            unit.unit_id: snapshot for snapshot in self._snapshots for unit in snapshot.units
        }
        children: dict[str, list[str]] = {unit_id: [] for unit_id in self._units}
        self._parents: dict[str, str] = {}
        for unit in self._units.values():
            if unit.parent_unit_id is None:
                continue
            parent = self._units.get(unit.parent_unit_id)
            if parent is None or parent.source_id != unit.source_id:
                raise CoverageError("parent and child units must belong to the same source")
            if self._snapshot_by_unit[parent.unit_id] is not self._snapshot_by_unit[unit.unit_id]:
                raise CoverageError("parent and child units must belong to the same snapshot")
            children[parent.unit_id].append(unit.unit_id)
            self._parents[unit.unit_id] = parent.unit_id
        self._children = {unit_id: tuple(items) for unit_id, items in children.items()}
        self._summaries = {
            summary.summary_unit_id: summary
            for snapshot in self._snapshots
            for summary in snapshot.verified_summaries
        }
        self._states = {unit_id: _State() for unit_id in self._units}

    @classmethod
    def from_sources(
        cls,
        sources: (
            Iterable[CollectionSnapshot]
            | Iterable[ReviewUnit]
            | Mapping[str, CollectionSnapshot | Iterable[ReviewUnit]]
        ),
    ) -> "CoverageLedger":
        if isinstance(sources, Mapping):
            snapshots = []
            for source_id, value in sources.items():
                snapshot = (
                    value
                    if isinstance(value, CollectionSnapshot)
                    else CollectionSnapshot(source_id, tuple(value))
                )
                if snapshot.source_id != source_id:
                    raise CoverageError("source key does not match snapshot source_id")
                snapshots.append(snapshot)
            return cls(snapshots)
        items = tuple(sources)
        if all(isinstance(item, CollectionSnapshot) for item in items):
            return cls(cast(tuple[CollectionSnapshot, ...], items))
        if all(isinstance(item, ReviewUnit) for item in items):
            grouped: dict[str, list[ReviewUnit]] = {}
            for unit in cast(tuple[ReviewUnit, ...], items):
                grouped.setdefault(unit.source_id, []).append(unit)
            return cls(
                CollectionSnapshot(source, tuple(units)) for source, units in grouped.items()
            )
        raise CoverageError("sources must contain only snapshots or review units")

    @property
    def snapshots(self) -> tuple[CollectionSnapshot, ...]:
        return self._snapshots

    @property
    def units(self) -> tuple[ReviewUnit, ...]:
        return tuple(self._units.values())

    @property
    def by_id(self) -> Mapping[str, ReviewUnit]:
        return MappingProxyType(self._units)

    def status(self, unit_id: str) -> CoverageStatus:
        return self._state(unit_id).status

    def receipt(self, unit_id: str) -> ReadReceipt | None:
        return self._state(unit_id).receipt

    def _required_children(self, unit_id: str) -> tuple[str, ...]:
        return tuple(
            child_id for child_id in self._children[unit_id] if self._units[child_id].required
        )

    def _children_cover_range(self, unit_id: str, children: tuple[str, ...]) -> bool:
        parent = self._units[unit_id]
        if parent.range_start is None:
            return True
        if not children or any(self._units[item].range_start is None for item in children):
            return False
        cursor = parent.range_start
        for child_id in sorted(children, key=lambda item: cast(int, self._units[item].range_start)):
            child = self._units[child_id]
            child_start = cast(int, child.range_start)
            child_end = cast(int, child.range_end)
            if child_start > cursor:
                return False
            cursor = max(cursor, child_end)
            if cursor >= cast(int, parent.range_end):
                return True
        return False

    def _derived_parent_status(self, unit_id: str) -> CoverageStatus:
        required_children = self._required_children(unit_id)
        if not required_children or not self._children_cover_range(unit_id, required_children):
            return CoverageStatus.PENDING
        statuses = tuple(self._states[item].status for item in required_children)
        if CoverageStatus.PENDING in statuses:
            return CoverageStatus.PENDING
        if all(status == CoverageStatus.SKIPPED for status in statuses):
            return CoverageStatus.SKIPPED
        if any(status in {CoverageStatus.REVIEWED, CoverageStatus.COVERED} for status in statuses):
            return CoverageStatus.COVERED
        return CoverageStatus.PENDING

    def _refresh_ancestors(self, unit_id: str, iteration: int | None) -> None:
        parent_id = self._parents.get(unit_id)
        while parent_id is not None:
            state = self._states[parent_id]
            desired = self._derived_parent_status(parent_id)
            if state.status != desired:
                self._states[parent_id] = (
                    _State()
                    if desired == CoverageStatus.PENDING
                    else _State(desired, transition_iteration=iteration)
                )
            parent_id = self._parents.get(parent_id)

    def _reject_containers(self, unit_ids: Iterable[str], action: str) -> None:
        containers = [item for item in unit_ids if self._children[item]]
        if containers:
            raise CoverageError(
                f"parent containers cannot be directly {action}: {', '.join(containers)}"
            )

    def _validate_receipt(self, receipt: ReadReceipt) -> tuple[ReviewUnit, _State]:
        if not isinstance(receipt, ReadReceipt) or receipt.unit_id not in self._units:
            raise CoverageError("receipt references an unknown unit")
        unit = self._units[receipt.unit_id]
        self._reject_containers((unit.unit_id,), "receipted")
        snapshot = self._snapshot_by_unit[unit.unit_id]
        if receipt.snapshot_id != snapshot.snapshot_id or receipt.locator != unit.locator:
            raise CoverageError("receipt does not match the collection snapshot")
        if (
            receipt.range_start != unit.range_start
            or receipt.range_end != unit.range_end
            or receipt.source_revision != unit.source_revision
        ):
            raise CoverageError("receipt does not match the unit source range or revision")
        if unit.range_start is not None and receipt.byte_count != (
            cast(int, unit.range_end) - unit.range_start
        ):
            raise CoverageError("receipt byte_count does not match the unit source range")
        if not receipt.complete:
            raise CoverageError("partial reads do not count as reviewed")
        if receipt.byte_count <= 0:
            raise CoverageError("empty reads do not count as reviewed")
        if unit.content_sha256 and unit.content_sha256 != receipt.content_sha256:
            raise CoverageError("receipt content digest does not match the discovered unit")
        return unit, self._states[unit.unit_id]

    def record_read(self, receipt: ReadReceipt, *, iteration: int | None = None) -> None:
        unit, state = self._validate_receipt(receipt)
        if state.status == CoverageStatus.SKIPPED and state.harness_non_substantive:
            raise CoverageError(f"cannot review harness-classified unit {unit.unit_id!r}")
        credited_iteration = (
            state.transition_iteration if state.status == CoverageStatus.REVIEWED else iteration
        )
        state.status, state.receipt, state.covered_by, state.skip_reason = (
            CoverageStatus.REVIEWED,
            receipt,
            None,
            None,
        )
        state.transition_iteration = credited_iteration
        state.harness_non_substantive = False
        self._refresh_ancestors(unit.unit_id, credited_iteration)

    def mark_reviewed(
        self,
        unit_ids: str | Iterable[str],
        evidence: ReadReceipt | Iterable[ReadReceipt] | Mapping[str, ReadReceipt],
        *,
        iteration: int | None = None,
    ) -> None:
        unit_ids = self._ids(unit_ids)
        receipts: tuple[ReadReceipt, ...]
        if isinstance(evidence, ReadReceipt):
            receipts = (evidence,)
        elif isinstance(evidence, Mapping):
            try:
                receipt_map = cast(Mapping[str, ReadReceipt], evidence)
                receipts = tuple(receipt_map[unit_id] for unit_id in unit_ids)
            except KeyError as exc:
                raise CoverageError(f"missing read receipt for unit {exc.args[0]!r}") from exc
        else:
            receipts = tuple(evidence)
        if (
            len(receipts) != len(unit_ids)
            or not all(isinstance(receipt, ReadReceipt) for receipt in receipts)
            or {receipt.unit_id for receipt in receipts} != set(unit_ids)
        ):
            raise CoverageError("mark_reviewed requires one matching ReadReceipt per unit")
        by_id = {receipt.unit_id: receipt for receipt in receipts}
        for unit_id in unit_ids:
            _, state = self._validate_receipt(by_id[unit_id])
            if state.status == CoverageStatus.SKIPPED and state.harness_non_substantive:
                raise CoverageError(f"cannot review harness-classified unit {unit_id!r}")
        for unit_id in unit_ids:
            self.record_read(by_id[unit_id], iteration=iteration)

    def mark_reviewed_by_uri(
        self,
        uri: str,
        evidence: ReadReceipt | Iterable[ReadReceipt] | Mapping[str, ReadReceipt],
        *,
        iteration: int | None = None,
    ) -> None:
        matches = [unit.unit_id for unit in self.units if unit.locator == uri]
        if len(matches) != 1:
            raise CoverageError(f"locator {uri!r} is missing or ambiguous")
        self.mark_reviewed(matches, evidence, iteration=iteration)

    def cover_from_summary(
        self,
        summary_unit_id: str,
        receipt: ReadReceipt,
        *,
        iteration: int | None = None,
    ) -> tuple[str, ...]:
        summary = self._summaries.get(summary_unit_id)
        if summary is None:
            raise CoverageError(f"unit {summary_unit_id!r} is not a verified summary")
        if receipt.unit_id != summary_unit_id:
            raise CoverageError("summary receipt belongs to a different unit")
        blocked = [
            item
            for item in summary.covered_unit_ids
            if self._states[item].status == CoverageStatus.SKIPPED
        ]
        if blocked:
            raise CoverageError(f"verified summary cannot cover skipped units: {blocked}")
        self._reject_containers(summary.covered_unit_ids, "covered")
        self.record_read(receipt, iteration=iteration)
        return self.cover(summary.covered_unit_ids, summary_unit_id, iteration=iteration)

    def cover(
        self,
        unit_ids: str | Iterable[str],
        summary_unit_id: str,
        *,
        iteration: int | None = None,
    ) -> tuple[str, ...]:
        unit_ids = self._ids(unit_ids)
        summary = self._summaries.get(summary_unit_id)
        summary_state = self._states.get(summary_unit_id)
        if summary is None:
            raise CoverageError(f"unit {summary_unit_id!r} is not a verified summary")
        if not summary_state or summary_state.status != CoverageStatus.REVIEWED:
            raise CoverageError("verified summary must be read before it can cover members")
        outside = [item for item in unit_ids if item not in summary.covered_unit_ids]
        blocked = [item for item in unit_ids if self._states[item].status == CoverageStatus.SKIPPED]
        if outside:
            raise CoverageError(f"summary {summary_unit_id!r} does not verify members: {outside}")
        if blocked:
            raise CoverageError(f"verified summary cannot cover skipped units: {blocked}")
        self._reject_containers(unit_ids, "covered")
        changed = tuple(
            item for item in unit_ids if self._states[item].status == CoverageStatus.PENDING
        )
        for item in changed:
            self._states[item].status = CoverageStatus.COVERED
            self._states[item].covered_by = summary_unit_id
            self._states[item].transition_iteration = iteration
        for item in changed:
            self._refresh_ancestors(item, iteration)
        return changed

    def skip(
        self,
        unit_ids: str | Iterable[str],
        reason: str,
        *,
        iteration: int | None = None,
    ) -> None:
        unit_ids, reason = self._ids(unit_ids), _skip_reason(reason)
        self._reject_containers(unit_ids, "skipped")
        if any(
            self._states[item].status in {CoverageStatus.REVIEWED, CoverageStatus.COVERED}
            for item in unit_ids
        ):
            raise CoverageError("cannot skip an already evidenced unit")
        for item in unit_ids:
            self._states[item] = _State(
                CoverageStatus.SKIPPED,
                skip_reason=reason,
                transition_iteration=iteration,
            )
            self._refresh_ancestors(item, iteration)

    def mark_harness_non_substantive(
        self,
        unit_ids: str | Iterable[str],
        reason: str,
        *,
        iteration: int | None = None,
    ) -> None:
        """Record a deterministic adapter classification unavailable to the agent.

        This is intentionally not exposed through ``ReportCoverageTool``.  It is
        reserved for a Harness-owned parser that inspected the full unit and can
        prove that it contains no business content to review.
        """

        unit_ids, reason = self._ids(unit_ids), _skip_reason(reason)
        self._reject_containers(unit_ids, "classified non-substantive")
        if any(
            self._states[item].status not in {CoverageStatus.PENDING, CoverageStatus.SKIPPED}
            for item in unit_ids
        ):
            raise CoverageError("only pending or skipped units can be classified non-substantive")
        for item in unit_ids:
            if self._states[item].harness_non_substantive:
                continue
            self._states[item] = _State(
                CoverageStatus.SKIPPED,
                skip_reason=reason,
                transition_iteration=iteration,
                harness_non_substantive=True,
            )
            self._refresh_ancestors(item, iteration)

    @staticmethod
    def _visible_state(state: _State, before_iteration: int | None) -> _State:
        if (
            before_iteration is not None
            and state.transition_iteration is not None
            and state.transition_iteration >= before_iteration
        ):
            return _State()
        return state

    def completion_issues(self, *, before_iteration: int | None = None) -> tuple[str, ...]:
        visible = {
            item: self._visible_state(state, before_iteration)
            for item, state in self._states.items()
        }
        pending = [
            item
            for item, state in visible.items()
            if self._units[item].required and state.status == CoverageStatus.PENDING
        ]
        issues = [f"pending review units: {', '.join(pending)}"] if pending else []
        awaiting_delivery = [
            item
            for item, state in self._states.items()
            if before_iteration is not None
            and self._units[item].required
            and state.transition_iteration is not None
            and state.transition_iteration >= before_iteration
        ]
        if awaiting_delivery:
            issues.append(
                "current-iteration coverage evidence is not usable until its tool result is "
                f"delivered: {', '.join(awaiting_delivery)}"
            )
        for snapshot in self._snapshots:
            substantive_units = [
                unit
                for unit in snapshot.units
                if unit.required
                and unit.substantive
                and not visible[unit.unit_id].harness_non_substantive
            ]
            if substantive_units and not any(
                unit.required
                and unit.substantive
                and not visible[unit.unit_id].harness_non_substantive
                and visible[unit.unit_id].status
                in {CoverageStatus.REVIEWED, CoverageStatus.COVERED}
                for unit in snapshot.units
            ):
                issues.append(
                    f"source {snapshot.source_id!r} has no substantive reviewed or covered unit"
                )
        return tuple(issues)

    def is_complete(self) -> bool:
        return not self.completion_issues()

    def gate_error(self, *, before_iteration: int | None = None) -> str | None:
        return "; ".join(self.completion_issues(before_iteration=before_iteration)) or None

    def _counts(self, unit_ids: Iterable[str]) -> dict[str, int]:
        counts = {status.value: 0 for status in CoverageStatus}
        for item in unit_ids:
            counts[self._states[item].status.value] += 1
        return {**counts, "total": sum(counts.values())}

    def summary(self) -> dict[str, Any]:
        issues = self.completion_issues()
        return {
            "counts": self._counts(self._states),
            "sources": {
                snapshot.source_id: {
                    "snapshot_id": snapshot.snapshot_id,
                    "counts": self._counts(unit.unit_id for unit in snapshot.units),
                    "has_substantive_evidence": any(
                        unit.required
                        and unit.substantive
                        and not self._states[unit.unit_id].harness_non_substantive
                        and self._states[unit.unit_id].status
                        in {CoverageStatus.REVIEWED, CoverageStatus.COVERED}
                        for unit in snapshot.units
                    )
                    or not any(
                        unit.required
                        and unit.substantive
                        and not self._states[unit.unit_id].harness_non_substantive
                        for unit in snapshot.units
                    ),
                }
                for snapshot in self._snapshots
            },
            "complete": not issues,
            "issues": list(issues),
        }

    def to_internal_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "snapshots": [
                {**_snapshot_payload(snapshot), "snapshot_id": snapshot.snapshot_id}
                for snapshot in self._snapshots
            ],
            "states": {
                unit_id: {
                    "status": state.status.value,
                    **(
                        {
                            "receipt": {
                                **_receipt_payload(state.receipt),
                                "receipt_id": state.receipt.receipt_id,
                            }
                        }
                        if state.receipt
                        else {}
                    ),
                    **({"covered_by": state.covered_by} if state.covered_by else {}),
                    **({"skip_reason": state.skip_reason} if state.skip_reason else {}),
                    **(
                        {"transition_iteration": state.transition_iteration}
                        if state.transition_iteration is not None
                        else {}
                    ),
                    **({"harness_non_substantive": True} if state.harness_non_substantive else {}),
                }
                for unit_id, state in self._states.items()
            },
        }

    @classmethod
    def from_internal_dict(cls, value: Mapping[str, Any]) -> "CoverageLedger":
        version = value.get("version")
        if (
            isinstance(version, bool)
            or version not in _SUPPORTED_VERSIONS
            or not isinstance(value.get("states"), Mapping)
        ):
            raise CoverageError("unsupported or invalid coverage serialization")
        snapshots = []
        for raw in value.get("snapshots", []):
            snapshots.append(
                CollectionSnapshot(
                    raw["source_id"],
                    tuple(ReviewUnit(**unit) for unit in raw.get("units", [])),
                    tuple(
                        VerifiedSummary(
                            summary["summary_unit_id"],
                            tuple(summary["covered_unit_ids"]),
                            summary["verifier"],
                            summary["membership_sha256"],
                        )
                        for summary in raw.get("verified_summaries", [])
                    ),
                    raw["snapshot_id"],
                )
            )
        ledger, raw_states = cls(snapshots), value["states"]
        if set(raw_states) != set(ledger._states):
            raise CoverageError("serialized states do not match snapshot units")
        for unit_id, raw in raw_states.items():
            try:
                state = _State(CoverageStatus(raw["status"]))
                if raw.get("receipt"):
                    state.receipt = ReadReceipt(**raw["receipt"])
                    if state.receipt.unit_id != unit_id:
                        raise CoverageError("serialized receipt belongs to another unit")
                    ledger._validate_receipt(state.receipt)
                state.covered_by = raw.get("covered_by")
                state.skip_reason = (
                    _skip_reason(raw["skip_reason"]) if raw.get("skip_reason") else None
                )
                transition_iteration = raw.get("transition_iteration")
                if transition_iteration is not None and (
                    isinstance(transition_iteration, bool)
                    or not isinstance(transition_iteration, int)
                    or transition_iteration < 1
                ):
                    raise CoverageError("invalid transition iteration")
                state.transition_iteration = transition_iteration
                harness_non_substantive = raw.get("harness_non_substantive", False)
                if not isinstance(harness_non_substantive, bool):
                    raise CoverageError("invalid non-substantive classification")
                state.harness_non_substantive = harness_non_substantive
                ledger._states[unit_id] = state
            except (KeyError, TypeError, ValueError) as exc:
                raise CoverageError(f"invalid serialized state for {unit_id!r}") from exc
        ledger._validate_states()
        return ledger

    def _validate_states(self) -> None:
        for unit_id, state in self._states.items():
            if self._children[unit_id]:
                expected = self._derived_parent_status(unit_id)
                if state.status != expected:
                    raise CoverageError(
                        f"parent unit {unit_id!r} does not match its child coverage states"
                    )
                if any(
                    (
                        state.receipt,
                        state.covered_by,
                        state.skip_reason,
                        state.harness_non_substantive,
                    )
                ) or (
                    state.status == CoverageStatus.PENDING
                    and state.transition_iteration is not None
                ):
                    raise CoverageError(f"parent unit {unit_id!r} contains direct evidence")
                continue
            if state.status == CoverageStatus.PENDING and any(
                (
                    state.receipt,
                    state.covered_by,
                    state.skip_reason,
                    state.transition_iteration,
                    state.harness_non_substantive,
                )
            ):
                raise CoverageError(f"pending unit {unit_id!r} contains evidence")
            if state.status == CoverageStatus.REVIEWED and (
                not state.receipt or state.covered_by or state.skip_reason
            ):
                raise CoverageError(f"reviewed unit {unit_id!r} has invalid evidence")
            if state.status == CoverageStatus.COVERED:
                summary = self._summaries.get(state.covered_by or "")
                if (
                    state.receipt
                    or state.skip_reason
                    or not summary
                    or unit_id not in summary.covered_unit_ids
                    or self._states[summary.summary_unit_id].status != CoverageStatus.REVIEWED
                ):
                    raise CoverageError(f"covered unit {unit_id!r} has invalid evidence")
            if state.status == CoverageStatus.SKIPPED and (
                state.receipt or state.covered_by or not state.skip_reason
            ):
                raise CoverageError(f"skipped unit {unit_id!r} has invalid evidence")
            if state.harness_non_substantive and state.status != CoverageStatus.SKIPPED:
                raise CoverageError(f"non-substantive unit {unit_id!r} must be recorded as skipped")

    def _state(self, unit_id: str) -> _State:
        try:
            return self._states[unit_id]
        except KeyError as exc:
            raise CoverageError(f"unknown unit_id {unit_id!r}") from exc

    def _ids(self, unit_ids: str | Iterable[str]) -> tuple[str, ...]:
        result = (unit_ids,) if isinstance(unit_ids, str) else tuple(unit_ids)
        if not result or len(result) != len(set(result)):
            raise CoverageError("unit_ids must be non-empty and unique")
        for item in result:
            self._state(item)
        return result


__all__ = [
    "CollectionSnapshot",
    "CoverageError",
    "CoverageLedger",
    "CoverageStatus",
    "ReadReceipt",
    "ReviewUnit",
    "VerifiedSummary",
]
