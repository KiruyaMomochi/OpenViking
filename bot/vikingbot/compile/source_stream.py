"""Bounded byte-range streaming and deterministic Compile source packets.

This module is deliberately limited to harness I/O and packetization.  It never
copies a source, invokes a model, or decides whether coverage is sufficient.
"""

from __future__ import annotations

import codecs
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from vikingbot.compile.adapters import (
    AdapterRegistry,
    CodexRolloutStreamDecoder,
    ReviewMessage,
    SourceAdapter,
    StreamRecord,
)


class SourceStreamError(RuntimeError):
    """Raised when a source cannot be streamed without violating its contract."""


class SourceStreamBudgetError(SourceStreamError):
    """Raised before a configured streaming budget can be exceeded."""


class RangeReader(Protocol):
    """Minimal reader needed by :func:`stream_source`.

    Implementations may return either an attribute object (for example the SDK's
    ``RangeRead``) or a mapping with the same fields.
    """

    async def read_range(
        self,
        locator: str,
        *,
        offset: int = 0,
        size: int = 16 * 1024 * 1024,
        expected_revision: str | None = None,
    ) -> Any: ...


PacketGroupConsumer = Callable[[tuple["StreamPacket", ...]], Awaitable[bool]]


@dataclass(slots=True)
class SourceReadMeter:
    """Task-wide byte/call counter shared by every streamed source."""

    max_bytes: int
    bytes_read: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        for name in ("max_bytes", "bytes_read", "calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.bytes_read > self.max_bytes:
            raise ValueError("bytes_read cannot exceed max_bytes")

    @property
    def remaining_bytes(self) -> int:
        return self.max_bytes - self.bytes_read


@dataclass(frozen=True, slots=True)
class SourceStreamLimits:
    """Hard bounds for a single streamed source."""

    range_bytes: int = 8 * 1024 * 1024
    max_io_bytes: int = 2 * 1024 * 1024 * 1024
    max_packets: int = 20_000
    max_record_bytes: int = 32 * 1024 * 1024
    packet_content_chars: int = 300_000

    def __post_init__(self) -> None:
        for name in (
            "range_bytes",
            "max_io_bytes",
            "max_packets",
            "max_record_bytes",
            "packet_content_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.range_bytes > 16 * 1024 * 1024:
            raise ValueError("range_bytes cannot exceed the 16 MiB read_range limit")


@dataclass(frozen=True, slots=True)
class StreamPacket:
    """One bounded semantic or unsupported slice of a streamed source."""

    packet_id: str
    source_id: str
    locator: str
    range_start: int
    range_end: int
    source_revision: str
    content: str = field(repr=False)
    substantive: bool
    supported: bool
    kind: str
    reason: str | None = None
    fragment_index: int | None = None
    fragment_count: int | None = None

    def __post_init__(self) -> None:
        if not self.packet_id or not self.source_id or not self.locator:
            raise ValueError("packet_id, source_id, and locator must be non-empty")
        if self.range_start < 0 or self.range_end < self.range_start:
            raise ValueError("packet byte range is invalid")
        if not self.source_revision:
            raise ValueError("source_revision must be non-empty")
        if not self.kind:
            raise ValueError("packet kind must be non-empty")
        fragmented = self.fragment_index is not None or self.fragment_count is not None
        if fragmented:
            if self.fragment_index is None or self.fragment_count is None:
                raise ValueError("fragment_index and fragment_count must be set together")
            if not 0 <= self.fragment_index < self.fragment_count:
                raise ValueError("fragment metadata is invalid")


@dataclass(frozen=True, slots=True)
class SourceStreamResult:
    """Bounded stream result for one immutable source revision."""

    packets: tuple[StreamPacket, ...]
    total_size: int
    revision: str
    io_bytes: int
    warnings: tuple[str, ...] = ()
    complete: bool = True
    pending_start: int | None = None
    stop_reason: str | None = None
    read_calls: int = 0

    def __post_init__(self) -> None:
        if self.complete:
            if self.pending_start is not None or self.stop_reason is not None:
                raise ValueError("a complete stream cannot have a pending tail")
        else:
            if self.pending_start is None or not 0 <= self.pending_start < self.total_size:
                raise ValueError("an incomplete stream requires a valid pending_start")
            if not self.stop_reason:
                raise ValueError("an incomplete stream requires a stop_reason")


@dataclass(frozen=True, slots=True)
class _Range:
    content: bytes
    start: int
    end: int
    next_offset: int | None
    eof: bool
    total_size: int
    revision: str


class _PacketSink:
    def __init__(
        self,
        *,
        source_id: str,
        locator: str,
        revision: str,
        limits: SourceStreamLimits,
    ) -> None:
        self.source_id = source_id
        self.locator = locator
        self.revision = revision
        self.limits = limits
        self.packets: list[StreamPacket] = []
        self.warnings: list[str] = []
        self.packet_count = 0

    def add(
        self,
        *,
        start: int,
        end: int,
        content: str,
        substantive: bool,
        supported: bool,
        kind: str,
        reason: str | None = None,
        fragment_index: int | None = None,
        fragment_count: int | None = None,
    ) -> None:
        if len(content) > self.limits.packet_content_chars:
            raise SourceStreamBudgetError("packet content character budget exceeded")
        if fragment_index == 0 and fragment_count is not None:
            if self.packet_count + fragment_count > self.limits.max_packets:
                raise SourceStreamBudgetError("source packet budget exceeded")
        elif self.packet_count >= self.limits.max_packets:
            raise SourceStreamBudgetError("source packet budget exceeded")
        identity = "\0".join(
            (
                self.source_id,
                self.locator,
                self.revision,
                str(start),
                str(end),
                kind,
                str(fragment_index) if fragment_index is not None else "-",
                str(fragment_count) if fragment_count is not None else "-",
            )
        )
        packet_id = f"stream::{hashlib.sha256(identity.encode()).hexdigest()[:32]}"
        self.packets.append(
            StreamPacket(
                packet_id=packet_id,
                source_id=self.source_id,
                locator=self.locator,
                range_start=start,
                range_end=end,
                source_revision=self.revision,
                content=content,
                substantive=substantive,
                supported=supported,
                kind=kind,
                reason=reason,
                fragment_index=fragment_index,
                fragment_count=fragment_count,
            )
        )
        self.packet_count += 1
        if reason:
            self.warnings.append(reason)

    def drain(self) -> tuple[StreamPacket, ...]:
        packets = tuple(self.packets)
        self.packets.clear()
        return packets


class _GenericPacketizer:
    """Incremental UTF-8 packetizer preserving exact consumed byte offsets."""

    def __init__(self, sink: _PacketSink) -> None:
        self.sink = sink
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")
        self.output_start = 0

    def feed(self, raw: bytes, *, final: bool) -> None:
        decoded = self.decoder.decode(raw, final=final)
        if not decoded:
            return
        run_start = 0
        run_unsupported = _is_surrogate(decoded[0])
        for index in range(1, len(decoded)):
            unsupported = _is_surrogate(decoded[index])
            if unsupported != run_unsupported:
                self._emit_run(decoded[run_start:index], unsupported=run_unsupported)
                run_start = index
                run_unsupported = unsupported
        self._emit_run(decoded[run_start:], unsupported=run_unsupported)

    def _emit_run(self, text: str, *, unsupported: bool) -> None:
        if not text:
            return
        if unsupported:
            raw = text.encode("utf-8", errors="surrogateescape")
            start = self.output_start
            self.output_start += len(raw)
            self.sink.add(
                start=start,
                end=self.output_start,
                content="",
                substantive=True,
                supported=False,
                kind="invalid_utf8",
                reason=f"invalid UTF-8 bytes at range [{start}, {self.output_start})",
            )
            return

        limit = self.sink.limits.packet_content_chars
        for char_start in range(0, len(text), limit):
            content = text[char_start : char_start + limit]
            raw_size = len(content.encode("utf-8"))
            start = self.output_start
            self.output_start += raw_size
            substantive = bool(content.strip())
            self.sink.add(
                start=start,
                end=self.output_start,
                content=content,
                substantive=substantive,
                supported=True,
                kind="text" if substantive else "structural_text",
            )


class _CodexPacketizer:
    def __init__(self, sink: _PacketSink, decoder: CodexRolloutStreamDecoder) -> None:
        self.sink = sink
        self.decoder = decoder
        self.pending_start: int | None = None
        self.pending_end: int | None = None
        self.pending_parts: list[str] = []
        self.pending_content_chars = 0
        self.pending_substantive = False

    def feed(self, raw: bytes) -> None:
        self._consume(self.decoder.feed(raw))

    def finish(self) -> None:
        self._consume(self.decoder.finish())
        self._flush()

    def _consume(self, records: tuple[StreamRecord, ...]) -> None:
        for record in records:
            if record.disposition == "unsupported":
                self._flush()
                self.sink.add(
                    start=record.start,
                    end=record.end,
                    content=record.opaque_excerpt or "",
                    substantive=True,
                    supported=False,
                    kind="codex_unsupported",
                    reason=record.warning or "unsupported Codex rollout record",
                )
                continue

            if record.disposition != "emitted":
                self._extend_empty(record.start, record.end)
                continue

            rendered = _render_messages(record.messages)
            if len(rendered) > self.sink.limits.packet_content_chars:
                self._flush()
                self._emit_fragments(record, rendered)
                continue

            if self.pending_parts and (
                self.pending_content_chars + 2 + len(rendered)
                > self.sink.limits.packet_content_chars
            ):
                self._flush()
            self._extend(record.start, record.end, rendered, substantive=True)

    def _extend_empty(self, start: int, end: int) -> None:
        if self.pending_start is None:
            self.pending_start = start
        elif self.pending_end != start:
            self._flush()
            self.pending_start = start
        self.pending_end = end

    def _extend(self, start: int, end: int, content: str, *, substantive: bool) -> None:
        if self.pending_start is None:
            self.pending_start = start
        elif self.pending_end != start:
            self._flush()
            self.pending_start = start
        self.pending_end = end
        if content:
            if self.pending_parts:
                self.pending_content_chars += 2
            self.pending_parts.append(content)
            self.pending_content_chars += len(content)
        self.pending_substantive = self.pending_substantive or substantive

    def _flush(self) -> None:
        if self.pending_start is None or self.pending_end is None:
            return
        content = "\n\n".join(self.pending_parts)
        self.sink.add(
            start=self.pending_start,
            end=self.pending_end,
            content=content,
            substantive=self.pending_substantive,
            supported=True,
            kind="codex_messages" if self.pending_substantive else "codex_structural",
        )
        self.pending_start = None
        self.pending_end = None
        self.pending_parts = []
        self.pending_content_chars = 0
        self.pending_substantive = False

    def _emit_fragments(self, record: StreamRecord, rendered: str) -> None:
        limit = self.sink.limits.packet_content_chars
        fragments = tuple(
            rendered[index : index + limit] for index in range(0, len(rendered), limit)
        )
        for index, fragment in enumerate(fragments):
            self.sink.add(
                start=record.start,
                end=record.end,
                content=fragment,
                substantive=bool(fragment.strip()),
                supported=True,
                kind=(
                    "codex_message_fragment"
                    if fragment.strip()
                    else "codex_structural_fragment"
                ),
                fragment_index=index,
                fragment_count=len(fragments),
            )


async def stream_source(
    reader: RangeReader,
    *,
    source_id: str,
    locator: str,
    adapter_registry: AdapterRegistry | None = None,
    adapter: SourceAdapter | None = None,
    limits: SourceStreamLimits | None = None,
    on_packet_group: PacketGroupConsumer | None = None,
    meter: SourceReadMeter | None = None,
) -> SourceStreamResult:
    """Stream an immutable source revision into bounded deterministic packets.

    ``read_range`` is the only source access used.  A candidate adapter failing
    to decode is surfaced as an error; it never falls back to a whole/raw read.
    """

    source_id = str(source_id).strip()
    locator = str(locator).strip()
    if not source_id or not locator:
        raise ValueError("source_id and locator must be non-empty")
    limits = limits or SourceStreamLimits()
    selected = adapter
    if selected is None:
        selected = (adapter_registry or AdapterRegistry()).candidate(locator)

    offset = 0
    revision: str | None = None
    total_size: int | None = None
    io_bytes = 0
    read_calls = 0
    sink: _PacketSink | None = None
    packetizer: _GenericPacketizer | _CodexPacketizer | None = None
    retained: list[StreamPacket] = []
    accepted_end = 0

    async def deliver_emitted() -> tuple[bool, str | None]:
        nonlocal accepted_end
        assert sink is not None
        emitted = sink.drain()
        if not emitted:
            return True, None
        groups = _packet_groups(emitted)
        for group in groups:
            if on_packet_group is not None:
                try:
                    accepted = await on_packet_group(group)
                except Exception as exc:
                    detail = str(exc).replace("\n", " ").strip()[:240]
                    return False, (
                        f"packet consumer failed ({type(exc).__name__}: {detail or 'no detail'})"
                    )
                if not accepted:
                    return False, "packet consumer budget exhausted"
            retained.extend(
                group
                if on_packet_group is None
                else tuple(replace(item, content="") for item in group)
            )
            accepted_end = max(accepted_end, group[0].range_end)
        return True, None

    async def partial(reason: str) -> SourceStreamResult:
        if sink is not None:
            accepted, consumer_reason = await deliver_emitted()
            if not accepted and consumer_reason:
                reason = consumer_reason
        assert revision is not None and total_size is not None
        if accepted_end >= total_size:
            _validate_packet_coverage(retained, total_size=total_size)
            return SourceStreamResult(
                packets=tuple(retained),
                total_size=total_size,
                revision=revision,
                io_bytes=io_bytes,
                warnings=tuple(sink.warnings if sink is not None else ()),
                read_calls=read_calls,
            )
        _validate_packet_prefix(retained, pending_start=accepted_end)
        return SourceStreamResult(
            packets=tuple(retained),
            total_size=total_size,
            revision=revision,
            io_bytes=io_bytes,
            warnings=tuple(sink.warnings if sink is not None else ()),
            complete=False,
            pending_start=accepted_end,
            stop_reason=reason,
            read_calls=read_calls,
        )

    while True:
        remaining_io = limits.max_io_bytes - io_bytes
        if meter is not None:
            remaining_io = min(remaining_io, meter.remaining_bytes)
        if remaining_io <= 0:
            error = SourceStreamBudgetError("source I/O byte budget exceeded")
            if on_packet_group is not None and revision is not None and total_size is not None:
                return await partial(str(error))
            raise error
        request_size = min(limits.range_bytes, remaining_io)
        read_calls += 1
        if meter is not None:
            meter.calls += 1
        try:
            raw_response = await reader.read_range(
                locator,
                offset=offset,
                size=request_size,
                expected_revision=revision,
            )
        except Exception as exc:
            if (
                on_packet_group is not None
                and revision is not None
                and total_size is not None
                and not _invalidates_revision(exc)
            ):
                detail = str(exc).replace("\n", " ").strip()[:240]
                return await partial(
                    f"range read failed ({type(exc).__name__}: {detail or 'no detail'})"
                )
            raise
        response = _coerce_range(raw_response)
        io_bytes += len(response.content)
        if meter is not None:
            meter.bytes_read += len(response.content)
            if meter.bytes_read > meter.max_bytes:  # pragma: no cover - hostile server guard
                raise SourceStreamBudgetError("source stage returned bytes past its hard budget")
        try:
            _validate_range(response, expected_offset=offset)
        except SourceStreamError as exc:
            if on_packet_group is not None and revision is not None and total_size is not None:
                detail = str(exc).replace("\n", " ").strip()[:240]
                return await partial(f"invalid range response ({detail})")
            raise

        if revision is None:
            revision = response.revision
            total_size = response.total_size
            if total_size > limits.max_io_bytes:
                error = SourceStreamBudgetError(
                    f"source size {total_size} exceeds I/O byte budget {limits.max_io_bytes}"
                )
                if on_packet_group is not None:
                    return await partial(str(error))
                raise error
            if total_size == 0:
                return SourceStreamResult(
                    packets=(),
                    total_size=0,
                    revision=revision,
                    io_bytes=io_bytes,
                    read_calls=read_calls,
                )
            sink = _PacketSink(
                source_id=source_id,
                locator=locator,
                revision=revision,
                limits=limits,
            )
            if selected is None:
                packetizer = _GenericPacketizer(sink)
            else:
                decoder = selected.stream_decoder(
                    locator,
                    max_record_bytes=limits.max_record_bytes,
                )
                packetizer = _CodexPacketizer(sink, decoder)
        elif response.revision != revision or response.total_size != total_size:
            raise SourceStreamError("source revision or size changed while streaming")

        assert packetizer is not None
        try:
            if isinstance(packetizer, _CodexPacketizer):
                packetizer.feed(response.content)
                if response.eof:
                    packetizer.finish()
            else:
                packetizer.feed(response.content, final=response.eof)
        except Exception as exc:
            if on_packet_group is None:
                raise
            detail = str(exc).replace("\n", " ").strip()[:240]
            return await partial(
                f"source decoder stopped ({type(exc).__name__}: {detail or 'no detail'})"
            )

        accepted, stop_reason = await deliver_emitted()
        if not accepted:
            assert stop_reason is not None
            return await partial(stop_reason)
        if response.eof:
            break
        assert response.next_offset is not None
        offset = response.next_offset

    assert sink is not None and revision is not None and total_size is not None
    if io_bytes != total_size:
        raise SourceStreamError(
            f"stream consumed {io_bytes} bytes but source reports {total_size} bytes"
        )
    _validate_packet_coverage(retained, total_size=total_size)
    return SourceStreamResult(
        packets=tuple(retained),
        total_size=total_size,
        revision=revision,
        io_bytes=io_bytes,
        warnings=tuple(sink.warnings),
        read_calls=read_calls,
    )


def _packet_groups(packets: tuple[StreamPacket, ...]) -> tuple[tuple[StreamPacket, ...], ...]:
    groups: list[tuple[StreamPacket, ...]] = []
    index = 0
    while index < len(packets):
        packet = packets[index]
        if packet.fragment_count is None:
            groups.append((packet,))
            index += 1
            continue
        end = index + packet.fragment_count
        group = packets[index:end]
        if (
            len(group) != packet.fragment_count
            or packet.fragment_index != 0
            or tuple(item.fragment_index for item in group) != tuple(range(len(group)))
            or len({(item.range_start, item.range_end) for item in group}) != 1
        ):
            raise SourceStreamError("fragmented source packet group is incomplete")
        groups.append(group)
        index = end
    return tuple(groups)


def _invalidates_revision(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == "CONFLICT" or (
        isinstance(exc, SourceStreamError) and "revision or size changed" in str(exc)
    )


def _coerce_range(value: Any) -> _Range:
    def field(name: str) -> Any:
        if isinstance(value, Mapping):
            try:
                return value[name]
            except KeyError as exc:
                raise SourceStreamError(f"read_range result is missing {name!r}") from exc
        try:
            return getattr(value, name)
        except AttributeError as exc:
            raise SourceStreamError(f"read_range result is missing {name!r}") from exc

    content = field("content")
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise SourceStreamError("read_range content must be bytes")
    try:
        return _Range(
            content=bytes(content),
            start=int(field("start")),
            end=int(field("end")),
            next_offset=(None if field("next_offset") is None else int(field("next_offset"))),
            eof=field("eof"),
            total_size=int(field("total_size")),
            revision=field("revision"),
        )
    except (TypeError, ValueError) as exc:
        raise SourceStreamError("read_range returned invalid metadata") from exc


def _validate_range(value: _Range, *, expected_offset: int) -> None:
    if not isinstance(value.eof, bool):
        raise SourceStreamError("read_range eof must be a boolean")
    if not isinstance(value.revision, str) or not value.revision:
        raise SourceStreamError("read_range revision must be a non-empty string")
    if value.total_size < 0:
        raise SourceStreamError("read_range total_size cannot be negative")
    if value.start != expected_offset:
        raise SourceStreamError(f"read_range started at {value.start}, expected {expected_offset}")
    if value.end != value.start + len(value.content):
        raise SourceStreamError("read_range end does not match returned content length")
    if value.end > value.total_size:
        raise SourceStreamError("read_range extends beyond reported source size")
    if value.eof:
        if value.end != value.total_size or value.next_offset is not None:
            raise SourceStreamError("EOF range metadata is inconsistent")
    else:
        if not value.content:
            raise SourceStreamError("non-EOF read_range returned no progress")
        if value.next_offset != value.end:
            raise SourceStreamError("read_range next_offset is inconsistent")


def _render_messages(messages: tuple[ReviewMessage, ...]) -> str:
    return "\n\n".join(f"[{message.role.upper()}]\n{message.text}" for message in messages)


def _is_surrogate(character: str) -> bool:
    return "\udc80" <= character <= "\udcff"


def _validate_packet_coverage(packets: list[StreamPacket], *, total_size: int) -> None:
    if total_size == 0:
        if packets:
            raise SourceStreamError("empty source produced non-empty packet ranges")
        return
    ranges = sorted({(packet.range_start, packet.range_end) for packet in packets})
    if not ranges or ranges[0][0] != 0:
        raise SourceStreamError("packets do not start at byte zero")
    covered_end = 0
    for start, end in ranges:
        if start > covered_end:
            raise SourceStreamError(f"packet coverage gap at byte range [{covered_end}, {start})")
        covered_end = max(covered_end, end)
    if covered_end != total_size:
        raise SourceStreamError(f"packets cover through byte {covered_end}, expected {total_size}")


def _validate_packet_prefix(
    packets: list[StreamPacket],
    *,
    pending_start: int,
) -> None:
    if pending_start == 0:
        if packets:
            raise SourceStreamError("zero-length source prefix produced packet ranges")
        return
    _validate_packet_coverage(packets, total_size=pending_start)


__all__ = [
    "PacketGroupConsumer",
    "RangeReader",
    "SourceReadMeter",
    "SourceStreamBudgetError",
    "SourceStreamError",
    "SourceStreamLimits",
    "SourceStreamResult",
    "StreamPacket",
    "stream_source",
]
