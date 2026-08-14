"""Incremental large-source preprocessing for Compile.

Source bytes are read only through bounded ``read_range`` calls.  Packet text is
mapped and released incrementally; snapshots retain only range metadata and
verified summaries, never a second copy of the source or temporary target files.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from vikingbot.compile.coverage import CollectionSnapshot, ReviewUnit, VerifiedSummary
from vikingbot.compile.models import CompileReviewView
from vikingbot.compile.source_scheduler import (
    SourceSchedulerBudget,
    SourceSummaryScheduler,
    SummaryOutcome,
    SummaryPacket,
)
from vikingbot.compile.source_stream import (
    SourceReadMeter,
    SourceStreamLimits,
    SourceStreamResult,
    StreamPacket,
    stream_source,
)


@dataclass(frozen=True, slots=True)
class LargeSourceSpec:
    source_id: str
    locator: str
    discovered_bytes: int | None


@dataclass(frozen=True, slots=True)
class PreparedLargeSources:
    snapshots: tuple[CollectionSnapshot, ...]
    review_views: dict[str, CompileReviewView]
    warnings: tuple[str, ...]
    harness_skips: tuple[tuple[str, str], ...]
    io_bytes: int
    map_packets: int


@dataclass(frozen=True, slots=True)
class _SummaryView:
    summary_id: str
    locator: str
    content: str
    source_id: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Streamed:
    spec: LargeSourceSpec
    result: SourceStreamResult


@dataclass(frozen=True, slots=True)
class _MapMember:
    spec: LargeSourceSpec
    packet: StreamPacket
    section: str


class _IncrementalMapper:
    """Batch complete map prompts while keeping resident source text bounded."""

    def __init__(
        self,
        scheduler: SourceSummaryScheduler,
        budget: SourceSchedulerBudget,
        *,
        source_id: str,
        reason: str,
        skill_contract: str,
    ) -> None:
        self.scheduler = scheduler
        self.budget = budget
        self.source_id = source_id
        self.reason = reason
        self.skill_contract = skill_contract
        self.current: list[_MapMember] = []
        self.queued: list[list[_MapMember]] = []
        self.outcomes: list[SummaryOutcome] = []

    async def accept_group(
        self,
        spec: LargeSourceSpec,
        group: tuple[StreamPacket, ...],
    ) -> bool:
        members = [
            _MapMember(spec, packet, _map_section(spec, packet))
            for packet in group
            if packet.supported and packet.substantive and packet.content.strip()
        ]
        if not members:
            return True

        combined = [*self.current, *members]
        if self._fits(combined) and self._can_eventually_schedule(combined):
            self.current = combined
            return True

        await self._queue_current()
        plans = self._partition(members)
        if plans is None or not self._plans_fit_stage(plans):
            return False

        # Admission above covers the whole raw fragment group before any of its
        # plans are queued.  Large groups may then drain in bounded batches.
        for plan in plans:
            if self.budget.output_chars >= self.scheduler.limits.total_output_chars:
                return False
            await self._queue_plan(plan)
        return True

    async def flush(self) -> None:
        await self._queue_current()
        await self._run_queued()

    async def seal_file(self) -> None:
        """Close one physical revision domain without serializing provider calls."""

        await self._queue_current()
        # Every verified summary is non-empty.  If one more successful call is
        # guaranteed to exhaust the remaining output budget, resolve the
        # queued call before spending I/O on another source.
        if (
            self.queued
            and self.scheduler.limits.total_output_chars - self.budget.output_chars <= 1
        ):
            await self._run_queued()

    async def _queue_current(self) -> None:
        if not self.current:
            return
        members = self.current
        self.current = []
        await self._queue_plan(members)

    async def _queue_plan(self, members: list[_MapMember]) -> None:
        self.queued.append(members)
        if len(self.queued) >= self.scheduler.limits.concurrency:
            await self._run_queued()

    async def _run_queued(self) -> None:
        if not self.queued:
            return
        packets = tuple(_make_map_packet(self.source_id, plan) for plan in self.queued)
        self.queued = []
        self.outcomes.extend(
            await self.scheduler.summarize(
                packets,
                reason=self.reason,
                skill_contract=self.skill_contract,
                budget=self.budget,
            )
        )

    def definitely_exhausted(self) -> bool:
        """Return whether no later source can make another map call."""

        reserved = self._budget_with_reservations(include_current=True)
        minimal = SummaryPacket(
            packet_id="probe",
            source_id=self.source_id,
            unit_ids=("probe",),
            content="x",
        )
        return self.scheduler.admission_error(
            minimal,
            reason=self.reason,
            skill_contract=self.skill_contract,
            budget=reserved,
        ) is not None

    def invalidate(self, unit_ids: set[str]) -> None:
        """Discard evidence derived from a source whose snapshot changed."""

        if not unit_ids:
            return
        self.current = [
            member for member in self.current if _leaf_id(member.packet) not in unit_ids
        ]
        self.queued = [
            remaining
            for plan in self.queued
            if (
                remaining := [
                    member for member in plan if _leaf_id(member.packet) not in unit_ids
                ]
            )
        ]
        self.outcomes = [
            outcome for outcome in self.outcomes if unit_ids.isdisjoint(outcome.unit_ids)
        ]

    def _fits(self, members: list[_MapMember]) -> bool:
        if not members:
            return True
        try:
            chars = self.scheduler.estimate_input_chars(
                _make_map_packet(self.source_id, members),
                reason=self.reason,
                skill_contract=self.skill_contract,
            )
        except (TypeError, ValueError):
            return False
        return chars <= self.scheduler.limits.per_packet_input_chars

    def _can_eventually_schedule(self, members: list[_MapMember]) -> bool:
        return self._reserve(members, self._budget_with_reservations())

    def _partition(self, members: list[_MapMember]) -> list[list[_MapMember]] | None:
        plans: list[list[_MapMember]] = []
        current: list[_MapMember] = []
        for member in members:
            candidate = [*current, member]
            if self._fits(candidate):
                current = candidate
                continue
            if current:
                plans.append(current)
                current = []
            if not self._fits([member]):
                return None
            current = [member]
        if current:
            plans.append(current)
        return plans

    def _plans_fit_stage(self, plans: list[list[_MapMember]]) -> bool:
        probe = self._budget_with_reservations()
        return all(self._reserve(members, probe) for members in plans)

    def _budget_with_reservations(
        self,
        *,
        include_current: bool = False,
    ) -> SourceSchedulerBudget:
        probe = SourceSchedulerBudget(
            calls=self.budget.calls,
            input_chars=self.budget.input_chars,
            output_chars=self.budget.output_chars,
        )
        for members in self.queued:
            probe.calls += 1
            probe.input_chars += self.scheduler.estimate_input_chars(
                _make_map_packet(self.source_id, members),
                reason=self.reason,
                skill_contract=self.skill_contract,
            )
        if include_current and self.current:
            probe.calls += 1
            probe.input_chars += self.scheduler.estimate_input_chars(
                _make_map_packet(self.source_id, self.current),
                reason=self.reason,
                skill_contract=self.skill_contract,
            )
        return probe

    def _reserve(self, members: list[_MapMember], probe: SourceSchedulerBudget) -> bool:
        packet = _make_map_packet(self.source_id, members)
        if (
            self.scheduler.admission_error(
                packet,
                reason=self.reason,
                skill_contract=self.skill_contract,
                budget=probe,
            )
            is not None
        ):
            return False
        probe.calls += 1
        probe.input_chars += self.scheduler.estimate_input_chars(
            packet,
            reason=self.reason,
            skill_contract=self.skill_contract,
        )
        return True


async def prepare_large_sources(
    reader,
    specs: Iterable[LargeSourceSpec],
    *,
    scheduler: SourceSummaryScheduler,
    stream_limits: SourceStreamLimits,
    reason: str,
    skill_contract: str,
    view_delivery_bytes: int = 1024 * 1024,
) -> PreparedLargeSources:
    """Incrementally map immutable inputs into evidence-bound summary views."""

    specs = tuple(specs)
    if not specs:
        return PreparedLargeSources((), {}, (), (), 0, 0)
    if view_delivery_bytes <= 0:
        raise ValueError("view_delivery_bytes must be positive")

    warnings: list[str] = []
    streamed: list[_Streamed] = []
    failed: list[tuple[LargeSourceSpec, str]] = []
    meter = SourceReadMeter(stream_limits.max_io_bytes)
    stage_budget = SourceSchedulerBudget()
    remaining_packets = stream_limits.max_packets
    outcomes: list[SummaryOutcome] = []
    effective_packet_chars = _stream_packet_limit(
        scheduler,
        reason=reason,
        skill_contract=skill_contract,
        stream_limit=stream_limits.packet_content_chars,
    )

    specs_by_source: dict[str, list[LargeSourceSpec]] = defaultdict(list)
    source_order: list[str] = []
    for spec in specs:
        if spec.source_id not in specs_by_source:
            source_order.append(spec.source_id)
        specs_by_source[spec.source_id].append(spec)

    for source_id in source_order:
        mapper = _IncrementalMapper(
            scheduler,
            stage_budget,
            source_id=source_id,
            reason=reason,
            skill_contract=skill_contract,
        )
        for spec in specs_by_source[source_id]:
            if mapper.definitely_exhausted():
                await mapper.flush()
                failed.append((spec, "source-stage map budget exhausted"))
                warnings.append(
                    f"Large source was not read because the map budget was exhausted: "
                    f"{spec.locator}"
                )
                continue
            if spec.discovered_bytes is not None and spec.discovered_bytes < 0:
                failed.append((spec, "inventory reported an invalid negative size"))
                warnings.append(
                    f"Large source has an invalid inventory size and remains pending: "
                    f"{spec.locator}"
                )
                continue
            if remaining_packets <= 0:
                failed.append((spec, "source-stage packet budget exhausted"))
                warnings.append(
                    f"Large source was not read because the packet budget was exhausted: "
                    f"{spec.locator}"
                )
                continue
            if meter.remaining_bytes <= 0 or (
                spec.discovered_bytes is not None
                and spec.discovered_bytes > 0
                and spec.discovered_bytes > meter.remaining_bytes
            ):
                failed.append((spec, "source-stage I/O budget exhausted"))
                warnings.append(
                    f"Large source exceeds the remaining I/O budget and remains pending: "
                    f"{spec.locator}"
                )
                continue

            accepted_ids: set[str] = set()

            async def accept_group(
                group: tuple[StreamPacket, ...],
                *,
                _spec: LargeSourceSpec = spec,
                _mapper: _IncrementalMapper = mapper,
                _accepted_ids: set[str] = accepted_ids,
            ) -> bool:
                accepted = await _mapper.accept_group(_spec, group)
                if accepted:
                    _accepted_ids.update(_leaf_id(packet) for packet in group)
                return accepted

            try:
                result = await stream_source(
                    reader,
                    source_id=spec.source_id,
                    locator=spec.locator,
                    limits=SourceStreamLimits(
                        range_bytes=stream_limits.range_bytes,
                        max_io_bytes=max(1, meter.remaining_bytes),
                        max_packets=remaining_packets,
                        max_record_bytes=stream_limits.max_record_bytes,
                        packet_content_chars=effective_packet_chars,
                    ),
                    on_packet_group=accept_group,
                    meter=meter,
                )
            except Exception as exc:
                mapper.invalidate(accepted_ids)
                detail = str(exc).replace("\n", " ").strip()[:240]
                failed.append((spec, detail or type(exc).__name__))
                warnings.append(
                    f"Large source could not be streamed and remains pending: {spec.locator} "
                    f"({type(exc).__name__}: {detail})"
                )
                continue
            if spec.discovered_bytes is not None and result.total_size != spec.discovered_bytes:
                mapper.invalidate(accepted_ids)
                failed.append((spec, "source size changed from the inventory snapshot"))
                warnings.append(
                    f"Large source size changed from {spec.discovered_bytes} to "
                    f"{result.total_size} and remains pending: {spec.locator}"
                )
                continue
            streamed.append(_Streamed(spec, result))
            remaining_packets -= len(result.packets)
            warnings.extend(result.warnings)
            if not result.complete:
                warnings.append(
                    f"Large source retained a verified prefix and left bytes "
                    f"[{result.pending_start}, {result.total_size}) pending: {spec.locator} "
                    f"({result.stop_reason})"
                )
            # Never let one map packet span physical revision domains.  Keep
            # completed files queued, though, so independent map calls can use
            # the configured cross-file concurrency instead of waiting one
            # timeout per file.
            await mapper.seal_file()
        await mapper.flush()
        outcomes.extend(mapper.outcomes)

    units_by_source: dict[str, list[ReviewUnit]] = defaultdict(list)
    harness_skips: list[tuple[str, str]] = []
    outcome_by_leaf = {unit_id: outcome for outcome in outcomes for unit_id in outcome.unit_ids}

    for spec, _error in failed:
        units_by_source[spec.source_id].append(
            ReviewUnit(
                unit_id=_id("unread", spec.source_id, spec.locator),
                source_id=spec.source_id,
                locator=spec.locator,
                kind="unread_source",
                discovered_bytes=spec.discovered_bytes,
            )
        )

    for item in streamed:
        _append_stream_units(
            item,
            units_by_source=units_by_source,
            harness_skips=harness_skips,
            warnings=warnings,
            outcome_by_leaf=outcome_by_leaf,
        )

    valid_unit_ids = {unit.unit_id for units in units_by_source.values() for unit in units}
    summary_views_by_source: dict[str, list[_SummaryView]] = defaultdict(list)
    for outcome in outcomes:
        if (
            not outcome.verified
            or not outcome.summary
            or not set(outcome.unit_ids) <= valid_unit_ids
        ):
            continue
        summary_id = _id("summary", outcome.packet_id)
        locator = _summary_locator(outcome.locator or f"viking://{outcome.source_id}", summary_id)
        summary_views_by_source[outcome.source_id].append(
            _SummaryView(
                summary_id=summary_id,
                locator=locator,
                content=_summary_body(outcome),
                source_id=outcome.source_id,
                member_ids=outcome.unit_ids,
            )
        )

    summaries_by_source: dict[str, list[VerifiedSummary]] = defaultdict(list)
    review_views: dict[str, CompileReviewView] = {}
    for source_id in source_order:
        packed_views, view_warnings = _pack_summary_views(
            summary_views_by_source[source_id],
            max_delivery_bytes=view_delivery_bytes,
        )
        warnings.extend(view_warnings)
        for view in packed_views:
            units_by_source[source_id].append(
                ReviewUnit(
                    unit_id=view.summary_id,
                    source_id=source_id,
                    locator=view.locator,
                    kind="summary",
                    substantive=True,
                    content_sha256=hashlib.sha256(view.content.encode()).hexdigest(),
                    discovered_bytes=len(view.content.encode()),
                    required=False,
                )
            )
            summaries_by_source[source_id].append(
                VerifiedSummary(
                    view.summary_id,
                    view.member_ids,
                    "compile-source-map-v1",
                )
            )
            review_views[view.locator] = CompileReviewView(
                view.content,
                (view.summary_id,),
            )

    snapshots = tuple(
        CollectionSnapshot(
            source_id,
            tuple(units_by_source[source_id]),
            tuple(summaries_by_source[source_id]),
        )
        for source_id in source_order
    )
    return PreparedLargeSources(
        snapshots=snapshots,
        review_views=review_views,
        warnings=tuple(dict.fromkeys(warnings)),
        harness_skips=tuple(harness_skips),
        io_bytes=meter.bytes_read,
        map_packets=stage_budget.calls,
    )


def _append_stream_units(
    item: _Streamed,
    *,
    units_by_source: dict[str, list[ReviewUnit]],
    harness_skips: list[tuple[str, str]],
    warnings: list[str],
    outcome_by_leaf: dict[str, SummaryOutcome],
) -> None:
    spec, result = item.spec, item.result
    source_units = units_by_source[spec.source_id]
    root_id = _id("source", spec.source_id, spec.locator, result.revision)
    if result.total_size == 0:
        source_units.append(
            ReviewUnit(
                unit_id=root_id,
                source_id=spec.source_id,
                locator=spec.locator,
                kind="empty_document",
                substantive=False,
                discovered_bytes=0,
            )
        )
        harness_skips.append(
            (root_id, "Harness byte-range scan verified that the source file is empty")
        )
        return

    source_units.append(
        ReviewUnit(
            unit_id=root_id,
            source_id=spec.source_id,
            locator=spec.locator,
            kind="source_file",
            substantive=(
                any(packet.substantive for packet in result.packets) or not result.complete
            ),
            discovered_bytes=result.total_size,
            range_start=0,
            range_end=result.total_size,
            source_revision=result.revision,
        )
    )
    packets_by_range: dict[tuple[int, int], list[StreamPacket]] = defaultdict(list)
    for packet in result.packets:
        packets_by_range[(packet.range_start, packet.range_end)].append(packet)

    for (start, end), packets in packets_by_range.items():
        fragmented = len(packets) > 1 or any(
            packet.fragment_count is not None for packet in packets
        )
        range_parent_id = _id("range", spec.source_id, spec.locator, result.revision, start, end)
        if fragmented:
            source_units.append(
                ReviewUnit(
                    unit_id=range_parent_id,
                    source_id=spec.source_id,
                    locator=spec.locator,
                    kind="source_range",
                    substantive=any(packet.substantive for packet in packets),
                    parent_unit_id=root_id,
                    discovered_bytes=end - start,
                    range_start=start,
                    range_end=end,
                    source_revision=result.revision,
                )
            )

        for packet in packets:
            leaf_id = _leaf_id(packet)
            source_units.append(
                ReviewUnit(
                    unit_id=leaf_id,
                    source_id=spec.source_id,
                    locator=spec.locator,
                    kind=packet.kind,
                    substantive=packet.substantive,
                    parent_unit_id=range_parent_id if fragmented else root_id,
                    discovered_bytes=end - start,
                    range_start=start,
                    range_end=end,
                    source_revision=result.revision,
                )
            )
            if packet.supported and not packet.substantive:
                harness_skips.append(
                    (
                        leaf_id,
                        "Harness adapter scanned this complete source range and verified "
                        "that it contains only structural metadata or control records",
                    )
                )
            elif not packet.supported:
                warnings.append(
                    f"Large source range remains unsupported: {spec.locator} "
                    f"[{start}, {end}) ({packet.reason})"
                )
            elif leaf_id not in outcome_by_leaf:
                warnings.append(
                    f"Large source range was not summarized: {spec.locator} "
                    f"[{start}, {end}) (not scheduled)"
                )
            elif not outcome_by_leaf[leaf_id].verified:
                warnings.append(
                    f"Large source range was not summarized: {spec.locator} "
                    f"[{start}, {end}) ({outcome_by_leaf[leaf_id].error})"
                )

    if not result.complete:
        assert result.pending_start is not None
        source_units.append(
            ReviewUnit(
                unit_id=_id(
                    "tail",
                    spec.source_id,
                    spec.locator,
                    result.revision,
                    result.pending_start,
                    result.total_size,
                ),
                source_id=spec.source_id,
                locator=spec.locator,
                kind="unread_range",
                substantive=True,
                parent_unit_id=root_id,
                discovered_bytes=result.total_size - result.pending_start,
                range_start=result.pending_start,
                range_end=result.total_size,
                source_revision=result.revision,
            )
        )


def _make_map_packet(source_id: str, members: list[_MapMember]) -> SummaryPacket:
    packet_ids = tuple(member.packet.packet_id for member in members)
    packet_id = _id("map", source_id, *packet_ids)
    return SummaryPacket(
        packet_id=packet_id,
        source_id=source_id,
        unit_ids=tuple(_leaf_id(member.packet) for member in members),
        locator=members[0].spec.locator,
        content="\n\n".join(member.section for member in members),
        metadata={
            "member_count": len(members),
            "members": [
                {
                    "locator": member.spec.locator,
                    "range_start": member.packet.range_start,
                    "range_end": member.packet.range_end,
                    "source_revision": member.packet.source_revision,
                    "fragment_index": member.packet.fragment_index,
                    "fragment_count": member.packet.fragment_count,
                }
                for member in members
            ],
        },
    )


def _stream_packet_limit(
    scheduler: SourceSummaryScheduler,
    *,
    reason: str,
    skill_contract: str,
    stream_limit: int,
) -> int:
    probe = SummaryPacket(
        packet_id="probe",
        source_id="probe",
        unit_ids=("probe",),
        locator="viking://probe",
        content="",
    )
    fixed = scheduler.estimate_input_chars(
        probe,
        reason=reason,
        skill_contract=skill_contract,
    )
    # Quotes, backslashes and newlines expand to two JSON characters.  Reserve
    # another 8 KiB for real locators, member metadata and source delimiters.
    # Rarer \u00xx controls may expand further and are rejected by exact mapper
    # admission rather than forcing every ordinary source into tiny packets.
    safe_chars = max(1, (scheduler.limits.per_packet_input_chars - fixed - 8192) // 2)
    return min(stream_limit, safe_chars)


def _map_section(spec: LargeSourceSpec, packet: StreamPacket) -> str:
    fragment = (
        f" fragment={packet.fragment_index + 1}/{packet.fragment_count}"
        if packet.fragment_index is not None and packet.fragment_count is not None
        else ""
    )
    return (
        f"[SOURCE locator={spec.locator} bytes=[{packet.range_start},{packet.range_end})"
        f" revision={packet.source_revision}{fragment}]\n{packet.content}"
    )


def _summary_body(outcome: SummaryOutcome) -> str:
    assert outcome.verified and outcome.summary is not None
    return (
        f"[Harness-verified source map; packet={outcome.packet_id}; "
        f"members={len(outcome.unit_ids)}.]\n\n{outcome.summary.strip()}"
    )


def _pack_summary_views(
    views: list[_SummaryView],
    *,
    max_delivery_bytes: int,
) -> tuple[list[_SummaryView], list[str]]:
    packed: list[_SummaryView] = []
    warnings: list[str] = []
    current: list[_SummaryView] = []

    def combine(items: list[_SummaryView]) -> _SummaryView:
        if len(items) == 1:
            return items[0]
        summary_id = _id("summary", *(item.summary_id for item in items))
        return _SummaryView(
            summary_id=summary_id,
            locator=_summary_locator(items[0].locator, summary_id),
            content="\n\n".join(item.content for item in items),
            source_id=items[0].source_id,
            member_ids=tuple(member_id for item in items for member_id in item.member_ids),
        )

    def emit(items: list[_SummaryView]) -> None:
        if not items:
            return
        view = combine(items)
        if _delivered_view_bytes(view) <= max_delivery_bytes:
            packed.append(view)
        else:
            warnings.append(
                f"Harness summary exceeds the read delivery budget and its source ranges "
                f"remain pending: {view.locator}"
            )

    for view in views:
        candidate = [*current, view]
        if len(candidate) <= 4 and _delivered_view_bytes(combine(candidate)) <= max_delivery_bytes:
            current = candidate
            continue
        emit(current)
        current = [view]
    emit(current)
    return packed, warnings


def _delivered_view_bytes(view: _SummaryView) -> int:
    rendered = "\n".join(
        (
            f"--- START OF {view.locator} ---",
            view.content,
            f"--- END OF {view.locator} ---",
        )
    )
    return len(rendered.encode())


def _summary_locator(locator: str, summary_id: str) -> str:
    return f"{locator}.compile-summary-{summary_id.rsplit(':', 1)[-1]}.md"


def _leaf_id(packet: StreamPacket) -> str:
    return _id("leaf", packet.source_id, packet.packet_id)


def _id(prefix: str, *parts: object) -> str:
    identity = "\0".join(str(part) for part in parts)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


__all__ = ["LargeSourceSpec", "PreparedLargeSources", "prepare_large_sources"]
