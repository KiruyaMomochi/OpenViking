import asyncio
import json
from types import SimpleNamespace

import pytest
from vikingbot.compile.coverage import CoverageLedger, ReadReceipt
from vikingbot.compile.source_pipeline import LargeSourceSpec, prepare_large_sources
from vikingbot.compile.source_scheduler import SourceSchedulerLimits, SourceSummaryScheduler
from vikingbot.compile.source_stream import SourceStreamLimits
from vikingbot.providers.base import LLMResponse

THREAD_ID = "019bc371-82cf-7d82-ad0b-96d026aacafe"
ROLLOUT_PATH = f"/tmp/.codex/sessions/2026/01/15/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"


class _Reader:
    def __init__(self, values):
        self.values = values

    async def read_range(self, locator, *, offset=0, size, expected_revision=None):
        raw = self.values[locator]
        revision = f"rev-{len(raw)}"
        assert expected_revision in {None, revision}
        content = raw[offset : offset + size]
        end = offset + len(content)
        return SimpleNamespace(
            content=content,
            start=offset,
            end=end,
            next_offset=None if end >= len(raw) else end,
            eof=end >= len(raw),
            total_size=len(raw),
            revision=revision,
        )


class _TrackingReader(_Reader):
    def __init__(self, values):
        super().__init__(values)
        self.calls = []

    async def read_range(self, locator, **kwargs):
        self.calls.append((locator, kwargs["offset"], kwargs["size"]))
        return await super().read_range(locator, **kwargs)


class _OneBrokenReader(_Reader):
    async def read_range(self, locator, **kwargs):
        if locator.endswith("broken.txt"):
            raise RuntimeError("temporary source failure")
        return await super().read_range(locator, **kwargs)


class _GeneratedReader:
    def __init__(self, total_size):
        self.total_size = total_size
        self.calls = []

    async def read_range(self, locator, *, offset=0, size, expected_revision=None):
        revision = f"rev-{self.total_size}"
        assert expected_revision in {None, revision}
        self.calls.append((offset, size))
        end = min(offset + size, self.total_size)
        return SimpleNamespace(
            content=b"x" * (end - offset),
            start=offset,
            end=end,
            next_offset=None if end >= self.total_size else end,
            eof=end >= self.total_size,
            total_size=self.total_size,
            revision=revision,
        )


class _FailsAfterPrefixReader(_Reader):
    async def read_range(self, locator, *, offset=0, **kwargs):
        if locator.endswith("broken.txt") and offset > 0:
            raise RuntimeError("transport stopped after a delivered prefix")
        return await super().read_range(locator, offset=offset, **kwargs)


class _ConflictsAfterPrefixReader(_Reader):
    async def read_range(self, locator, *, offset=0, **kwargs):
        if locator.endswith("conflict.txt") and offset > 0:
            error = RuntimeError("source revision changed")
            error.code = "CONFLICT"
            raise error
        return await super().read_range(locator, offset=offset, **kwargs)


class _Provider:
    def __init__(self):
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        envelope = json.loads(kwargs["messages"][1]["content"])
        return LLMResponse(content=f"summary {envelope['packet']['packet_id']}")


class _SlowProvider(_Provider):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.peak = 0

    async def chat(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return await super().chat(**kwargs)
        finally:
            self.active -= 1


class _OneCharProvider(_Provider):
    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(content="x")


def _limits(**overrides):
    values = {
        "range_bytes": 7,
        "max_io_bytes": 100_000,
        "max_packets": 100,
        "max_record_bytes": 10_000,
        "packet_content_chars": 20,
    }
    values.update(overrides)
    return SourceStreamLimits(**values)


def _scheduler(provider, **overrides):
    values = {
        "concurrency": 2,
        "per_packet_input_chars": 50_000,
        "max_calls": 10,
        "total_input_chars": 50_000,
        "total_output_chars": 10_000,
        "timeout_seconds": 1,
        "max_tokens": 100,
    }
    values.update(overrides)
    return SourceSummaryScheduler(
        provider,
        model="map",
        limits=SourceSchedulerLimits(**values),
    )


@pytest.mark.asyncio
async def test_pipeline_groups_many_files_for_one_source_and_coverage_completes():
    values = {
        "viking://resources/source/a.txt": b"alpha beta gamma",
        "viking://resources/source/b.txt": b"delta epsilon zeta",
    }
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader(values),
        [LargeSourceSpec("src_1", locator, len(raw)) for locator, raw in values.items()],
        scheduler=_scheduler(provider),
        stream_limits=_limits(packet_content_chars=8),
        reason="Make a report",
        skill_contract="Preserve facts",
    )

    assert len(prepared.snapshots) == 1
    assert prepared.snapshots[0].source_id == "src_1"
    assert len(provider.calls) < sum(unit.kind == "text" for unit in prepared.snapshots[0].units)
    assert prepared.review_views
    ledger = CoverageLedger(prepared.snapshots)
    for unit_id, reason in prepared.harness_skips:
        ledger.mark_harness_non_substantive(unit_id, reason)
    for summary in prepared.snapshots[0].verified_summaries:
        unit = ledger.by_id[summary.summary_unit_id]
        ledger.cover_from_summary(
            summary.summary_unit_id,
            ReadReceipt.for_content(
                prepared.snapshots[0],
                summary.summary_unit_id,
                prepared.review_views[unit.locator].content,
            ),
        )
    assert ledger.is_complete()


@pytest.mark.asyncio
async def test_pipeline_submits_complete_map_packets_concurrently():
    locator = "viking://resources/source/concurrent.txt"
    raw = b"x" * 50_000
    provider = _SlowProvider()
    scheduler = _scheduler(
        provider,
        concurrency=3,
        per_packet_input_chars=20_000,
        max_calls=20,
        total_input_chars=400_000,
    )

    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, len(raw))],
        scheduler=scheduler,
        stream_limits=_limits(
            range_bytes=len(raw),
            max_io_bytes=len(raw),
            max_packets=1_000,
            packet_content_chars=len(raw),
        ),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert prepared.map_packets >= scheduler.limits.concurrency
    assert provider.peak == scheduler.limits.concurrency


@pytest.mark.asyncio
async def test_pipeline_batches_independent_physical_files_concurrently():
    values = {
        f"viking://resources/source/{index}.txt": f"file {index} facts".encode()
        for index in range(4)
    }
    provider = _SlowProvider()
    scheduler = _scheduler(provider, concurrency=4)

    prepared = await prepare_large_sources(
        _Reader(values),
        [LargeSourceSpec("src_1", locator, len(raw)) for locator, raw in values.items()],
        scheduler=scheduler,
        stream_limits=_limits(range_bytes=100, packet_content_chars=100),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert len(provider.calls) == 4
    assert provider.peak == 4
    assert len(prepared.review_views) > 0


@pytest.mark.parametrize(
    ("max_calls", "total_input_chars", "total_output_chars"),
    ((1, 50_000, 10_000), (10, 50_000, 1), (10, 1_450, 10_000)),
)
@pytest.mark.asyncio
async def test_exhausted_map_stage_does_not_read_the_next_source(
    max_calls,
    total_input_chars,
    total_output_chars,
):
    first = "viking://resources/source/first.txt"
    second = "viking://resources/source/second.txt"
    values = {first: b"first source", second: b"second source"}
    reader = _TrackingReader(values)
    provider = _OneCharProvider()

    prepared = await prepare_large_sources(
        reader,
        [
            LargeSourceSpec("src_1", first, len(values[first])),
            LargeSourceSpec("src_1", second, len(values[second])),
        ],
        scheduler=_scheduler(
            provider,
            max_calls=max_calls,
            total_input_chars=total_input_chars,
            total_output_chars=total_output_chars,
        ),
        stream_limits=_limits(range_bytes=100),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert [locator for locator, _offset, _size in reader.calls] == [first]
    assert len(provider.calls) == 1
    assert any(
        unit.kind == "unread_source" and unit.locator == second
        for unit in prepared.snapshots[0].units
    )


def _meta():
    return {
        "type": "session_meta",
        "payload": {
            "id": THREAD_ID,
            "timestamp": "2026-01-15T15:55:54.000Z",
            "cwd": "/workspace",
            "originator": "codex_cli_rs",
            "cli_version": "0.105.0",
            "source": "cli",
        },
    }


@pytest.mark.asyncio
async def test_fragment_children_share_range_but_all_are_summary_covered():
    raw = (
        "\n".join(
            json.dumps(record)
            for record in (
                _meta(),
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "x" * 100}],
                    },
                },
            )
        )
        + "\n"
    ).encode()
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader({ROLLOUT_PATH: raw}),
        [LargeSourceSpec("src_1", ROLLOUT_PATH, len(raw))],
        scheduler=_scheduler(provider),
        stream_limits=_limits(packet_content_chars=20),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    fragment_units = [unit for unit in snapshot.units if unit.kind == "codex_message_fragment"]
    assert len(fragment_units) > 1
    assert len({(unit.range_start, unit.range_end) for unit in fragment_units}) == 1
    ledger = CoverageLedger((snapshot,))
    for unit_id, reason in prepared.harness_skips:
        ledger.mark_harness_non_substantive(unit_id, reason)
    for summary in snapshot.verified_summaries:
        summary_unit = ledger.by_id[summary.summary_unit_id]
        ledger.cover_from_summary(
            summary.summary_unit_id,
            ReadReceipt.for_content(
                snapshot,
                summary.summary_unit_id,
                prepared.review_views[summary_unit.locator].content,
            ),
        )
    assert ledger.is_complete()


@pytest.mark.asyncio
async def test_structural_only_source_needs_no_model_call_and_is_harness_skipped():
    raw = (
        "\n".join(
            json.dumps(row)
            for row in (
                _meta(),
                {"type": "event_msg", "payload": {"type": "task_complete"}},
            )
        )
        + "\n"
    ).encode()
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader({ROLLOUT_PATH: raw}),
        [LargeSourceSpec("src_1", ROLLOUT_PATH, len(raw))],
        scheduler=_scheduler(provider),
        stream_limits=_limits(),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert provider.calls == []
    ledger = CoverageLedger(prepared.snapshots)
    for unit_id, reason in prepared.harness_skips:
        ledger.mark_harness_non_substantive(unit_id, reason)
    assert ledger.is_complete()


@pytest.mark.asyncio
async def test_one_stream_failure_stays_pending_without_losing_successful_peer():
    values = {
        "viking://resources/source/broken.txt": b"broken",
        "viking://resources/source/good.txt": b"good source material",
    }
    provider = _Provider()
    prepared = await prepare_large_sources(
        _OneBrokenReader(values),
        [LargeSourceSpec("src_1", locator, len(raw)) for locator, raw in values.items()],
        scheduler=_scheduler(provider),
        stream_limits=_limits(),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    assert any(unit.kind == "unread_source" for unit in snapshot.units)
    assert prepared.review_views
    assert any("broken.txt" in warning for warning in prepared.warnings)
    ledger = CoverageLedger((snapshot,))
    for summary in snapshot.verified_summaries:
        unit = ledger.by_id[summary.summary_unit_id]
        ledger.cover_from_summary(
            summary.summary_unit_id,
            ReadReceipt.for_content(
                snapshot,
                summary.summary_unit_id,
                prepared.review_views[unit.locator].content,
            ),
        )
    assert "unread_source" not in (ledger.gate_error() or "")
    assert "unread:" in (ledger.gate_error() or "")


@pytest.mark.asyncio
async def test_revision_conflict_does_not_invalidate_completed_peer_file_summary():
    values = {
        "viking://resources/source/good.txt": b"good source material",
        "viking://resources/source/conflict.txt": b"conflicting source material",
    }
    prepared = await prepare_large_sources(
        _ConflictsAfterPrefixReader(values),
        [LargeSourceSpec("src_1", locator, len(raw)) for locator, raw in values.items()],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(range_bytes=7),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    assert any(unit.kind == "unread_source" for unit in snapshot.units)
    good_leaf_ids = {
        unit.unit_id
        for unit in snapshot.units
        if unit.locator.endswith("good.txt") and unit.kind == "text"
    }
    assert good_leaf_ids
    assert any(
        good_leaf_ids <= set(summary.covered_unit_ids)
        for summary in snapshot.verified_summaries
    )


@pytest.mark.asyncio
async def test_summary_views_never_combine_members_from_different_sources():
    values = {
        "viking://resources/one/doc.txt": b"source one material",
        "viking://resources/two/doc.txt": b"source two material",
    }
    prepared = await prepare_large_sources(
        _Reader(values),
        [
            LargeSourceSpec("src_1", "viking://resources/one/doc.txt", 19),
            LargeSourceSpec("src_2", "viking://resources/two/doc.txt", 19),
        ],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(packet_content_chars=100),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert len(prepared.snapshots) == 2
    assert len(prepared.review_views) == 2
    for snapshot in prepared.snapshots:
        unit_ids = {unit.unit_id for unit in snapshot.units}
        assert all(
            set(summary.covered_unit_ids) <= unit_ids for summary in snapshot.verified_summaries
        )


@pytest.mark.asyncio
async def test_failed_read_without_returned_bytes_does_not_spend_later_io_budget():
    values = {
        "viking://resources/source/broken.txt": b"123456",
        "viking://resources/source/good.txt": b"abcdef",
    }
    prepared = await prepare_large_sources(
        _OneBrokenReader(values),
        [
            LargeSourceSpec("src_1", "viking://resources/source/broken.txt", 6),
            LargeSourceSpec("src_1", "viking://resources/source/good.txt", 6),
        ],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(max_io_bytes=6),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert prepared.review_views
    assert sum(unit.kind == "unread_source" for unit in prepared.snapshots[0].units) == 1
    assert prepared.io_bytes == 6


@pytest.mark.asyncio
async def test_long_skill_contract_shrinks_stream_packets_before_map_budget():
    locator = "viking://resources/source/large.txt"
    raw = b"x" * 240_000
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, len(raw))],
        scheduler=_scheduler(
            provider,
            per_packet_input_chars=80_000,
            max_calls=20,
            total_input_chars=2_000_000,
        ),
        stream_limits=_limits(
            range_bytes=100_000,
            max_io_bytes=300_000,
            packet_content_chars=240_000,
        ),
        reason="r" * 100_000,
        skill_contract="s" * 100_000,
    )

    assert provider.calls
    assert prepared.review_views
    assert not any("per-packet input" in warning for warning in prepared.warnings)


@pytest.mark.asyncio
async def test_virtual_150mb_dense_source_stops_reading_and_keeps_covered_prefix():
    total_size = 150 * 1024 * 1024
    reader = _GeneratedReader(total_size)
    provider = _Provider()
    prepared = await prepare_large_sources(
        reader,
        [LargeSourceSpec("src_1", "viking://resources/source/huge.txt", total_size)],
        scheduler=_scheduler(
            provider,
            per_packet_input_chars=50_000,
            max_calls=1,
            total_input_chars=50_000,
        ),
        stream_limits=_limits(
            range_bytes=64 * 1024,
            max_io_bytes=200 * 1024 * 1024,
            max_packets=10_000,
            packet_content_chars=240_000,
        ),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    tail = next(unit for unit in snapshot.units if unit.kind == "unread_range")
    assert tail.range_start and tail.range_start < total_size
    assert prepared.io_bytes == 64 * 1024
    assert reader.calls == [(0, 64 * 1024)]
    assert len(provider.calls) == 1
    assert prepared.review_views

    ledger = CoverageLedger((snapshot,))
    for summary in snapshot.verified_summaries:
        unit = ledger.by_id[summary.summary_unit_id]
        view = prepared.review_views[unit.locator]
        assert isinstance(view.content, str)
        ledger.cover_from_summary(
            summary.summary_unit_id,
            ReadReceipt.for_content(snapshot, summary.summary_unit_id, view.content),
        )
    assert ledger.status(tail.unit_id).value == "pending"
    assert not ledger.is_complete()
    assert any(
        ledger.status(unit.unit_id).value == "covered"
        for unit in snapshot.units
        if unit.kind == "text"
    )


@pytest.mark.asyncio
async def test_bytes_returned_before_failure_still_consume_shared_io_budget():
    values = {
        "viking://resources/source/broken.txt": b"123456",
        "viking://resources/source/good.txt": b"abcdef",
    }
    prepared = await prepare_large_sources(
        _FailsAfterPrefixReader(values),
        [
            LargeSourceSpec("src_1", "viking://resources/source/broken.txt", 6),
            LargeSourceSpec("src_1", "viking://resources/source/good.txt", 6),
        ],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(range_bytes=3, max_io_bytes=8),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert prepared.io_bytes == 3
    assert sum(unit.kind == "unread_range" for unit in prepared.snapshots[0].units) == 1
    assert sum(unit.kind == "unread_source" for unit in prepared.snapshots[0].units) == 1
    assert any("remaining I/O budget" in warning for warning in prepared.warnings)


@pytest.mark.asyncio
async def test_json_escape_heavy_source_is_split_before_exact_prompt_limit():
    locator = "viking://resources/source/backslashes.txt"
    raw = b"\\" * 240_000
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, len(raw))],
        scheduler=_scheduler(
            provider,
            per_packet_input_chars=320_000,
            max_calls=64,
            total_input_chars=12_000_000,
        ),
        stream_limits=_limits(
            range_bytes=100_000,
            max_io_bytes=300_000,
            packet_content_chars=240_000,
        ),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert provider.calls
    assert not any("per-packet input" in warning for warning in prepared.warnings)
    assert not any(unit.kind == "unread_range" for unit in prepared.snapshots[0].units)


@pytest.mark.asyncio
async def test_rare_six_character_json_escapes_still_fail_closed_at_exact_limit():
    locator = "viking://resources/source/controls.txt"
    raw = b"\x00" * 20_000
    provider = _Provider()
    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, len(raw))],
        scheduler=_scheduler(
            provider,
            per_packet_input_chars=50_000,
            max_calls=10,
            total_input_chars=500_000,
        ),
        stream_limits=_limits(
            range_bytes=len(raw),
            max_io_bytes=len(raw),
            packet_content_chars=len(raw),
        ),
        reason="Summarize",
        skill_contract="Markdown",
    )

    assert provider.calls == []
    tail = next(unit for unit in prepared.snapshots[0].units if unit.kind == "unread_range")
    assert tail.range_start == 0


@pytest.mark.asyncio
async def test_unknown_inventory_size_uses_returned_size_without_type_error():
    locator = "viking://resources/source/unknown.txt"
    raw = b"unknown-size source material"
    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, None)],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(max_io_bytes=100),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    root = next(unit for unit in snapshot.units if unit.kind == "source_file")
    assert root.discovered_bytes == len(raw)
    assert prepared.io_bytes == len(raw)
    assert prepared.review_views


@pytest.mark.asyncio
async def test_unknown_oversized_source_keeps_exact_unread_tail_within_io_budget():
    locator = "viking://resources/source/unknown-huge.txt"
    raw = b"x" * 20
    prepared = await prepare_large_sources(
        _Reader({locator: raw}),
        [LargeSourceSpec("src_1", locator, None)],
        scheduler=_scheduler(_Provider()),
        stream_limits=_limits(range_bytes=5, max_io_bytes=5),
        reason="Summarize",
        skill_contract="Markdown",
    )

    snapshot = prepared.snapshots[0]
    tail = next(unit for unit in snapshot.units if unit.kind == "unread_range")
    assert (tail.range_start, tail.range_end) == (0, len(raw))
    assert prepared.io_bytes == 5
    assert not prepared.review_views
