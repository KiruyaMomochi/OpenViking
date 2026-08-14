import json
from types import SimpleNamespace

import pytest
from vikingbot.compile.adapters import AdapterError
from vikingbot.compile.source_stream import (
    SourceReadMeter,
    SourceStreamBudgetError,
    SourceStreamLimits,
    stream_source,
)

THREAD_ID = "019bc371-82cf-7d82-ad0b-96d026aacafe"
ROLLOUT_PATH = f"/tmp/.codex/sessions/2026/01/15/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"


class FakeRangeReader:
    def __init__(self, content: bytes, *, revision: str = "rev-1", as_mapping: bool = False):
        self.content = content
        self.revision = revision
        self.as_mapping = as_mapping
        self.calls = []

    async def read_range(
        self,
        locator,
        *,
        offset=0,
        size=16 * 1024 * 1024,
        expected_revision=None,
    ):
        self.calls.append(
            {
                "locator": locator,
                "offset": offset,
                "size": size,
                "expected_revision": expected_revision,
            }
        )
        chunk = self.content[offset : offset + size]
        end = offset + len(chunk)
        result = {
            "content": chunk,
            "start": offset,
            "end": end,
            "next_offset": None if end >= len(self.content) else end,
            "eof": end >= len(self.content),
            "total_size": len(self.content),
            "revision": self.revision,
        }
        return result if self.as_mapping else SimpleNamespace(**result)


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


def _message(role: str, text: str):
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _jsonl(*records) -> bytes:
    return ("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n").encode()


def _limits(**overrides) -> SourceStreamLimits:
    values = {
        "range_bytes": 8,
        "max_io_bytes": 1024 * 1024,
        "max_packets": 1000,
        "max_record_bytes": 1024 * 1024,
        "packet_content_chars": 1000,
    }
    values.update(overrides)
    return SourceStreamLimits(**values)


def _assert_gapless(result):
    ranges = sorted({(packet.range_start, packet.range_end) for packet in result.packets})
    cursor = 0
    for start, end in ranges:
        assert start <= cursor
        cursor = max(cursor, end)
    assert cursor == result.total_size


@pytest.mark.asyncio
@pytest.mark.parametrize("as_mapping", [False, True])
async def test_stream_source_uses_bounded_ranges_and_revision_chain(as_mapping):
    reader = FakeRangeReader(b"abcdefghijklmnopqrstuvwxyz", as_mapping=as_mapping)

    result = await stream_source(
        reader,
        source_id="source-1",
        locator="viking://resources/plain.txt",
        limits=_limits(range_bytes=7),
    )

    assert max(call["size"] for call in reader.calls) == 7
    assert [call["offset"] for call in reader.calls] == [0, 7, 14, 21]
    assert [call["expected_revision"] for call in reader.calls] == [
        None,
        "rev-1",
        "rev-1",
        "rev-1",
    ]
    assert result.io_bytes == result.total_size == 26
    assert result.revision == "rev-1"
    assert "".join(packet.content for packet in result.packets) == "abcdefghijklmnopqrstuvwxyz"
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_codex_packetization_is_independent_of_read_boundaries():
    raw = _jsonl(
        _meta(),
        _message("user", "请总结 café ☕"),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        _message("assistant", "完成。"),
    )
    result = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=5),
    )

    rendered = "\n".join(packet.content for packet in result.packets)
    assert "请总结 café ☕" in rendered
    assert "完成。" in rendered
    assert all(packet.supported for packet in result.packets)
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_codex_unknown_record_is_isolated_and_later_records_survive():
    before = _jsonl(_meta(), _message("user", "before"))
    unknown = b'{"type":"future_record","payload":{"opaque":true}}\n'
    after = _jsonl(_message("assistant", "after"))
    raw = before + unknown + after

    result = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=11),
    )

    unsupported = [packet for packet in result.packets if not packet.supported]
    assert len(unsupported) == 1
    assert unsupported[0].range_start == len(before)
    assert unsupported[0].range_end == len(before) + len(unknown)
    assert unsupported[0].kind == "codex_unsupported"
    rendered = "\n".join(packet.content for packet in result.packets if packet.supported)
    assert "before" in rendered
    assert "after" in rendered
    assert result.warnings == (unsupported[0].reason,)
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_codex_structural_only_source_produces_non_substantive_coverage_packet():
    raw = _jsonl(
        _meta(),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    )

    result = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(),
    )

    assert len(result.packets) == 1
    packet = result.packets[0]
    assert (packet.range_start, packet.range_end) == (0, len(raw))
    assert packet.kind == "codex_structural"
    assert packet.supported is True
    assert packet.substantive is False
    assert packet.content == ""


@pytest.mark.asyncio
async def test_oversized_codex_message_is_fragmented_with_stable_metadata():
    raw = _jsonl(_meta(), _message("user", "abcdefghij" * 9))
    first = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=13, packet_content_chars=23),
    )
    second = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=17, packet_content_chars=23),
    )

    fragments = [packet for packet in first.packets if packet.fragment_count is not None]
    assert len(fragments) > 1
    assert [packet.fragment_index for packet in fragments] == list(range(len(fragments)))
    assert {packet.fragment_count for packet in fragments} == {len(fragments)}
    assert len({(packet.range_start, packet.range_end) for packet in fragments}) == 1
    assert all(len(packet.content) <= 23 for packet in fragments)
    assert [packet.packet_id for packet in first.packets] == [
        packet.packet_id for packet in second.packets
    ]
    assert "".join(packet.content for packet in fragments).startswith("[USER]\n")
    _assert_gapless(first)


@pytest.mark.asyncio
async def test_whitespace_only_codex_fragments_are_non_substantive():
    raw = _jsonl(_meta(), _message("user", "hello" + " " * 120 + "bye"))
    result = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=17, packet_content_chars=20),
    )

    fragments = [packet for packet in result.packets if packet.fragment_count is not None]
    structural = [packet for packet in fragments if not packet.substantive]
    assert structural
    assert all(packet.kind == "codex_structural_fragment" for packet in structural)
    assert all(not packet.content.strip() for packet in structural)
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_codex_packet_cap_counts_all_message_separators_exactly():
    raw = _jsonl(_meta(), *(_message("user", f"m{index}") for index in range(80)))
    cap = 200
    result = await stream_source(
        FakeRangeReader(raw),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(range_bytes=73, packet_content_chars=cap),
    )

    assert result.complete
    assert all(len(packet.content) <= cap for packet in result.packets)
    assert any(packet.kind == "codex_messages" for packet in result.packets)
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_generic_utf8_boundaries_preserve_exact_byte_ranges():
    raw = "Aé中☕Z".encode()
    result = await stream_source(
        FakeRangeReader(raw),
        source_id="plain",
        locator="viking://resources/plain.txt",
        limits=_limits(range_bytes=2, packet_content_chars=2),
    )

    assert "".join(packet.content for packet in result.packets) == raw.decode()
    for packet in result.packets:
        assert raw[packet.range_start : packet.range_end].decode() == packet.content
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_invalid_utf8_is_isolated_and_streaming_continues():
    raw = b"before\xffafter"
    result = await stream_source(
        FakeRangeReader(raw),
        source_id="plain",
        locator="viking://resources/plain.txt",
        limits=_limits(range_bytes=3),
    )

    unsupported = [packet for packet in result.packets if not packet.supported]
    assert len(unsupported) == 1
    assert raw[unsupported[0].range_start : unsupported[0].range_end] == b"\xff"
    assert "".join(packet.content for packet in result.packets if packet.supported) == (
        "beforeafter"
    )
    _assert_gapless(result)


@pytest.mark.asyncio
async def test_source_io_budget_fails_without_requesting_past_the_budget():
    reader = FakeRangeReader(b"123456")

    with pytest.raises(SourceStreamBudgetError, match="source size 6"):
        await stream_source(
            reader,
            source_id="plain",
            locator="viking://resources/plain.txt",
            limits=_limits(range_bytes=4, max_io_bytes=5),
        )

    assert reader.calls == [
        {
            "locator": "viking://resources/plain.txt",
            "offset": 0,
            "size": 4,
            "expected_revision": None,
        }
    ]


@pytest.mark.asyncio
async def test_packet_and_record_budgets_are_hard_limits():
    with pytest.raises(SourceStreamBudgetError, match="packet budget"):
        await stream_source(
            FakeRangeReader(b"abc"),
            source_id="plain",
            locator="viking://resources/plain.txt",
            limits=_limits(packet_content_chars=1, max_packets=2),
        )

    raw = _jsonl(_meta(), _message("user", "too large"))
    with pytest.raises(AdapterError, match="exceeds 40 bytes"):
        await stream_source(
            FakeRangeReader(raw),
            source_id="session-source",
            locator=ROLLOUT_PATH,
            limits=_limits(max_record_bytes=40),
        )


@pytest.mark.asyncio
async def test_empty_codex_rollout_is_verified_empty_without_decoder_failure():
    result = await stream_source(
        FakeRangeReader(b""),
        source_id="session-source",
        locator=ROLLOUT_PATH,
        limits=_limits(),
    )

    assert result.complete
    assert result.total_size == 0
    assert result.packets == ()


@pytest.mark.asyncio
async def test_incremental_consumer_releases_content_and_keeps_exact_pending_tail():
    seen = []

    async def consume(group):
        seen.append("".join(packet.content for packet in group))
        return len(seen) == 1

    meter = SourceReadMeter(100)
    result = await stream_source(
        FakeRangeReader(b"abcdef"),
        source_id="plain",
        locator="viking://resources/plain.txt",
        limits=_limits(range_bytes=6, packet_content_chars=2),
        on_packet_group=consume,
        meter=meter,
    )

    assert seen == ["ab", "cd"]
    assert not result.complete
    assert result.pending_start == 2
    assert [(packet.range_start, packet.range_end) for packet in result.packets] == [(0, 2)]
    assert result.packets[0].content == ""
    assert result.io_bytes == meter.bytes_read == 6
    assert result.read_calls == meter.calls == 1


@pytest.mark.asyncio
async def test_incremental_packet_cap_preserves_prefix_instead_of_rolling_back_source():
    async def consume(_group):
        return True

    result = await stream_source(
        FakeRangeReader(b"abcdef"),
        source_id="plain",
        locator="viking://resources/plain.txt",
        limits=_limits(range_bytes=6, packet_content_chars=1, max_packets=2),
        on_packet_group=consume,
    )

    assert not result.complete
    assert result.pending_start == 2
    assert [(packet.range_start, packet.range_end) for packet in result.packets] == [
        (0, 1),
        (1, 2),
    ]
    assert "packet budget" in result.stop_reason
