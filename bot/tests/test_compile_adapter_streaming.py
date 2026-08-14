import json

import pytest
from vikingbot.compile.adapters import AdapterError, CodexRolloutAdapter

THREAD_ID = "019bc371-82cf-7d82-ad0b-96d026aacafe"
ROLLOUT_PATH = f"/tmp/.codex/sessions/2026/01/15/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"


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


def _jsonl(*records) -> bytes:
    return ("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n").encode()


def _message(role: str, text: str):
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def test_stream_adapter_is_invariant_to_byte_chunk_boundaries():
    raw = _jsonl(
        _meta(),
        _message("user", "请总结 café ☕"),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        _message("assistant", "完成。"),
    )
    adapter = CodexRolloutAdapter()
    expected = adapter.adapt_stream(path=ROLLOUT_PATH, chunks=[raw])

    for split in range(1, len(raw)):
        actual = adapter.adapt_stream(
            path=ROLLOUT_PATH,
            chunks=[raw[:split], raw[split:]],
        )
        assert actual.render() == expected.render()
        assert actual.dispositions == expected.dispositions
        assert [(item.start, item.end) for item in actual.records] == [
            (item.start, item.end) for item in expected.records
        ]

    assert "请总结 café ☕" in expected.render()
    assert "完成。" in expected.render()
    assert expected.dispositions == {
        "emitted": 2,
        "structural": 2,
        "duplicate": 0,
        "unsupported": 0,
    }


def test_stream_adapter_keeps_valid_records_around_unknown_and_malformed_records():
    raw = b"".join(
        [
            _jsonl(_meta(), _message("user", "before")),
            b'{"type":"future_record","payload":{"value":"opaque"}}\n',
            b"{malformed\n",
            _jsonl(_message("assistant", "after")),
        ]
    )

    adapted = CodexRolloutAdapter().adapt_stream(
        path=ROLLOUT_PATH,
        chunks=[raw[:17], raw[17:63], raw[63:]],
    )

    assert "before" in adapted.render()
    assert "after" in adapted.render()
    assert adapted.dispositions["unsupported"] == 2
    assert len(adapted.warnings) == 2
    assert all(record.opaque_excerpt for record in adapted.records if record.warning)
    complete_view = adapted.render_with_unsupported()
    assert complete_view.index("before") < complete_view.index("future_record")
    assert complete_view.index("future_record") < complete_view.index("after")
    assert "UNSUPPORTED SOURCE RECORD" in complete_view


def test_stream_adapter_filters_current_control_and_tool_payloads():
    raw = _jsonl(
        _meta(),
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "output": "SECRET TOOL OUTPUT",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "tool_search_output",
                "tools": ["SECRET TOOL SCHEMA"],
            },
        },
        {
            "type": "compacted",
            "payload": {
                "replacement_history": [
                    {
                        "type": "compaction",
                        "encrypted_content": "SECRET ENCRYPTED COMPACTION",
                    }
                ]
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "thread_settings_applied", "settings": "SECRET"},
        },
        {
            "type": "event_msg",
            "payload": {"type": "mcp_tool_call_end", "result": "SECRET MCP RESULT"},
        },
        {
            "type": "event_msg",
            "payload": {"type": "thread_rolled_back", "num_turns": 1},
        },
        {
            "type": "event_msg",
            "payload": {"type": "web_search_end", "result": "SECRET"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:SECRET"},
                    {"type": "input_text", "text": "visible"},
                ],
            },
        },
    )

    adapted = CodexRolloutAdapter().adapt_stream(path=ROLLOUT_PATH, chunks=[raw])

    assert adapted.render().endswith("[USER]\nvisible\n")
    assert "SECRET" not in adapted.render()
    assert adapted.dispositions["unsupported"] == 0


def test_stream_adapter_bounds_unfinished_single_record():
    decoder = CodexRolloutAdapter().stream_decoder(ROLLOUT_PATH, max_record_bytes=64)

    with pytest.raises(AdapterError, match="exceeds 64 bytes"):
        decoder.feed(b"x" * 65)


def test_small_stream_can_retain_a_complete_unknown_record_without_raw_fallback():
    opaque = "x" * 5000 + "TAIL"
    raw = _jsonl(_meta(), {"type": "future_record", "payload": {"opaque": opaque}})

    adapted = CodexRolloutAdapter().adapt_stream(
        path=ROLLOUT_PATH,
        chunks=[raw],
        opaque_excerpt_chars=len(raw),
    )

    assert "TAIL" in adapted.render_with_unsupported()
    assert adapted.dispositions["unsupported"] == 1


def test_strict_canonicalize_accepts_known_new_control_events():
    raw = _jsonl(
        _meta(),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        {"type": "event_msg", "payload": {"type": "patch_apply_end"}},
        _message("assistant", "kept"),
    ).decode()

    assert "kept" in CodexRolloutAdapter().canonicalize(raw, path=ROLLOUT_PATH)
