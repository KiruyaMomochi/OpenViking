"""Local recall regressions, loaded through Hermes external discovery."""

import time
import pytest

from .test_gateway_recall import initialize, profile_scope


def recall(provider, content):
    class Client:
        def get(self, *args, **kwargs):
            return {"result": {"content": content}}

    return provider._build_prefetch_entries(
        Client(), [{"uri": "viking://user/default/memories/events/example.md",
                    "level": 2, "category": "events", "abstract": "ignored"}],
        prefer_abstract=False, max_injected_chars=10000,
        deadline=time.monotonic() + 10, request_timeout=1, full_read_limit=1,
    )


def test_prefetch_projects_recall_without_transcript_or_metadata(external_provider):
    _, provider, _, _ = external_provider("projection")
    content = (
        "# Summary\nFirst summary paragraph.\n\n\nSecond summary paragraph.\n"
        "# 2026-09-10 (Thursday) ChatLog:\n"
        "**33**: First paragraph.\n\n\nSecond paragraph.\n\n"
        "# Evidence\n"
        "```python\nfirst = 1\n\n\nsecond = 2\n**speaker**: preserved inside code\n```\n\n\n"
        "**orphan**: transcript row\n"
        "After the fence.\n\n<!-- MEMORY_FIELDS\n{}\n-->"
    )
    original = content
    recalled = recall(provider, content)[0]
    assert "First summary paragraph.\n  \n  \n  Second summary paragraph." in recalled

    assert "ChatLog" not in recalled
    assert "First paragraph." not in recalled
    assert "Second paragraph." not in recalled
    assert "transcript row" not in recalled
    assert "MEMORY_FIELDS" not in recalled
    assert "first = 1\n  \n  \n  second = 2" in recalled
    assert "**speaker**: preserved inside code" in recalled
    assert "After the fence." in recalled
    assert content == original  # Projection must leave the stored input intact.


@pytest.mark.parametrize("heading", ["# ChatLog:", "# 2026-09-10 (Thursday) ChatLog:"])
def test_undated_and_dated_chatlogs(external_provider, heading):
    _, provider, _, _ = external_provider("chatlog")
    assert "drop this" not in recall(provider, f"# Summary\nKeep this.\n{heading}\n**user**: drop this\n")[0]
    assert recall(provider, f"{heading}\n**user**: drop this\n") == []


def test_server_context_projection_and_cap(external_provider, monkeypatch):
    home, provider, module, manager, _ = initialize(external_provider, monkeypatch, compress="server", scope="shared")
    content = "# Summary\nKeep this.\nChatLog:\n**user**: drop this\n<!-- MEMORY_FIELDS\n{}\n-->"
    monkeypatch.setattr(module._VikingClient, "post", lambda *args, **kwargs: {"result": {"digest": content}})
    try:
        with profile_scope(home):
            result = provider._search_prefetch_context("Recall preferences", session_id="shared-group", client=provider._client)
        assert result.startswith('<retrieved-memory provider="openviking" category="context">')
        assert "Keep this." in result
        assert "drop this" not in result
        assert "ChatLog" not in result
        assert "MEMORY_FIELDS" not in result
        content = "x" * 5000
        with profile_scope(home):
            assert provider._search_prefetch_context("Recall preferences", client=provider._client) == ""
    finally:
        manager.shutdown_all()
