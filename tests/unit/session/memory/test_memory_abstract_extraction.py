"""Regression tests for memory abstract extraction used by vector metadata."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service import reindex_executor as reindex_module
from openviking.service.reindex_executor import ReindexExecutor, _ReindexCounters
from openviking.session.memory.memory_updater import MemoryUpdater, MemoryUpdateResult
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.asyncio
async def test_vectorize_memory_stores_summary_abstract_but_embeds_full_content():
    uri = "viking://user/test/memories/events/day.md"
    content = (
        "Summary only.\n"
        "2026-06-13 (Saturday) ChatLog:\n"
        "[user]: transcript should be embedded but not stored as abstract\n"
    )
    viking_fs = MagicMock()
    viking_fs.read_file = AsyncMock(return_value=content)
    updater = MemoryUpdater()
    updater._get_viking_fs = MagicMock(return_value=viking_fs)
    updater._vikingdb = MagicMock()
    updater._vikingdb.enqueue_embedding_msg = AsyncMock(return_value=True)
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.USER)

    result = MemoryUpdateResult()
    result.add_written(uri)
    await updater._vectorize_memories(
        result,
        ctx=ctx,
    )

    embedding_msg = updater._vikingdb.enqueue_embedding_msg.call_args.args[0]
    assert embedding_msg.context_data["abstract"] == "Summary only."
    assert "ChatLog" not in embedding_msg.context_data["abstract"]
    assert "[user]" not in embedding_msg.context_data["abstract"]
    assert "ChatLog" in embedding_msg.message


@pytest.mark.asyncio
async def test_reindex_memory_regenerates_summary_abstract_from_body(monkeypatch):
    uri = "viking://user/test/memories/events/day.md"
    body = (
        "Summary from body.\n"
        "2026-06-13 (Saturday) ChatLog:\n"
        "[assistant]: stale transcript should not be reindexed as abstract\n"
    )
    viking_fs = MagicMock()
    viking_fs.exists = AsyncMock(return_value=True)
    viking_fs.stat = AsyncMock(return_value={"isDir": False})
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: viking_fs)

    executor = ReindexExecutor()
    executor._safe_read_text = AsyncMock(return_value=body)
    executor._best_file_summary = AsyncMock(return_value="")
    executor._fetch_existing_record = AsyncMock(
        return_value={"abstract": body, "uri": uri}
    )
    executor._upsert_context = AsyncMock()
    executor._chunk_memory_body = MagicMock(return_value=[])
    counters = _ReindexCounters()
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ROOT)

    await executor._reindex_memory_vectors(uri=uri, counters=counters, ctx=ctx)

    kwargs = executor._upsert_context.call_args.kwargs
    assert kwargs["abstract"] == "Summary from body."
    assert "ChatLog" not in kwargs["abstract"]
    assert kwargs["vector_text"] == body


@pytest.mark.asyncio
async def test_reindex_memory_prefers_overview_summary_over_stale_existing_abstract(monkeypatch):
    uri = "viking://user/test/memories/events/day.md"
    body = (
        "Body summary.\n"
        "2026-06-13 (Saturday) ChatLog:\n"
        "[assistant]: transcript\n"
    )
    viking_fs = MagicMock()
    viking_fs.exists = AsyncMock(return_value=True)
    viking_fs.stat = AsyncMock(return_value={"isDir": False})
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: viking_fs)

    executor = ReindexExecutor()
    executor._safe_read_text = AsyncMock(return_value=body)
    executor._best_file_summary = AsyncMock(return_value="Overview summary.")
    executor._fetch_existing_record = AsyncMock(return_value={"abstract": body})
    executor._upsert_context = AsyncMock()
    executor._chunk_memory_body = MagicMock(return_value=[])
    counters = _ReindexCounters()
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ROOT)

    await executor._reindex_memory_vectors(uri=uri, counters=counters, ctx=ctx)

    kwargs = executor._upsert_context.call_args.kwargs
    assert kwargs["abstract"] == "Overview summary."
    assert kwargs["vector_text"] == body
