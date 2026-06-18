# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for reindexing single-file URIs under tree locks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from openviking.server.identity import RequestContext, Role
from openviking.service import reindex_executor as reindex_module
from openviking.service.reindex_executor import ReindexExecutor
from openviking.session.memory.utils.memory_file_utils import _extract_memory_abstract
from openviking.storage.transaction import init_lock_manager, release_all_locks, reset_lock_manager
from openviking_cli.session.user_id import UserIdentifier


def test_reindex_memory_abstract_uses_summary_before_chatlog():
    body = (
        "Concise memory summary for search injection.\n"
        "2026-06-13 (Saturday) ChatLog:\n"
        "[assistant]: verbose transcript should not be metadata\n"
    )

    abstract = _extract_memory_abstract(body)

    assert abstract == "Concise memory summary for search injection."
    assert "ChatLog" not in abstract
    assert "verbose transcript" not in abstract


class _FakeVikingFS:
    def __init__(self, uri: str, path: str):
        self._uri = uri
        self._path = path

    def _uri_to_path(self, uri: str, ctx=None):
        assert uri == self._uri
        return self._path

    async def exists(self, uri: str, ctx=None):
        return uri == self._uri

    async def stat(self, uri: str, ctx=None):
        assert uri == self._uri
        return {"isDir": False}

    async def read_file(self, uri: str, ctx=None):
        assert uri == self._uri
        return "# Profile\nSingle file reindex source.\n"


async def test_reindex_single_file_uri_acquires_tree_lock_without_not_a_directory(
    agfs_client, test_dir, monkeypatch, caplog
):
    uri = "viking://resources/profile.md"
    path = f"{test_dir}/profile.md"
    agfs_client.write(path, b"# Profile\nSingle file reindex source.\n")

    viking_fs = _FakeVikingFS(uri, path)
    service = SimpleNamespace(
        viking_fs=viking_fs,
        vikingdb_manager=SimpleNamespace(has_queue_manager=True),
    )
    monkeypatch.setattr(reindex_module, "get_service", lambda: service)
    monkeypatch.setattr(reindex_module, "get_viking_fs", lambda: viking_fs)

    executor = ReindexExecutor()
    executor._fetch_existing_record = AsyncMock(return_value=None)
    executor._upsert_context = AsyncMock()
    ctx = RequestContext(
        user=UserIdentifier("acc1", "test_user"),
        role=Role.ROOT,
    )

    try:
        result = await executor._run(
            uri=uri,
            object_type="resource",
            mode="vectors_only",
            ctx=ctx,
        )
    finally:
        pass

    assert result["status"] == "completed"
    assert result["uri"] == uri
    assert result["scanned_records"] == 1
    assert result["rebuilt_records"] == 1
    assert "Not a directory" not in caplog.text
    assert "Failed to create lock file" not in caplog.text
