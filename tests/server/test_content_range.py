# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Focused tests for additive raw byte-range reads."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.server.routers import content as content_router
from openviking.service.fs_service import MAX_BYTE_RANGE_SIZE, FSService
from openviking_cli.exceptions import ConflictError, InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier


@pytest.fixture
def request_context() -> RequestContext:
    return RequestContext(
        user=UserIdentifier("default", "range-reader"),
        role=Role.USER,
    )


def _revision(size: int, mod_time: str) -> str:
    payload = json.dumps(
        [size, mod_time],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_fs_read_range_directly_forwards_offset_and_size(request_context):
    uri = "viking://resources/blob.bin"
    stat = {"size": 12, "modTime": "2026-08-13T09:30:00Z"}
    viking_fs = SimpleNamespace(
        stat=AsyncMock(side_effect=[stat.copy(), stat.copy()]),
        read=AsyncMock(return_value=b"\x02\x03\x04\x05"),
    )

    result = await FSService(viking_fs=viking_fs).read_range(
        uri,
        ctx=request_context,
        offset=2,
        size=4,
    )

    viking_fs.read.assert_awaited_once_with(uri, 2, 4, request_context)
    assert result == {
        "content": b"\x02\x03\x04\x05",
        "start": 2,
        "end": 6,
        "next_offset": 6,
        "eof": False,
        "total_size": 12,
        "revision": _revision(12, "2026-08-13T09:30:00Z"),
    }


@pytest.mark.asyncio
async def test_fs_read_range_rejects_expected_revision_before_read(request_context):
    uri = "viking://resources/blob.bin"
    viking_fs = SimpleNamespace(
        stat=AsyncMock(return_value={"size": 3, "modTime": "rev-2"}),
        read=AsyncMock(),
    )

    with pytest.raises(ConflictError, match="does not match"):
        await FSService(viking_fs=viking_fs).read_range(
            uri,
            ctx=request_context,
            size=1,
            expected_revision="rev-1",
        )

    viking_fs.read.assert_not_awaited()
    viking_fs.stat.assert_awaited_once_with(uri, ctx=request_context)


@pytest.mark.asyncio
async def test_fs_read_range_rejects_revision_change_during_read(request_context):
    uri = "viking://resources/blob.bin"
    viking_fs = SimpleNamespace(
        stat=AsyncMock(
            side_effect=[
                {"size": 3, "modTime": "rev-1"},
                {"size": 3, "modTime": "rev-2"},
            ]
        ),
        read=AsyncMock(return_value=b"a"),
    )

    with pytest.raises(ConflictError, match="changed while reading"):
        await FSService(viking_fs=viking_fs).read_range(
            uri,
            ctx=request_context,
            size=1,
        )


@pytest.mark.asyncio
async def test_fs_read_range_reports_eof_with_exclusive_end(request_context):
    uri = "viking://resources/blob.bin"
    stat = {"size": 6, "modTime": "stable"}
    viking_fs = SimpleNamespace(
        stat=AsyncMock(side_effect=[stat.copy(), stat.copy()]),
        read=AsyncMock(return_value=b"ef"),
    )

    result = await FSService(viking_fs=viking_fs).read_range(
        uri,
        ctx=request_context,
        offset=4,
        size=10,
    )

    assert result["content"] == b"ef"
    assert result["end"] == 6
    assert result["next_offset"] is None
    assert result["eof"] is True


@pytest.mark.asyncio
async def test_fs_read_range_enforces_maximum_size(request_context):
    service = FSService(viking_fs=SimpleNamespace())

    with pytest.raises(InvalidArgumentError, match="size must be between"):
        await service.read_range(
            "viking://resources/blob.bin",
            ctx=request_context,
            size=MAX_BYTE_RANGE_SIZE + 1,
        )


@pytest.mark.asyncio
async def test_fs_read_range_rejects_offset_beyond_eof(request_context):
    viking_fs = SimpleNamespace(
        stat=AsyncMock(return_value={"size": 3, "modTime": "now", "isDir": False}),
        read=AsyncMock(),
    )

    with pytest.raises(InvalidArgumentError, match="beyond the file size"):
        await FSService(viking_fs=viking_fs).read_range(
            "viking://resources/file.bin",
            ctx=request_context,
            offset=4,
            size=1,
        )

    viking_fs.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_fs_read_range_rejects_directory_before_read(request_context):
    uri = "viking://resources/directory"
    viking_fs = SimpleNamespace(
        stat=AsyncMock(return_value={"size": 0, "modTime": "stable", "isDir": True}),
        read=AsyncMock(),
    )

    with pytest.raises(InvalidArgumentError, match="Cannot read directory as file") as exc_info:
        await FSService(viking_fs=viking_fs).read_range(
            uri,
            ctx=request_context,
            size=1,
        )

    assert exc_info.value.details == {
        "resource": uri,
        "expected": "file",
        "actual": "directory",
    }
    viking_fs.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_range_route_returns_bytes_and_metadata_headers(monkeypatch, request_context):
    read_range = AsyncMock(
        return_value={
            "content": b"\x00\xff",
            "start": 7,
            "end": 9,
            "next_offset": 9,
            "eof": False,
            "total_size": 20,
            "revision": "stable-revision",
        }
    )
    monkeypatch.setattr(
        content_router,
        "get_service",
        lambda: SimpleNamespace(fs=SimpleNamespace(read_range=read_range)),
    )

    response = await content_router.read_range(
        uri="viking://resources/blob.bin",
        offset=7,
        size=2,
        expected_revision="stable-revision",
        _ctx=request_context,
    )

    assert response.body == b"\x00\xff"
    assert response.media_type == "application/octet-stream"
    assert response.headers["x-openviking-start"] == "7"
    assert response.headers["x-openviking-end"] == "9"
    assert response.headers["x-openviking-next-offset"] == "9"
    assert response.headers["x-openviking-eof"] == "false"
    assert response.headers["x-openviking-total-size"] == "20"
    assert response.headers["x-openviking-revision"] == "stable-revision"
    read_range.assert_awaited_once_with(
        "viking://resources/blob.bin",
        ctx=request_context,
        offset=7,
        size=2,
        expected_revision="stable-revision",
    )


@pytest.mark.asyncio
async def test_existing_line_read_and_download_paths_remain_unchanged(request_context):
    uri = "viking://resources/legacy.txt"
    viking_fs = SimpleNamespace(
        read_file=AsyncMock(return_value="one\ntwo\nthree\n"),
        read_file_bytes=AsyncMock(return_value=b"one\ntwo\nthree\n"),
    )
    service = FSService(viking_fs=viking_fs)

    assert await service.read(uri, ctx=request_context, offset=1, limit=1) == "two\n"
    assert await service.read_file_bytes(uri, ctx=request_context) == b"one\ntwo\nthree\n"
    viking_fs.read_file.assert_awaited_once_with(uri, ctx=request_context)
    viking_fs.read_file_bytes.assert_awaited_once_with(uri, ctx=request_context)
