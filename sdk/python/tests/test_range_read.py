from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openviking_sdk import AsyncHTTPClient, SyncHTTPClient
from openviking_sdk.client import RangeRead


@pytest.mark.asyncio
async def test_async_read_range_forwards_request_and_parses_mixed_case_headers():
    client = AsyncHTTPClient(url="http://localhost:1933")
    client._request = AsyncMock(
        return_value=SimpleNamespace(
            is_success=True,
            content=b"abc",
            headers={
                "x-openviking-start": "4",
                "X-OPENVIKING-END": "7",
                "X-OpenViking-Next-Offset": "7",
                "x-openviking-eof": "FALSE",
                "X-OPENVIKING-TOTAL-SIZE": "11",
                "x-openviking-revision": "revision-1",
            },
        )
    )

    result = await client.read_range(
        "viking://resources/blob.bin",
        offset=4,
        size=3,
        expected_revision="revision-1",
    )

    assert result == RangeRead(
        content=b"abc",
        start=4,
        end=7,
        next_offset=7,
        eof=False,
        total_size=11,
        revision="revision-1",
    )
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/content/range",
        params={
            "uri": "viking://resources/blob.bin",
            "offset": 4,
            "size": 3,
            "expected_revision": "revision-1",
        },
    )


@pytest.mark.asyncio
async def test_async_read_range_tolerates_missing_metadata_headers():
    client = AsyncHTTPClient(url="http://localhost:1933")
    client._request = AsyncMock(
        return_value=SimpleNamespace(
            is_success=True,
            content=b"xyz",
            headers={},
        )
    )

    result = await client.read_range(
        "viking://resources/blob.bin",
        offset=8,
        size=3,
    )

    assert result == RangeRead(
        content=b"xyz",
        start=8,
        end=11,
        next_offset=None,
        eof=True,
        total_size=11,
        revision="",
    )


def test_sync_read_range_forwards_to_async_client():
    client = SyncHTTPClient(url="http://localhost:1933")
    expected = RangeRead(
        content=b"a",
        start=2,
        end=3,
        next_offset=3,
        eof=False,
        total_size=5,
        revision="revision-1",
    )

    client._async_client.read_range = AsyncMock(return_value=expected)
    result = client.read_range(
        "viking://resources/blob.bin",
        offset=2,
        size=1,
        expected_revision="revision-1",
    )

    assert result is expected
    client._async_client.read_range.assert_awaited_once_with(
        "viking://resources/blob.bin",
        offset=2,
        size=1,
        expected_revision="revision-1",
    )
