import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking_cli import _sdk_import
from openviking_cli.exceptions import ConflictError


@pytest.mark.asyncio
@pytest.mark.parametrize("request_primitive", ["direct_http", "shared_request"])
async def test_http_compat_imports_with_old_sdk_and_preserves_existing_methods(
    monkeypatch, request_primitive
):
    class OldAsyncHTTPClient:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            response = SimpleNamespace(
                is_success=True,
                content=b"bc",
                headers={
                    "x-openviking-start": "1",
                    "X-OpenViking-End": "3",
                    "x-openviking-next-offset": "3",
                    "X-OPENVIKING-EOF": "false",
                    "x-openviking-total-size": "4",
                    "X-OpenViking-Revision": "revision-1",
                },
            )
            self._http = SimpleNamespace(get=AsyncMock(return_value=response))
            if request_primitive == "shared_request":
                self._request = AsyncMock(return_value=response)

        async def read(self, uri, offset=0, limit=-1):
            return f"{uri}:{offset}:{limit}"

        def _handle_response_data(self, response):
            data = response.json()
            if data.get("status") == "error":
                self._raise_exception(data["error"])
            return data

    class OldSyncHTTPClient:
        def __init__(self, *args, **kwargs):
            pass

    old_sdk = SimpleNamespace(
        AsyncHTTPClient=OldAsyncHTTPClient,
        SyncHTTPClient=OldSyncHTTPClient,
        client=SimpleNamespace(),
    )
    monkeypatch.setattr(_sdk_import, "import_openviking_sdk", lambda: old_sdk)

    compat_path = Path(__file__).resolve().parents[2] / "openviking_cli/client/_http_compat.py"
    module_name = "_test_http_compat_with_old_sdk"
    spec = importlib.util.spec_from_file_location(module_name, compat_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    fallback = module.RangeRead(
        content=b"a",
        start=0,
        end=1,
        next_offset=None,
        eof=True,
        total_size=1,
        revision="rev",
    )
    assert fallback.content == b"a"

    client = module.AsyncHTTPClient()
    assert await client.read("viking://resources/legacy", offset=1, limit=2) == (
        "viking://resources/legacy:1:2"
    )
    ranged = await client.read_range(
        "resources/legacy",
        offset=1,
        size=2,
        expected_revision="revision-1",
    )
    assert ranged == module.RangeRead(
        content=b"bc",
        start=1,
        end=3,
        next_offset=3,
        eof=False,
        total_size=4,
        revision="revision-1",
    )
    expected_params = {
        "uri": "viking://resources/legacy",
        "offset": 1,
        "size": 2,
        "expected_revision": "revision-1",
    }
    if request_primitive == "shared_request":
        client._request.assert_awaited_once_with(
            "GET",
            "/api/v1/content/range",
            params=expected_params,
        )
        client._http.get.assert_not_awaited()
    else:
        client._http.get.assert_awaited_once_with(
            "/api/v1/content/range",
            params=expected_params,
        )

    error_response = SimpleNamespace(
        is_success=False,
        content=b"",
        headers={},
        json=lambda: {
            "status": "error",
            "error": {
                "code": "CONFLICT",
                "message": "revision changed",
                "details": {"resource": "viking://resources/legacy"},
            },
        },
    )
    if request_primitive == "shared_request":
        client._request = AsyncMock(return_value=error_response)
    else:
        client._http.get = AsyncMock(return_value=error_response)
    with pytest.raises(ConflictError, match="revision changed"):
        await client.read_range("viking://resources/legacy", size=1)
