import asyncio
from types import SimpleNamespace

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.dataclass import MemoryTypeSchema
from openviking.session.memory.extract_loop import ExtractLoop
from openviking.session.memory.memory_isolation_handler import MemoryIsolationHandler
from openviking.session.memory.memory_updater import ExtractContext
from openviking_cli.session.user_id import UserIdentifier


@pytest.fixture(autouse=True)
def _drain_background_tasks():
    """Keep this pure ExtractLoop unit test independent from session client fixtures."""
    yield


class _Provider:
    def __init__(self, messages, schema):
        self.messages = messages
        self._schema = schema
        self.read_file_contents = {}
        self._extract_context = ExtractContext(messages)

    def get_memory_schemas(self, ctx):
        return [self._schema]


class _Operations(dict):
    def __getattr__(self, name):
        return self.get(name)


def _resolve_single_event_for_messages(messages):
    schema = MemoryTypeSchema(
        memory_type="events",
        directory="viking://user/{{ user_space }}/memories/events",
        filename_template="{{ extract_context.get_year(ranges) }}/{{ extract_context.get_month(ranges) }}/{{ extract_context.get_day(ranges) }}/{{ event_name }}.md",
        operation_mode="add_only",
    )
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    provider = _Provider(messages, schema)
    extract_context = provider._extract_context
    isolation_handler = MemoryIsolationHandler(ctx, extract_context, allow_self=True)
    loop = ExtractLoop(
        vlm=SimpleNamespace(model="test-model"),
        viking_fs=object(),
        ctx=ctx,
        context_provider=provider,
        isolation_handler=isolation_handler,
    )
    loop._extract_context = extract_context

    operations = _Operations(
        events=[{"page_id": 100, "event_name": "old_binary_check", "ranges": "0"}],
        delete_uris=[],
        links=[],
    )

    return asyncio.run(loop.resolve_operations(operations))


def test_uri_less_upserts_are_rejected(monkeypatch):
    # A user range bypasses the separate assistant-only guard. Force URI
    # resolution to fail so this pins the URI-less guard itself.
    monkeypatch.setattr(MemoryIsolationHandler, "calculate_memory_uris", lambda *a, **k: [])
    resolved, raw_links = _resolve_single_event_for_messages(
        [
            Message(
                id="u1",
                role="user",
                parts=[TextPart("I completed the gateway restart.")],
                created_at="2026-06-12T19:59:00+00:00",
            )
        ]
    )

    assert raw_links == []
    assert resolved.upsert_operations == []
    assert resolved.errors == []
