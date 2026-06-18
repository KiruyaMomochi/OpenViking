import asyncio

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


class _DummyVLM:
    model = "test-model"


class _Operations(dict):
    def __getattr__(self, name):
        return self.get(name)


def _resolve_single_event_for_messages(messages):
    schema = MemoryTypeSchema(
        memory_type="events",
        description="events",
        fields=[],
        directory="viking://user/{{ user_space }}/memories/events",
        filename_template="{{ extract_context.get_year(ranges) }}/{{ extract_context.get_month(ranges) }}/{{ extract_context.get_day(ranges) }}/{{ event_name }}.md",
        content_template=None,
        embedding_template=None,
        enabled=True,
        operation_mode="add_only",
        agent_only=False,
        overview_template=None,
    )
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    extract_context = ExtractContext(messages)
    provider = _Provider(messages, schema)
    provider._extract_context = extract_context
    isolation_handler = MemoryIsolationHandler(ctx, extract_context, allow_self=True)
    loop = ExtractLoop(
        vlm=_DummyVLM(),
        viking_fs=object(),
        ctx=ctx,
        context_provider=provider,
        isolation_handler=isolation_handler,
    )
    loop._extract_context = extract_context

    operations = _Operations(
        events=[
            {
                "page_id": 100,
                "event_name": "old_binary_check",
                "goal": "debug gateway",
                "summary": "The assistant checked the running process and found an old binary.",
                "ranges": "0",
            }
        ],
        delete_uris=[],
        links=[],
    )

    resolved, raw_links = asyncio.run(loop.resolve_operations(operations))

    return resolved, raw_links


def test_events_with_unowned_ranges_are_skipped_before_apply():
    """LLM-created events for assistant-only ranges must not abort the whole batch."""

    resolved, raw_links = _resolve_single_event_for_messages(
        [
            Message(
                id="a1",
                role="assistant",
                parts=[TextPart("I checked the running process and found an old binary.")],
                created_at="2026-06-12T19:59:00+00:00",
            )
        ]
    )

    assert raw_links == []
    assert resolved.upsert_operations == []
    assert resolved.errors == []


def test_events_with_user_ranges_still_resolve_to_self_memory_uri():
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
    assert len(resolved.upsert_operations) == 1
    assert resolved.errors == []
    assert resolved.upsert_operations[0].uris == [
        "viking://user/default/memories/events/2026/06/12/old_binary_check.md"
    ]
