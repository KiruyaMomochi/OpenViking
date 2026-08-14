import asyncio
import json
from types import SimpleNamespace

import pytest
from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.agent.tools.compile import (
    CompileScopedTool,
    ReportCoverageTool,
    SubmitWikiBundleTool,
)
from vikingbot.compile.coverage import (
    CollectionSnapshot,
    CoverageLedger,
    CoverageStatus,
    ReviewUnit,
    VerifiedSummary,
)
from vikingbot.compile.models import (
    CompileIncomplete,
    CompileLimits,
    CompileReviewView,
    CompileTask,
    SanitizedCompileRequest,
    WikiBundleDraft,
    utc_now,
)
from vikingbot.compile.service import BotCompileService
from vikingbot.providers.base import LLMResponse


class _ReadTool(Tool):
    def __init__(self, name: str, content_by_uri: dict[str, str] | None = None):
        self._name = name
        self.content_by_uri = content_by_uri or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test read tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, tool_context: ToolContext, **kwargs) -> str:
        del tool_context
        if self.name != "openviking_multi_read":
            return json.dumps(kwargs)
        sections = []
        for uri in kwargs["uris"]:
            sections.extend(
                [
                    f"--- START OF {uri} ---",
                    self.content_by_uri[uri],
                    f"--- END OF {uri} ---",
                ]
            )
        return "\n".join(sections)


def _scoped(
    tool: Tool,
    ledger: CoverageLedger,
    roots: tuple[str, ...],
    *,
    limits: CompileLimits | None = None,
    result_budget: dict[str, int] | None = None,
    evidence_cache: dict[str, str] | None = None,
) -> CompileScopedTool:
    return CompileScopedTool(
        tool,
        roots=roots,
        limits=limits or CompileLimits(),
        result_budget=result_budget if result_budget is not None else {"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        evidence_cache=evidence_cache,
    )


class _ContextClient:
    def __init__(self, root: str, files: dict[str, bytes]):
        self.client = self
        self.root = root
        self.files = files
        self.range_calls = []
        self.raw_calls = []

    async def overview(self, uri):
        assert uri == self.root
        return "Source overview"

    async def list_resources(self, *, path, recursive, node_limit, show_all_hidden=False):
        assert path == self.root
        assert node_limit > 0
        if not recursive:
            assert show_all_hidden is True
        return [
            {
                "uri": locator,
                "name": locator.rsplit("/", 1)[-1],
                "isDir": False,
                "size": len(content),
            }
            for locator, content in self.files.items()
        ]

    async def read_range(
        self,
        locator,
        offset=0,
        size=16 * 1024 * 1024,
        expected_revision=None,
    ):
        content = self.files[locator]
        revision = f"revision-{len(content)}"
        assert expected_revision in {None, revision}
        self.range_calls.append((locator, offset, size, expected_revision))
        chunk = content[offset : offset + size]
        end = offset + len(chunk)
        return SimpleNamespace(
            content=chunk,
            start=offset,
            end=end,
            next_offset=None if end >= len(content) else end,
            eof=end >= len(content),
            total_size=len(content),
            revision=revision,
        )

    async def read_raw(self, locator):
        self.raw_calls.append(locator)
        return self.files[locator].decode()


class _MapProvider:
    def __init__(self):
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(content="bounded faithful source summary")


@pytest.mark.asyncio
async def test_large_source_uses_stream_map_then_summary_delivery_covers_ranges(tmp_path):
    root = "viking://resources/source"
    locator = f"{root}/large.txt"
    client = _ContextClient(root, {locator: ("important fact\n" * 100).encode()})
    provider = _MapProvider()
    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path)),
        limits=CompileLimits(
            tool_result_bytes=512,
            tool_total_result_bytes=1024,
            source_range_bytes=32,
            source_total_io_bytes=10_000,
            source_max_record_bytes=1000,
            source_packet_chars=80,
            source_max_packets=100,
            source_map_calls=10,
            source_map_input_chars=100_000,
            source_map_output_chars=10_000,
            source_map_timeout_seconds=1,
            source_map_max_tokens=100,
        ),
    )

    context = await service._build_source_context(
        client,
        [root],
        reason="Compile facts",
        skill_contract="Write a report",
        provider=provider,
        model="map",
        session_id="cmp_large",
    )

    assert context.large_mode is True
    assert client.range_calls
    assert client.raw_calls == []
    assert provider.calls
    assert context.sources[0]["verified_summary_views"] == list(context.review_views)
    reader = CompileScopedTool(
        _ReadTool("openviking_multi_read"),
        roots=(root,),
        limits=service.limits,
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=context.coverage,
        review_views=context.review_views,
    )
    result = await reader.execute(
        ToolContext(iteration=1),
        uris=list(context.review_views),
    )

    assert "bounded faithful source summary" in result
    assert context.coverage.gate_error(before_iteration=1)
    assert context.coverage.gate_error(before_iteration=2) is None


@pytest.mark.asyncio
async def test_small_source_keeps_fast_path_without_range_or_map_calls(tmp_path):
    root = "viking://resources/source"
    locator = f"{root}/small.md"
    client = _ContextClient(root, {locator: b"small body"})
    provider = _MapProvider()
    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path))
    )

    context = await service._build_source_context(
        client,
        [root],
        provider=provider,
        model="map",
    )

    assert context.large_mode is False
    assert client.range_calls == []
    assert client.raw_calls == []
    assert provider.calls == []
    assert context.review_views == {}


@pytest.mark.asyncio
async def test_unknown_inventory_size_selects_large_range_path(tmp_path):
    root = "viking://resources/source"
    locator = f"{root}/unknown.txt"

    class UnknownSizeClient(_ContextClient):
        async def list_resources(
            self, *, path, recursive, node_limit, show_all_hidden=False
        ):
            assert path == self.root
            assert node_limit > 0
            if not recursive:
                assert show_all_hidden is True
            return [{"uri": locator, "name": "unknown.txt", "isDir": False}]

    client = UnknownSizeClient(root, {locator: b"source with unknown inventory size"})
    provider = _MapProvider()
    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path)),
        limits=CompileLimits(
            source_range_bytes=8,
            source_total_io_bytes=1024,
            source_packet_chars=32,
            source_map_calls=10,
            source_map_input_chars=100_000,
            source_map_output_chars=10_000,
            source_map_timeout_seconds=1,
            source_map_max_tokens=100,
        ),
    )

    context = await service._build_source_context(
        client,
        [root],
        provider=provider,
        model="map",
    )

    assert context.large_mode is True
    assert client.range_calls
    assert provider.calls
    assert context.review_views


@pytest.mark.asyncio
async def test_multiple_sources_leave_submission_blocked_when_one_is_unread(tmp_path):
    roots = ("viking://resources/one", "viking://resources/two")

    class Client:
        client = None

        def __init__(self):
            self.client = self

        async def overview(self, uri):
            return f"Overview of {uri}"

        async def list_resources(self, *, path, recursive, node_limit, show_all_hidden=False):
            assert node_limit > 0
            if not recursive:
                assert show_all_hidden is True
            return [
                {
                    "uri": f"{path}/document.md",
                    "name": "document.md",
                    "isDir": False,
                }
            ]

    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path))
    )
    source_context = await service._build_source_context(Client(), list(roots))
    first, second = source_context.coverage.units
    reader = _scoped(
        _ReadTool("openviking_multi_read", {first.locator: "reviewed source one"}),
        source_context.coverage,
        roots,
    )

    await reader.execute(ToolContext(), uris=[first.locator])
    result = await SubmitWikiBundleTool(
        source_ids={"src_1", "src_2"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        coverage=source_context.coverage,
    ).execute(ToolContext(), pages=[])

    assert source_context.coverage.status(first.unit_id) == CoverageStatus.REVIEWED
    assert source_context.coverage.status(second.unit_id) == CoverageStatus.PENDING
    assert result.startswith("Error: Coverage gate rejected the bundle")
    assert second.unit_id in result


@pytest.mark.asyncio
async def test_same_iteration_read_cannot_race_submission_gate():
    uri = "viking://resources/source/document.md"
    unit = ReviewUnit("doc", "src_1", uri)
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    reader = _scoped(
        _ReadTool("openviking_multi_read", {uri: "source content"}),
        ledger,
        ("viking://resources/source",),
    )
    submit = SubmitWikiBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        coverage=ledger,
    )
    current = ToolContext(iteration=1, iteration_limit=10)

    # AgentLoop may complete the read coroutine before the concurrently issued
    # stop tool.  Both calls still belong to one assistant turn, so the model has
    # not received the read result when it authored the submission.
    await reader.execute(current, uris=[uri])
    same_turn_submit = await submit.execute(current, pages=[])

    assert same_turn_submit.startswith("Error: Coverage gate rejected")
    assert "not usable until its tool result is delivered" in same_turn_submit
    assert submit.candidate_coverage_issue is not None
    next_turn_submit = await submit.execute(ToolContext(iteration=2, iteration_limit=10), pages=[])
    assert next_turn_submit.startswith("Wiki bundle accepted")
    assert submit.candidate_coverage_issue is None


@pytest.mark.asyncio
async def test_harness_verified_empty_adapter_unit_is_skipped_only_after_delivery_budget():
    uri = "viking://resources/source/rollout.jsonl"
    unit = ReviewUnit("codex", "src_1", uri, kind="codex_session")
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))

    async def empty_codex_view():
        return ""

    review_view = CompileReviewView(empty_codex_view, ("codex",))

    blocked_tool = CompileScopedTool(
        _ReadTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(tool_total_result_bytes=1),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        review_views={uri: review_view},
    )
    blocked = await blocked_tool.execute(ToolContext(iteration=1, iteration_limit=10), uris=[uri])

    assert blocked == "Error: Compile task tool-result budget exceeded."
    assert ledger.status("codex") == CoverageStatus.PENDING

    tool = CompileScopedTool(
        _ReadTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        review_views={uri: review_view},
    )
    result = await tool.execute(ToolContext(iteration=2, iteration_limit=10), uris=[uri])

    assert "no substantive Codex conversation content" in result
    assert ledger.status("codex") == CoverageStatus.SKIPPED
    assert ledger.gate_error(before_iteration=2)
    assert ledger.gate_error(before_iteration=3) is None


@pytest.mark.asyncio
async def test_harness_empty_adapter_read_upgrades_an_agent_skip():
    uri = "viking://resources/source/rollout.jsonl"
    unit = ReviewUnit("codex", "src_1", uri, kind="codex_session")
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    ledger.skip(
        "codex",
        "Metadata-only Codex rollout appears to contain no conversation records",
        iteration=1,
    )

    async def empty_codex_view():
        return ""

    tool = CompileScopedTool(
        _ReadTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        review_views={uri: CompileReviewView(empty_codex_view, ("codex",))},
    )

    result = await tool.execute(ToolContext(iteration=2, iteration_limit=10), uris=[uri])

    assert "no substantive Codex conversation content" in result
    assert ledger.status("codex") == CoverageStatus.SKIPPED
    assert ledger.gate_error(before_iteration=2)
    assert ledger.gate_error(before_iteration=3) is None


@pytest.mark.asyncio
async def test_discovered_large_unit_is_rejected_before_underlying_read():
    uri = "viking://resources/source/huge.bin"
    unit = ReviewUnit(
        "huge",
        "src_1",
        uri,
        discovered_bytes=CompileLimits().tool_result_bytes + 1,
    )
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))

    class MustNotRead(_ReadTool):
        async def execute(self, tool_context, **kwargs):
            raise AssertionError(f"oversized unit was read: {tool_context}, {kwargs}")

    tool = _scoped(
        MustNotRead("openviking_multi_read"),
        ledger,
        ("viking://resources/source",),
    )

    result = await tool.execute(ToolContext(), uris=[uri])

    assert "exceeds its safe read budget" in result
    assert ledger.status("huge") == CoverageStatus.PENDING


@pytest.mark.asyncio
async def test_adapted_batch_preserves_each_item_when_combined_result_is_oversized():
    first_uri = "viking://resources/source/one.jsonl"
    second_uri = "viking://resources/source/two.jsonl"
    units = (
        ReviewUnit("one", "src_1", first_uri, kind="codex_session"),
        ReviewUnit("two", "src_1", second_uri, kind="codex_session"),
    )
    ledger = CoverageLedger((CollectionSnapshot("src_1", units),))
    calls = []

    async def first_view():
        calls.append("one")
        return "a" * 40

    async def second_view():
        calls.append("two")
        return "b" * 40

    tool = CompileScopedTool(
        _ReadTool("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(tool_result_bytes=280),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        review_views={
            first_uri: CompileReviewView(first_view, ("one",)),
            second_uri: CompileReviewView(second_view, ("two",)),
        },
    )

    result = await tool.execute(ToolContext(), uris=[first_uri, second_uri])

    assert "a" * 40 in result
    assert "item omitted because a delivery budget was exhausted" in result
    assert calls == ["one", "two"]
    assert ledger.status("one") == CoverageStatus.REVIEWED
    assert ledger.status("two") == CoverageStatus.PENDING


@pytest.mark.asyncio
async def test_delegated_batch_records_fitting_items_when_one_item_is_oversized():
    first_uri = "viking://resources/source/first.md"
    second_uri = "viking://resources/source/second.md"
    units = (
        ReviewUnit("first", "src_1", first_uri),
        ReviewUnit("second", "src_1", second_uri),
    )
    ledger = CoverageLedger((CollectionSnapshot("src_1", units),))
    tool = _scoped(
        _ReadTool(
            "openviking_multi_read",
            {first_uri: "a" * 200, second_uri: "kept"},
        ),
        ledger,
        ("viking://resources/source",),
        limits=CompileLimits(tool_result_bytes=240),
    )

    result = await tool.execute(ToolContext(iteration=1), uris=[first_uri, second_uri])

    assert "item omitted because a delivery budget was exhausted" in result
    assert "kept" in result
    assert ledger.status("first") == CoverageStatus.PENDING
    assert ledger.status("second") == CoverageStatus.REVIEWED


@pytest.mark.asyncio
async def test_multi_read_isolates_one_item_exception_and_reviews_later_item():
    first_uri = "viking://resources/source/broken.md"
    second_uri = "viking://resources/source/good.md"
    ledger = CoverageLedger(
        (
            CollectionSnapshot(
                "src_1",
                (
                    ReviewUnit("broken", "src_1", first_uri),
                    ReviewUnit("good", "src_1", second_uri),
                ),
            ),
        )
    )

    class SometimesRaises(_ReadTool):
        async def execute(self, tool_context, **kwargs):
            if kwargs["uris"] == [first_uri]:
                raise RuntimeError("source temporarily unavailable")
            return await super().execute(tool_context, **kwargs)

    tool = _scoped(
        SometimesRaises("openviking_multi_read", {second_uri: "good body"}),
        ledger,
        ("viking://resources/source",),
    )

    result = await tool.execute(ToolContext(iteration=1), uris=[first_uri, second_uri])

    assert "source temporarily unavailable" in result
    assert "good body" in result
    assert ledger.status("broken") == CoverageStatus.PENDING
    assert ledger.status("good") == CoverageStatus.REVIEWED


@pytest.mark.asyncio
async def test_one_delivery_reviews_overlapping_source_units_for_same_locator():
    locator = "viking://resources/source/nested/shared.md"
    snapshots = (
        CollectionSnapshot(
            "src_1",
            (ReviewUnit("src_1:shared", "src_1", locator),),
        ),
        CollectionSnapshot(
            "src_2",
            (ReviewUnit("src_2:shared", "src_2", locator),),
        ),
    )
    ledger = CoverageLedger(snapshots)
    reader = _scoped(
        _ReadTool("openviking_multi_read", {locator: "shared material"}),
        ledger,
        ("viking://resources/source", "viking://resources/source/nested"),
    )

    await reader.execute(ToolContext(), uris=[locator])

    assert ledger.summary()["counts"]["reviewed"] == 2
    assert ledger.is_complete()


@pytest.mark.asyncio
async def test_source_inventory_walks_beyond_depth_three_and_includes_hidden_files(tmp_path):
    root = "viking://resources/source"
    tree = {
        root: [
            {"uri": f"{root}/.private.md", "name": ".private.md", "isDir": False},
            {"uri": f"{root}/one", "name": "one", "isDir": True},
        ],
        f"{root}/one": [{"uri": f"{root}/one/two", "name": "two", "isDir": True}],
        f"{root}/one/two": [{"uri": f"{root}/one/two/three", "name": "three", "isDir": True}],
        f"{root}/one/two/three": [
            {"uri": f"{root}/one/two/three/four", "name": "four", "isDir": True}
        ],
        f"{root}/one/two/three/four": [
            {
                "uri": f"{root}/one/two/three/four/deep.md",
                "name": "deep.md",
                "isDir": False,
            }
        ],
    }

    class Client:
        client = None

        def __init__(self):
            self.client = self
            self.inventory_calls = []

        async def overview(self, uri):
            return f"Overview of {uri}"

        async def list_resources(self, *, path, recursive, node_limit, show_all_hidden=False):
            assert node_limit > 0
            if recursive:
                return []
            assert show_all_hidden is True
            self.inventory_calls.append(path)
            return tree.get(path, [])

    client = Client()
    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path))
    )

    source_context = await service._build_source_context(client, [root])

    assert {unit.locator for unit in source_context.coverage.units} == {
        f"{root}/.private.md",
        f"{root}/one/two/three/four/deep.md",
    }
    assert client.inventory_calls == [
        root,
        f"{root}/one",
        f"{root}/one/two",
        f"{root}/one/two/three",
        f"{root}/one/two/three/four",
    ]


@pytest.mark.asyncio
async def test_inventory_overflow_is_compile_incomplete_and_finishes_incomplete(
    monkeypatch, tmp_path
):
    root = "viking://resources/source"

    class Client:
        async def list_resources(self, **kwargs):
            assert kwargs == {
                "path": root,
                "recursive": False,
                "node_limit": 2,
                "show_all_hidden": True,
            }
            return [
                {"uri": f"{root}/one.md", "name": "one.md", "isDir": False},
                {"uri": f"{root}/two.md", "name": "two.md", "isDir": False},
            ]

    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path)),
        limits=CompileLimits(coverage_inventory_entries=1),
    )
    with pytest.raises(CompileIncomplete) as raised:
        await service._snapshot_source_inventory(Client(), root)
    assert raised.value.code == "INPUT_SNAPSHOT_INCOMPLETE"

    request = SanitizedCompileRequest.model_validate(
        {
            "from": [root],
            "to": "viking://resources/wiki",
            "skill": "viking://agent/skills/wiki",
            "reason": "Compile",
        }
    )
    task = CompileTask(
        task_id="cmp_inventory_overflow",
        principal_scope="owner",
        sanitized_request=request,
        status="accepted",
        stage="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    await service.store.create(task)

    async def execute(*args, **kwargs):
        del args, kwargs
        await service._snapshot_source_inventory(Client(), root)

    monkeypatch.setattr(service, "_execute_task", execute)
    await service._run_task(task.task_id, request, {})

    completed = await service.store.get(task.task_id)
    assert completed is not None
    assert completed.status == "incomplete"
    assert completed.stage == "incomplete"
    assert completed.error is not None
    assert completed.error.code == "INPUT_SNAPSHOT_INCOMPLETE"


@pytest.mark.asyncio
async def test_inventory_snapshot_budget_is_shared_across_source_roots(tmp_path):
    roots = ("viking://resources/one", "viking://resources/two")

    class Client:
        async def list_resources(self, *, path, **kwargs):
            del kwargs
            return [{"uri": f"{path}/doc.md", "name": "doc.md", "isDir": False}]

    service = BotCompileService(
        agent_loop=SimpleNamespace(config=SimpleNamespace(bot_data_path=tmp_path)),
        limits=CompileLimits(coverage_inventory_entries=1),
    )
    budget = {"remaining": 1}

    await service._snapshot_source_inventory(Client(), roots[0], task_budget=budget)
    with pytest.raises(CompileIncomplete, match="task snapshot budget"):
        await service._snapshot_source_inventory(Client(), roots[1], task_budget=budget)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("openviking_search", {"target_uri": "viking://resources/source", "query": "x"}),
        ("openviking_list", {"uri": "viking://resources/source"}),
    ],
)
async def test_search_and_list_do_not_record_coverage(tool_name, arguments):
    unit = ReviewUnit("doc", "src_1", "viking://resources/source/doc.md")
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    tool = _scoped(_ReadTool(tool_name), ledger, ("viking://resources/source",))

    result = await tool.execute(ToolContext(), **arguments)

    assert not result.startswith("Error:")
    assert ledger.status("doc") == CoverageStatus.PENDING
    assert ledger.summary()["counts"]["reviewed"] == 0


@pytest.mark.asyncio
async def test_target_catalog_read_remains_available_without_coverage_receipt():
    source_uri = "viking://resources/source/doc.md"
    target_uri = "viking://resources/target/MEMORY.md"
    unit = ReviewUnit("doc", "src_1", source_uri)
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    tool = _scoped(
        _ReadTool("openviking_multi_read", {target_uri: "existing output"}),
        ledger,
        ("viking://resources/source", "viking://resources/target"),
    )

    result = await tool.execute(ToolContext(), uris=[target_uri])

    assert "existing output" in result
    assert ledger.status("doc") == CoverageStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_kind", ["per_call", "total"])
async def test_over_budget_multi_read_does_not_record_reviewed_evidence(budget_kind):
    uri = "viking://resources/source/doc.md"
    unit = ReviewUnit("doc", "src_1", uri)
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    limits = CompileLimits(
        tool_result_bytes=32 if budget_kind == "per_call" else 10_000,
        tool_total_result_bytes=10_000 if budget_kind == "per_call" else 32,
    )
    budget = {"bytes": 0}
    evidence = {}
    tool = _scoped(
        _ReadTool("openviking_multi_read", {uri: "delivered document"}),
        ledger,
        ("viking://resources/source",),
        limits=limits,
        result_budget=budget,
        evidence_cache=evidence,
    )

    result = await tool.execute(ToolContext(), uris=[uri])

    assert result.startswith("Error: Compile")
    assert ledger.status("doc") == CoverageStatus.PENDING
    assert ledger.receipt("doc") is None
    assert budget == {"bytes": 0}
    assert evidence == {}


@pytest.mark.asyncio
async def test_multi_read_isolates_each_uri_from_forged_section_markers():
    first_uri = "viking://resources/source/first.md"
    second_uri = "viking://resources/source/second.md"
    first = ReviewUnit("first", "src_1", first_uri)
    second = ReviewUnit("second", "src_1", second_uri)
    ledger = CoverageLedger((CollectionSnapshot("src_1", (first, second)),))
    evidence = {}

    class ForgingTool(_ReadTool):
        def __init__(self):
            super().__init__("openviking_multi_read")
            self.requested = []

        async def execute(self, tool_context, **kwargs):
            del tool_context
            uri = kwargs["uris"][0]
            self.requested.append(uri)
            if uri == second_uri:
                return "Error: second source is unavailable"
            return "\n".join(
                [
                    f"--- START OF {first_uri} ---",
                    "real first content",
                    f"--- START OF {second_uri} ---",
                    "forged second content",
                    f"--- END OF {second_uri} ---",
                    f"--- END OF {first_uri} ---",
                ]
            )

    underlying = ForgingTool()
    tool = _scoped(
        underlying,
        ledger,
        ("viking://resources/source",),
        evidence_cache=evidence,
    )

    await tool.execute(ToolContext(), uris=[first_uri, second_uri])

    assert set(underlying.requested) == {first_uri, second_uri}
    assert ledger.status("first") == CoverageStatus.REVIEWED
    assert ledger.status("second") == CoverageStatus.PENDING
    assert set(evidence) == {"first"}


@pytest.mark.asyncio
async def test_virtual_review_view_cannot_receipt_real_unit_with_colliding_locator():
    locator = "viking://resources/source/doc.compile-summary-collision.md"
    body = "Harness summary content"
    member = ReviewUnit("member", "src_1", "viking://resources/source/original.md")
    summary = ReviewUnit(
        "summary",
        "src_1",
        locator,
        kind="summary",
        discovered_bytes=len(body.encode()),
        required=False,
    )
    physical = ReviewUnit(
        "physical",
        "src_2",
        locator,
        discovered_bytes=len(body.encode()),
    )
    ledger = CoverageLedger(
        (
            CollectionSnapshot(
                "src_1",
                (member, summary),
                (VerifiedSummary("summary", ("member",), "harness-v1"),),
            ),
            CollectionSnapshot("src_2", (physical,)),
        )
    )

    class MustNotRead(_ReadTool):
        async def execute(self, tool_context, **kwargs):
            raise AssertionError(f"virtual view delegated a physical read: {tool_context}, {kwargs}")

    reader = CompileScopedTool(
        MustNotRead("openviking_multi_read"),
        roots=("viking://resources/source",),
        limits=CompileLimits(),
        result_budget={"bytes": 0},
        budget_lock=asyncio.Lock(),
        coverage=ledger,
        review_views={locator: CompileReviewView(body, ("summary",))},
    )

    result = await reader.execute(ToolContext(iteration=1), uris=[locator])

    assert body in result
    assert ledger.status("summary") == CoverageStatus.REVIEWED
    assert ledger.status("member") == CoverageStatus.COVERED
    assert ledger.status("physical") == CoverageStatus.PENDING
    assert ledger.is_complete() is False


@pytest.mark.asyncio
async def test_verified_summary_read_can_cover_members_and_unlock_submission():
    summary = ReviewUnit("summary", "src_1", "viking://resources/source/summary.md", kind="summary")
    first = ReviewUnit("first", "src_1", "viking://resources/source/first.md")
    second = ReviewUnit("second", "src_1", "viking://resources/source/second.md")
    snapshot = CollectionSnapshot(
        "src_1",
        (summary, first, second),
        (VerifiedSummary("summary", ("first", "second"), "harness-v1"),),
    )
    ledger = CoverageLedger((snapshot,))
    reader = _scoped(
        _ReadTool("openviking_multi_read", {summary.locator: "verified summary"}),
        ledger,
        ("viking://resources/source",),
    )

    await reader.execute(ToolContext(), uris=[summary.locator])
    report = await ReportCoverageTool(ledger).execute(
        ToolContext(),
        action="cover",
        unit_ids=["first", "second"],
        summary_unit_id="summary",
    )
    submitted = await SubmitWikiBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        coverage=ledger,
    ).execute(ToolContext(), pages=[])

    assert json.loads(report)["complete"] is True
    assert ledger.status("first") == CoverageStatus.COVERED
    assert submitted.startswith("Wiki bundle accepted")


@pytest.mark.asyncio
async def test_coverage_report_rejects_skip_without_a_specific_reason():
    unit = ReviewUnit("doc", "src_1", "viking://resources/source/doc.md")
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))

    result = await ReportCoverageTool(ledger).execute(
        ToolContext(), action="skip", unit_ids=["doc"], reason="not relevant"
    )

    assert result.startswith("Error: Invalid coverage report")
    assert "source-specific reason" in result
    assert ledger.status("doc") == CoverageStatus.PENDING


def test_coverage_is_internal_task_state_and_not_part_of_a_bundle():
    request = SanitizedCompileRequest.model_validate(
        {
            "from": ["viking://resources/source"],
            "to": "viking://resources/wiki",
            "skill": "viking://agent/skills/wiki",
            "reason": "Compile",
        }
    )
    unit = ReviewUnit("doc", "src_1", "viking://resources/source/doc.md")
    ledger = CoverageLedger((CollectionSnapshot("src_1", (unit,)),))
    task = CompileTask(
        task_id="cmp_internal_coverage",
        principal_scope="owner",
        sanitized_request=request,
        status="running",
        stage="agent",
        created_at=utc_now(),
        updated_at=utc_now(),
        coverage_summary=ledger.to_internal_dict(),
    )
    bundle = WikiBundleDraft.model_validate({"pages": [], "files": [], "links": []})

    assert "coverage_summary" not in task.public_dict()
    assert "coverage" not in bundle.model_dump()
    assert "coverage_summary" not in bundle.model_dump()
