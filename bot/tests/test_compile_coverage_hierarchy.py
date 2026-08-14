import copy
import hashlib
import json

import pytest
from vikingbot.compile.coverage import (
    CollectionSnapshot,
    CoverageError,
    CoverageLedger,
    CoverageStatus,
    ReadReceipt,
    ReviewUnit,
    VerifiedSummary,
)


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _unit(
    unit_id: str,
    *,
    body: str | None = None,
    parent: str | None = None,
    start: int | None = None,
    end: int | None = None,
    revision: str | None = "rev-1",
    required: bool = True,
    kind: str = "document",
) -> ReviewUnit:
    return ReviewUnit(
        unit_id,
        "source",
        f"viking://resources/source/{unit_id}",
        kind=kind,
        content_sha256=_digest(body) if body is not None else None,
        parent_unit_id=parent,
        range_start=start,
        range_end=end,
        source_revision=revision,
        required=required,
    )


def _contiguous_snapshot() -> tuple[CollectionSnapshot, dict[str, str]]:
    bodies = {"chunk-1": "first chunk", "chunk-2": "第二块"}
    split = len(bodies["chunk-1"].encode())
    end = split + len(bodies["chunk-2"].encode())
    return (
        CollectionSnapshot(
            "source",
            (
                _unit("document", start=0, end=end),
                _unit("chunk-1", body=bodies["chunk-1"], parent="document", start=0, end=split),
                _unit("chunk-2", body=bodies["chunk-2"], parent="document", start=split, end=end),
            ),
        ),
        bodies,
    )


def test_contiguous_required_chunks_complete_parent_and_preserve_iteration_gate():
    snapshot, bodies = _contiguous_snapshot()
    ledger = CoverageLedger((snapshot,))

    first = ReadReceipt.for_content(snapshot, "chunk-1", bodies["chunk-1"])
    assert (first.range_start, first.range_end, first.source_revision) == (
        0,
        len(bodies["chunk-1"].encode()),
        "rev-1",
    )
    ledger.record_read(first, iteration=2)
    assert ledger.status("document") == CoverageStatus.PENDING

    ledger.record_read(ReadReceipt.for_content(snapshot, "chunk-2", bodies["chunk-2"]), iteration=3)
    assert ledger.status("document") == CoverageStatus.COVERED
    assert ledger.gate_error(before_iteration=3)
    assert ledger.gate_error(before_iteration=4) is None

    with pytest.raises(CoverageError, match="parent containers"):
        ledger.record_read(
            ReadReceipt.for_content(
                snapshot,
                "document",
                "x" * (snapshot.units[0].range_end or 0),
            )
        )
    with pytest.raises(CoverageError, match="parent containers"):
        ledger.skip("document", "Generated parent manifest is represented by its chunks")


def test_gap_in_required_child_ranges_keeps_parent_pending():
    first_body, second_body = "first", "second"
    snapshot = CollectionSnapshot(
        "source",
        (
            _unit("document", start=0, end=12),
            _unit("chunk-1", body=first_body, parent="document", start=0, end=5),
            _unit("chunk-2", body=second_body, parent="document", start=6, end=12),
        ),
    )
    ledger = CoverageLedger((snapshot,))

    ledger.record_read(ReadReceipt.for_content(snapshot, "chunk-1", first_body))
    ledger.record_read(ReadReceipt.for_content(snapshot, "chunk-2", second_body))

    assert ledger.status("document") == CoverageStatus.PENDING
    assert "pending review units: document" in (ledger.gate_error() or "")


def test_receipt_rejects_cross_snapshot_range_revision_and_content_mismatches():
    snapshot, bodies = _contiguous_snapshot()
    other_snapshot = CollectionSnapshot(
        "source",
        snapshot.units + (_unit("optional-index", required=False),),
    )
    ledger = CoverageLedger((snapshot,))

    with pytest.raises(CoverageError, match="collection snapshot"):
        ledger.record_read(ReadReceipt.for_content(other_snapshot, "chunk-1", bodies["chunk-1"]))

    wrong_range = ReadReceipt(
        snapshot.snapshot_id,
        "chunk-1",
        snapshot.units[1].locator,
        _digest(bodies["chunk-1"]),
        len(bodies["chunk-1"].encode()),
        True,
        range_start=1,
        range_end=1 + len(bodies["chunk-1"].encode()),
        source_revision="rev-1",
    )
    with pytest.raises(CoverageError, match="range or revision"):
        ledger.record_read(wrong_range)

    wrong_revision = ReadReceipt(
        snapshot.snapshot_id,
        "chunk-1",
        snapshot.units[1].locator,
        _digest(bodies["chunk-1"]),
        len(bodies["chunk-1"].encode()),
        True,
        range_start=0,
        range_end=len(bodies["chunk-1"].encode()),
        source_revision="rev-2",
    )
    with pytest.raises(CoverageError, match="range or revision"):
        ledger.record_read(wrong_revision)

    with pytest.raises(CoverageError, match="content digest"):
        ledger.record_read(ReadReceipt.for_content(snapshot, "chunk-1", "wrong chunk"))

    wrong_byte_count = ReadReceipt(
        snapshot.snapshot_id,
        "chunk-1",
        snapshot.units[1].locator,
        _digest(bodies["chunk-1"]),
        len(bodies["chunk-1"].encode()) - 1,
        True,
        range_start=0,
        range_end=len(bodies["chunk-1"].encode()),
        source_revision="rev-1",
    )
    with pytest.raises(CoverageError, match="byte_count"):
        ledger.record_read(wrong_byte_count)


def test_reviewed_and_skipped_children_cover_parent_while_all_skipped_skips_it():
    snapshot, bodies = _contiguous_snapshot()
    ledger = CoverageLedger((snapshot,))
    reason = "Generated chunk duplicates the source manifest metadata"

    ledger.record_read(ReadReceipt.for_content(snapshot, "chunk-1", bodies["chunk-1"]))
    ledger.skip("chunk-2", reason)
    assert ledger.status("document") == CoverageStatus.COVERED

    all_skipped = CoverageLedger((snapshot,))
    all_skipped.skip(("chunk-1", "chunk-2"), reason)
    assert all_skipped.status("document") == CoverageStatus.SKIPPED

    all_skipped.record_read(ReadReceipt.for_content(snapshot, "chunk-1", bodies["chunk-1"]))
    assert all_skipped.status("document") == CoverageStatus.COVERED


def test_verified_summary_can_cover_multiple_leaf_children_and_refresh_parent():
    first_body, second_body, summary_body = "first", "second", "verified summary"
    first_end = len(first_body.encode())
    parent_end = first_end + len(second_body.encode())
    summary = _unit("summary", body=summary_body, required=False, kind="summary")
    snapshot = CollectionSnapshot(
        "source",
        (
            _unit("document", start=0, end=parent_end),
            _unit("chunk-1", body=first_body, parent="document", start=0, end=first_end),
            _unit(
                "chunk-2",
                body=second_body,
                parent="document",
                start=first_end,
                end=parent_end,
            ),
            summary,
        ),
        verified_summaries=(VerifiedSummary("summary", ("chunk-1", "chunk-2"), "test-adapter-v2"),),
    )
    ledger = CoverageLedger((snapshot,))

    changed = ledger.cover_from_summary(
        "summary", ReadReceipt.for_content(snapshot, "summary", summary_body), iteration=2
    )

    assert changed == ("chunk-1", "chunk-2")
    assert ledger.status("chunk-1") == CoverageStatus.COVERED
    assert ledger.status("chunk-2") == CoverageStatus.COVERED
    assert ledger.status("document") == CoverageStatus.COVERED
    assert ledger.gate_error(before_iteration=2)
    assert ledger.gate_error(before_iteration=3) is None


def test_optional_summary_does_not_block_parent_or_pending_gate():
    body = "required body"
    body_bytes = len(body.encode())
    snapshot = CollectionSnapshot(
        "source",
        (
            _unit("document", start=0, end=body_bytes),
            _unit("chunk", body=body, parent="document", start=0, end=body_bytes),
            _unit(
                "summary",
                parent="document",
                required=False,
                kind="summary",
            ),
        ),
    )
    ledger = CoverageLedger((snapshot,))

    ledger.record_read(ReadReceipt.for_content(snapshot, "chunk", body))

    assert ledger.status("document") == CoverageStatus.COVERED
    assert ledger.status("summary") == CoverageStatus.PENDING
    assert ledger.gate_error() is None


def test_v2_round_trip_and_v1_default_digest_compatibility():
    legacy_unit = ReviewUnit("legacy", "legacy-source", "viking://resources/legacy/doc")
    legacy_payload = {
        "source_id": "legacy-source",
        "units": [
            {
                "unit_id": "legacy",
                "source_id": "legacy-source",
                "locator": "viking://resources/legacy/doc",
                "kind": "document",
                "substantive": True,
                "content_sha256": None,
                "discovered_bytes": None,
            }
        ],
        "verified_summaries": [],
    }
    expected_snapshot_id = hashlib.sha256(
        json.dumps(
            legacy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    snapshot = CollectionSnapshot("legacy-source", (legacy_unit,), snapshot_id=expected_snapshot_id)
    receipt = ReadReceipt.for_content(snapshot, "legacy", "legacy body")
    legacy_receipt_payload = {
        "snapshot_id": snapshot.snapshot_id,
        "unit_id": "legacy",
        "locator": legacy_unit.locator,
        "content_sha256": _digest("legacy body"),
        "byte_count": len("legacy body".encode()),
        "complete": True,
    }
    assert (
        receipt.receipt_id
        == hashlib.sha256(
            json.dumps(
                legacy_receipt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )

    ledger = CoverageLedger((snapshot,))
    ledger.record_read(receipt)
    serialized = ledger.to_internal_dict()
    assert serialized["version"] == 2
    assert set(serialized["snapshots"][0]["units"][0]) == set(legacy_payload["units"][0])
    assert set(serialized["states"]["legacy"]["receipt"]) == {
        *legacy_receipt_payload,
        "receipt_id",
    }

    v1 = copy.deepcopy(serialized)
    v1["version"] = 1
    restored = CoverageLedger.from_internal_dict(v1)
    assert restored.to_internal_dict()["version"] == 2
    assert restored.status("legacy") == CoverageStatus.REVIEWED

    hierarchy, bodies = _contiguous_snapshot()
    hierarchy_ledger = CoverageLedger((hierarchy,))
    for unit_id, body in bodies.items():
        hierarchy_ledger.record_read(ReadReceipt.for_content(hierarchy, unit_id, body))
    hierarchical_v2 = hierarchy_ledger.to_internal_dict()
    assert CoverageLedger.from_internal_dict(hierarchical_v2).to_internal_dict() == hierarchical_v2

    tampered = copy.deepcopy(hierarchical_v2)
    tampered["snapshots"][0]["units"][1]["range_end"] = 9
    with pytest.raises(CoverageError, match="snapshot digest"):
        CoverageLedger.from_internal_dict(tampered)


def test_hierarchy_rejects_cycles_cross_source_parents_and_invalid_ranges():
    with pytest.raises(CoverageError, match="cycle"):
        CollectionSnapshot(
            "source",
            (
                _unit("one", parent="two"),
                _unit("two", parent="one"),
            ),
        )

    with pytest.raises(CoverageError, match="same collection snapshot and source"):
        CollectionSnapshot("source", (_unit("orphan", parent="other-snapshot"),))

    with pytest.raises(CoverageError, match="source range"):
        _unit("bad-range", start=10, end=10)

    with pytest.raises(CoverageError, match="contained"):
        CollectionSnapshot(
            "source",
            (
                _unit("parent", start=0, end=10),
                _unit("child", parent="parent", start=0, end=11),
            ),
        )
