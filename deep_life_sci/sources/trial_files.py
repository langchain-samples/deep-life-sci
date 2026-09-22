"""Stage oversized trial records before they cross the PTC bridge.

Small records retain their existing shape. Large ones become explicit file manifests;
the complete JSON is for Python, while indexed sections let read-only analysts select
evidence without reading the whole record. A file is not a token optimisation unless
the reader selects sections. Outcomes keep their own group definitions and analyses;
adverse-event sections repeat the module's context and denominators.

24 KB of compact UTF-8 JSON is a conservative spill threshold, not a token estimate.
Sections are semantic units, not hard-size chunks: a complex outcome can still require
paging or Python extraction. Pretty JSON makes read_file's line pagination useful.

Paths include a content hash so concurrent requests for different projections cannot
overwrite each other's evidence. Every fetch restages, including host-cache hits: a
new sandbox must never inherit paths to files that existed only in its predecessor.
Assembly only captures the backend; all I/O occurs when the tool is invoked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any

from deep_life_sci.paths import TRIAL_FILES_DIR
from deep_life_sci.sources.ctgov import ClinicalTrialsError, ctgov_fetch

INLINE_RECORD_BYTES = 24_000


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ).encode("utf-8")


def _sections(record: dict) -> list[tuple[str, Any]]:
    """Lossless semantic slices; result-module metadata accompanies each child."""
    protocol = {k: v for k, v in record.items() if k != "posted_results"}
    sections = []
    for key, value in list(protocol.items()):
        if len(_json_bytes(value)) > INLINE_RECORD_BYTES // 2:
            sections.append((f"Protocol: {key}", {key: protocol.pop(key)}))
    sections.insert(0, ("Protocol and trial identity", protocol))
    for name, module in (record.get("posted_results") or {}).items():
        if name == "outcomeMeasuresModule":
            context = {k: v for k, v in module.items() if k != "outcomeMeasures"}
            for i, outcome in enumerate(module.get("outcomeMeasures") or []):
                label = f"Outcome {i + 1}: {outcome.get('type', '')} {outcome.get('title', '')}"
                sections.append((label, {**context, "outcomeMeasures": [outcome]}))
            if not module.get("outcomeMeasures"):
                sections.append((name, module))
        elif name == "adverseEventsModule":
            event_keys = {"seriousEvents", "otherEvents"}
            context = {k: v for k, v in module.items() if k not in event_keys}
            sections.append(("Adverse events: summary, groups and denominators", context))
            for key in sorted(event_keys):
                by_organ: dict[str, list] = defaultdict(list)
                for event in module.get(key) or []:
                    by_organ[event.get("organSystem") or "Unspecified"].append(event)
                for organ, events in by_organ.items():
                    sections.append((f"Adverse events: {key}, {organ}",
                                     {**context, key: events}))
        elif name == "baselineCharacteristicsModule":
            context = {k: v for k, v in module.items() if k != "measures"}
            sections.append(("Baseline: groups and denominators", context))
            for measure in module.get("measures") or []:
                sections.append((f"Baseline: {measure.get('title', '')}",
                                 {**context, "measures": [measure]}))
        else:
            sections.append((name, module))
    return sections


def _prepare(record: dict) -> tuple[dict, list[tuple[str, bytes]]]:
    compact = _json_bytes(record)
    if len(compact) <= INLINE_RECORD_BYTES:
        return record, []

    digest = hashlib.sha256(compact).hexdigest()[:20]
    # nct_id was validated by the source client; no source titles enter paths.
    root = f"{TRIAL_FILES_DIR}/{record['nct_id']}/{digest}"
    path = f"{root}/record.json"
    uploads = [(path, _json_bytes(record, pretty=True))]
    index = []
    for i, (label, value) in enumerate(_sections(record)):
        section_path = f"{root}/section-{i:04d}.json"
        data = _json_bytes({"nct_id": record["nct_id"], "section": label, "data": value},
                           pretty=True)
        uploads.append((section_path, data))
        index.append({"section": label, "path": section_path,
                      "bytes": len(data), "lines": data.count(b"\n") + 1})
    index_path = f"{root}/index.json"
    uploads.append((index_path, _json_bytes(index, pretty=True)))
    manifest = {k: record.get(k) for k in ("nct_id", "title", "status", "has_results", "url")}
    manifest.update(storage="file", path=path, index_path=index_path,
                    bytes=len(compact), section_count=len(index))
    return manifest, uploads


def make_trial_fetch(backend: Any):
    """Same PTC name/schema, with sandbox staging on the success path."""
    async def fetch(nct_ids: list[str], include: list[str] | None = None) -> dict:
        result = await ctgov_fetch.coroutine(nct_ids=nct_ids, include=include)
        records = {}
        for nct_id, record in result["records"].items():
            manifest, uploads = await asyncio.to_thread(_prepare, record)
            if uploads:
                # Bound upload batches; a trial can have hundreds of outcome sections.
                for start in range(0, len(uploads), 50):
                    batch = uploads[start:start + 50]
                    try:
                        responses = await backend.aupload_files(batch)
                    except OSError as exc:
                        raise ClinicalTrialsError(f"Could not stage {nct_id}: {exc}") from exc
                    if len(responses) != len(batch) or any(r.error for r in responses):
                        raise ClinicalTrialsError(
                            f"Could not stage all files for {nct_id}; retry ctgov_fetch"
                        )
            records[nct_id] = manifest
        return {**result, "records": records}

    return ctgov_fetch.model_copy(update={
        "coroutine": fetch,
        "description": ctgov_fetch.description + (
            "\nRecords over 24 KB are staged in the sandbox and replaced by a manifest "
            "with storage='file', path (complete JSON), index_path and section_count. "
            "Pass the manifest to trial-analyst; it reads the index and relevant sections. "
            "Small records stay inline. Files are under /workspace/retrieved/ctgov, "
            "outside deliverables. Cache hits also restage files."
        ),
    })
