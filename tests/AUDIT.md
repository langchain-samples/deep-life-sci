# Test suite audit — 2026-09-15

The original 862 tests all passed, but the suite was not comprehensive. It had strong
coverage of PubMed parsing, cache semantics, trial projections, and many upload parsers.
Its largest omissions were the operations that connect those helpers: sandbox recovery,
artifact publication, upload persistence/restoration, PMC retrieval and staging, model
construction, and agent execution. Passing helper tests did not establish those contracts.

The revised suite collects 967 tests, including 105 additional cases. No tests are skipped
or marked xfail. The changes include production fixes because the new regression tests
exposed incorrect behavior; expected results were not changed to accept those bugs.

## Reproduced and fixed

| Issue | Observed failure before the fix | Regression coverage |
|---|---|---|
| Artifact retries | A download exception, short download batch, or failed UI publication permanently suppressed subsequent publication within the run. | `test_artifact_lifecycle.py`: only remember a fingerprint after successful publication; retain retry eligibility on failures. |
| Replacement uploads | A different attachment with the same filename and byte count left the old sandbox contents in place. | `test_upload_lifecycle.py`: fresh attachments overwrite existing bytes and refresh the manifest. |
| S3 listing XML | XML entities in object keys and continuation tokens were used literally. Missing continuation tokens silently truncated a listing. | `test_pmc_transport.py`: parse XML with namespaces and entity decoding; reject malformed/incomplete pagination. |
| Sparse workbook headers | A header in cell C1 was reported as column B when B1 was absent from the XML. | `test_workbook_probe.py`: actual OOXML ZIP fixtures preserve empty column positions. |
| Empty environment overrides | Explicit empty effort settings were overwritten by dotenv in both CLI and eval entry points. | `test_cli.py`: empty and nonempty overrides survive a simulated dotenv load. |
| PDF decryption status | The probe ignored a zero return from `PdfReader.decrypt`, treating failed decryption as success until a later read failed. | `test_pdf_probe.py`: both failure statuses and exceptions produce the password-protected note. |

## Test quality changes

- Disable dotenv loading before collection. The old `pytest_configure` explanation was
  incorrect: that hook runs before collection, so it cannot undo collection-time dotenv
  imports. Reassert tracing and environment isolation for each test.
- Redirect all source cache readers, writers, evaluator lookups, and sweep roots to
  temporary directories by default. Previously this depended on tests remembering to
  request individual fixtures.
- Block socket connection attempts during tests. A teardown assertion catches attempts
  even when production code catches the immediate exception. A subprocess regression
  verifies this enforcement. This is an accidental-I/O guard, not an OS network sandbox.
- Replace scheduler-sensitive throttle and cadence checks with controlled clocks, retaining
  concurrency and lock contention in the throttle test.
- Replace keyword-only prompt assertions with checks for the actual instruction; verify
  complete environment variable sets rather than their length. Compare prompts across
  different dates instead of comparing a call to itself.
- Rename and strengthen the seed rubric test. It claimed to validate literal YAML block
  formatting but only tested whether any rubric was nonempty. The real scoring contract is
  that every example has a nonempty text rubric; legitimate plain and folded scalars remain
  supported.
- Exercise complete POST payloads and batched trial identifiers, rather than merely the
  HTTP method or number of requests.

## Runtime coverage added

- Sandbox connection retry bounds, delay caps, cancellation, re-acquisition, concurrent
  rebind deduplication, stale-client cleanup, install guards, and session cleanup.
- Graph reads avoiding acquisition, run acquisition off the event loop, existing/stopped
  sandbox reuse, and expired sandbox replacement.
- Artifact encoding, size caps, checkpoint deduplication, thread separation, regenerated
  files, publication failures, and preservation of tool results/errors.
- Upload message replacement through the real message reducer; durable storage through
  `InMemoryStore`; subsequent turns, recycled containers, thread separation, failed store
  writes, missing probe results, and real `ModelRequest` manifest injection.
- PMC version resolution, positive and negative caching, triage payload omission, full-text
  section selection/fallback, figures, supplementary uploads, and S3 failures/pagination.
- Both model provider constructors, role timeouts, effort handling, Responses API use,
  root streaming, and search-only tool binding.
- Agent assembly plus an actual compiled graph executing QuickJS → PubMed HTTP mock →
  artifact middleware → final scripted model response. QuickJS, tool wrappers, and graph
  execution are real; no model or remote container is called.
- CLI stream filtering, runner trajectory/artifact extraction, and judge response parsing.
- Workbook ZIP parsing and PDF orchestration/sidecar contracts.

## Validation and limits

- Full suite: **967 passed** on Python 3.13 and **967 passed** on Python 3.12.13.
  The 3.12 run used a separate environment installed from `uv.lock` with
  `uv sync --python 3.12 --locked --no-default-groups --group test`.
- Full suite also run in reverse collection order to check shared-state assumptions.
- Repository-wide `ruff check .` and `git diff --check` pass.
- Two visible upstream warnings remain: LangChain QuickJS beta API and LangSmith's use of
  deprecated `ast.Str`. They are not suppressed.
- Each production fix above was preceded by an observed failing regression test.

This is substantially broader unit and local integration coverage, not evidence of complete
branch coverage or live deployment correctness. A lightweight call trace was used to locate
untested functions; it is not a coverage percentage (worker-thread calls are not captured).

Remaining integration work requires the actual sandbox image and services: real PDF/Pillow/
GenBank decoding, scientific package provisioning and warmup, real SDK connection recovery,
provider response schemas, deployment/checkpoint concurrency, and browser artifact rendering.
PDF tests substitute the reader/decoder boundaries and must not be presented as validating
file decoding. Setup scripts, frontend patches, performance instrumentation, and the remote
evaluation/sync harness also do not have comprehensive tests here. Python 3.12 and 3.13 have now both been exercised locally; a CI matrix would keep
that compatibility checked on future changes.

No project dependencies or lockfile versions were changed. The Python 3.12 environment
was installed through Socket Firewall 1.15.1, using its published macOS ARM64 release
asset after the safety skill's older download URLs returned 404.
