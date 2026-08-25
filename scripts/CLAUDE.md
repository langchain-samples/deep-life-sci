# scripts/

`setup.py` and `dev.py` are the front door. Their docstrings cover what each step does and
why they are Python rather than shell; this file is the contract for the part with no code to
read it off — the chat UI clone.

## The chat UI is a patched clone

The UI is `langchain-ai/agent-chat-ui` (`UI_REPO`) cloned to `.chat-ui/`, **inside the repo**
so nothing needs a writable directory beside it, and **gitignored**. Two consequences shape
everything below: an edit made in the clone is a lost edit, and an un-patched clone sits there
with the feature simply absent and nothing saying so.

`AGENT_CHAT_UI=<path>` points at a checkout elsewhere.

Setup also runs `npm ci` in `ui/`, whose components the *graph server* bundles. Without those
deps the bundler logs `Could not resolve "xlsx"`, answers `/ui/<graph>/entrypoint.js` with a
200 anyway, and the component is silently absent.

`.dockerignore` must keep excluding `.chat-ui` — `langgraph build`'s context is the repo root,
so an unignored clone ships a dev-only Next app in the deploy image. Its comments list what
must *not* be excluded.

## Two halves, and which one a change belongs in

**Patches** (`apply_patches`) are for upstream-shaped edits: small, anchored, plausibly things
upstream would take. **The overlay** (`chat-ui-overlay/`, copied by `ensure_overlay`) is for
product surface: no upstream counterpart, never converging, and the worst possible fit for a
search-and-replace living inside a Python string. A new component goes in the overlay, and the
patch that mounts it is the only edit to upstream's files.

Fork upstream if an overlay component ever needs to *replace* `index.tsx` wholesale, or if the
app's identity has to change; `UI_REPO` is then the only line to move.

## Rules for a new patch

Every patch function's docstring says what it is for. What is not in any one of them:

- **Anchor on an exact upstream string and print the snippet rather than guess past a moved
  anchor.** A moved anchor must not clobber an upstream fix. `patch_next_config` also bails if
  upstream ever grows a `rewrites` key, since a second one would silently shadow the first.
- **Every patch needs an entry in `PATCH_MARKS`.** The mark is what the patch's own early-out
  tests *and* what `dev.py` re-checks on every launch to re-apply a stale clone. A patch with
  no mark is a patch that silently stops being applied. Mark `present=False` for a patch that
  works by removing something.
- **Prefer a new patch over another edit inside an existing one**, so a clone that already has
  the old patch still picks the new one up. That is why the rewrite and `devIndicators` are
  separate, and why the heading layout and the 🧪 are not folded into the rename.
- **Accept the prior baseline when a patch's shape changes** (`patch_uploads` takes both
  upstream's and the earlier allowlist), so an existing clone upgrades in place.
- **Changing `APP_NAME` means appending the old value to `PRIOR_APP_NAMES`.** The rename is a
  search-and-replace on upstream's own product name, so a clone patched under an earlier name
  has no `Agent Chat` left to match — and the clone being gitignored makes that list the only
  record the rename has.
- **The upload allowlist tests `isSupportedUpload`/`isSpreadsheetUpload`, never the MIME
  list.** Windows with Excel installed reports a `.csv` as `application/vnd.ms-excel` and some
  browsers report `""`: extension first, MIME second. `accept="*/*"` stays on the composer
  input so a `.xls` reaches the toast telling the user to re-save it rather than being greyed
  out of the picker. Images and PDFs stay refused — an attachment the graph does not intercept
  is model context and nothing else.

The `/ui/:path*` rewrite is the one patch that is load-bearing rather than cosmetic: without
it the artifact components never render (see the same-origin invariant in the root
`CLAUDE.md`).
