# frontend/

The chat UI: a Next app that began as agent-chat-ui and is now this repo's own code.
`UPSTREAM.md` says where it came from; the two commits that added this directory are
upstream as published and then this project's changes, so their diff is the whole of what
we changed. Upstream is not tracked. Apply a later upstream fix by hand, and prefer
changing a component outright to wrapping it: nothing re-applies edits here any more.

Run from this directory:

```bash
npx pnpm@10.5.1 lint             # eslint; warnings only today
npx pnpm@10.5.1 exec tsc --noEmit
npx pnpm@10.5.1 build
```

`uv run scripts/dev.py` from the repo root starts this with `next dev` beside the agent
server. It must not install anything: `scripts/setup.py` owns `pnpm install` here.

## Two builds

`next.config.mjs` makes two apps from one source tree:

- **`next dev`** (local): a Node server with the `/ui/*` rewrite and the API passthrough
  (`src/app/api/[..._path]/route.dev.ts`, picked up only through the dev `pageExtensions`).
  No sign-in, against a local server that has no auth.
- **`DEEP_LIFE_SCI_STATIC=1 pnpm build`** (the deployment image; `scripts/deploy.py`): plain
  files under `/app`, served by the agent server itself (`deep_life_sci/webapp.py`), so the
  page, the API and `/ui/*` share one origin. No rewrite, no route handler, no Node at run
  time. Try it locally by building it and running a server with the deploy config minus
  `dockerfile_lines` and `allow_langsmith_api_keys`; `paths.UI_DIR` finds `frontend/out`.
  The second matters: without the platform's LangSmith auth behind it, that option lets
  requests with no token through.

## Sign-in

`src/lib/auth.ts` and `src/providers/Auth.tsx`. Only the static build signs in: it fetches
`/app/config.json`, which the server fills from the deployment's `OIDC_*` settings, and signs
in through the provider with the authorization-code flow (oidc-client-ts). The provider
sends the browser back to `/app/auth/callback`.

- **Every request to the agent server goes through `createClient` (`src/providers/client.ts`)
  or `authFetch`**, which add the token per request. A new call made with plain `fetch` or a
  `Client` of its own is unauthenticated and gets 401 once deployed.
- **A run outlives the page.** Submits use `onDisconnect: "continue"`, and the stream
  rejoins on load (`reconnectOnMount`, which keeps only the run id in `sessionStorage`), so
  a reload, a sign-in redirect or a dropped connection never ends a run. Cancel ends it on
  the server through `useCancelRun`, which cancels the thread's active runs. Never go back
  to `onDisconnect: "cancel"`: it cancels on every disconnect, not just on Cancel.
- **Tokens stay in memory** (`InMemoryWebStorage`). Don't move them to `localStorage` or
  `sessionStorage`; a reload signs in again, silently while the provider session lasts.
- The callback navigates client-side (`router.replace`), never by reload: a reload would
  drop the token it just received.

## Contracts with the agent server

- **`/ui/*` must be served from this app's own origin.** The server answers `/ui/<graph>`
  with a script tag whose `src` is host-relative, so the browser resolves it against this
  page. `next.config.mjs` rewrites `/ui/*` to the server; without it the artifact
  components render as empty divs while the run is otherwise fine. The rewrite follows
  `LANGGRAPH_API_URL`, the same variable the API passthrough in `src/app/api` reads, which
  is how `dev.py --remote` points both at a deployment. Keep the two on one variable.
- **The browser never holds a LangSmith API key.** A local server needs none, and
  `dev.py --remote` adds one server-side in the API passthrough. Upstream's key field,
  which kept the key in `localStorage`, is removed; don't bring it back.
- **The thread list requests a metadata projection** (`src/providers/Thread.tsx`), never
  full thread values: QuickJS snapshots make each thread megabytes.
- **The upload allowlist tracks `UPLOAD_KINDS`** in `deep_life_sci/middleware/uploads.py`.
  `UPLOAD_SUFFIXES` (`src/lib/multimodal-utils.ts`) and `isSupportedUpload`
  (`src/hooks/use-file-upload.tsx`) decide what the composer accepts, and
  `tests/test_invariants.py` checks the suffix list against the server's. A type the UI
  accepts that the graph has no reader for reaches the model as context and nothing else.
  - Test the extension first and the MIME type second: Windows with Excel installed reports
    a `.csv` as `application/vnd.ms-excel`, and some browsers report `""`.
  - Keep `accept="*/*"` on the composer input, so an `.xls` reaches the toast asking the
    user to re-save it instead of being greyed out of the picker.
  - Every accepted upload is sent as a `type: "file"` block, images included. Upstream sent
    images as `type: "image"` to put them in model context; here they are transport to the
    sandbox.
