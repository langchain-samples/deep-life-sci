# scripts/

`setup.py` and `dev.py` are the front door, and `deploy.py` puts the agent server on
LangSmith. Their docstrings cover what each step does and why they are Python rather than
shell; this file is what no single one of them says.

## Setup owns installs; the launcher never does

`setup.py` installs everything: the virtualenv, the sandbox snapshot, a private Node when
the machine has none new enough, `frontend/`'s dependencies (`pnpm install
--frozen-lockfile` at its `packageManager` pin), and the artifact components' dependencies
(`npm ci` at the repo root, where `ui/` is an npm workspace). `dev.py` checks for those and
names the fix rather than installing anything itself.

The artifact components are bundled by the *graph server*, not by the frontend. Check the
bundler output when one does not render: a missing dependency shows up as `Could not
resolve "xlsx"` while `/ui/<graph>/entrypoint.js` still answers 200, so the component is
simply absent with nothing else saying so.

The chat UI's own rules are in `frontend/CLAUDE.md`. `.dockerignore` keeps `frontend/`
out of the deploy image; its comments list what must *not* be excluded.

## Platforms

`.github/workflows/install-smoke.yml` runs setup and `dev.py` on native Windows and on a
bare Ubuntu 22.04 with no C compiler. Those are the two setups testers have broken on:

- `.python-version` pins 3.12, the deployment's version. Without a pin, a machine whose
  own Python is too old gets the newest one from uv, and `bsdiff4` (via langchain-quickjs)
  has no 3.14 wheels, so setup dies compiling it. Before raising the pin, check that
  every package in `uv.lock` has wheels for the new version.
- On Windows `dev.py` runs `langgraph dev --no-reload`; the comment at the call says why.

## Deploying

`deploy.py` wraps `langgraph deploy`; its docstring says why. It owns `.env.deploy` the way
`setup.py` owns `.env`, and rewrites `langgraph.deploy.json` from `langgraph.json` when the
two differ. It never edits `.env`.
