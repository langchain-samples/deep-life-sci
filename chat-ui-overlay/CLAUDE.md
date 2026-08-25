# chat-ui-overlay/

Our own chat UI components, tracked in git and reviewable in a diff. This directory **mirrors
the clone's `src/`** — a file's path here is where it lands — and `setup.py:ensure_overlay`
copies it in on every setup and launch.

**The copy is one way.** `.chat-ui/` is a gitignored clone, so an edit made over there is a
lost edit whatever happens; losing it on the next launch beats keeping it and diverging
silently. Edit the file here.

Only `.ts`/`.tsx` files are copied, and only when the content differs, so `next dev` is not
handed a changed mtime and a rebuild on every launch of an unchanged app.

A component here still needs an anchored patch in `setup.py` to mount it — see
`scripts/CLAUDE.md` for which changes belong in the overlay and which in a patch.
