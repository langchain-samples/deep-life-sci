**Concept: a Deep Agents implementation for life sciences clients with skills and tooling to act as a research assistant for biologists**

Principle: This is a demo. Keep everything as simple as possible.

**Necessary tools**

- Boolean PubMed search, returning list of PMIDs with paper names, lead author, and publication year. This is `esearch` then `esummary`—two calls, and esummary carries all the metadata we need. Can dump to filesystem as JSON or CSV if more than a certain number of results.
- Retrieve PubMed abstracts by PMID, batched via `efetch`. Caches locally on retrieval so nothing is fetched twice.
- Ask question of an abstract using a subagent. The orchestrating code batch-fetches the abstracts first and passes the text into each subagent's prompt, so subagents do no I/O of their own. Must be able to fan out a large number of subagents.

**Constraint driving the above:** NCBI rate-limits E-utilities to 3 requests/sec (10 with an API key). That's why abstracts are batch-fetched up front instead of pulled per-subagent—fifty subagents each hitting the API would just collect 429s.

The API has enough sharp edges to be worth its own writeup—silent query mangling, malformed PMIDs returning the wrong paper, hard batch caps that fail as HTTP 200. See `pubmed_api_notes/` (gitignored) for per-service notes; the tools need to account for those, but they don't belong in this doc.

To start with, we’ll just do abstract, since that’s always just text. Papers, with their associated multimodal and multi-file-type data will come later.

Rather than exposing tools to the model directly, we’ll have it compose them in code via LangChain’s interpreter (https://docs.langchain.com/oss/python/deepagents/interpreters). The interpreter runs JavaScript in QuickJS with no network, filesystem, or shell access, so the PubMed calls themselves still have to be real Python tools—they reach the interpreter through programmatic tool calling (`ptc=[...]`), which exposes them as async functions on a `tools` global (`pubmed_search` → `tools.pubmedSearch(...)`). So: two thin Python tools, and each gets a segment in the prompt with a reference JS snippet showing how to call and compose it.

That covers orchestration but not quantitative work—reading a corpus and then actually computing over it (group by year, tally model organisms, run a statistic, draw a plot) needs real Python, which QuickJS can’t give us. So there are **two code surfaces**, and the prompt has to draw the line between them or the model reaches for the wrong one:

- **`eval` (QuickJS)** — orchestration. Search, fetch, fan out, collect. Everything reaches outside through `tools.*`.
- **`execute` (sandbox shell)** — Python 3 with numpy/pandas/scipy/statsmodels/scikit-survival/scikit-learn/matplotlib plus biopython and rdkit, in an isolated Linux container. This comes from swapping the agent’s *backend* to a LangSmith sandbox, which adds `execute` alongside the filesystem tools.

Both go in the PTC allowlist, so one `eval` call can run search → fetch → write → compute → collect. The filesystem tools are in there too and operate on the sandbox’s filesystem, the same one `execute` sees.

The sandbox starts empty and is deleted when the run ends, so **PubMed data doesn’t appear in it by itself—the agent writes it there.** That’s cheap: PTC tool output is marshalled straight into the JS heap and never enters model context, so piping `fetch_abstracts` records into `tools.writeFile` costs no tokens. The on-disk abstract cache stays host-side and stays invisible to the agent; it exists to serve the rate limit, not the agent. Booting the sandbox is a ~95s `pip install` unless you bake the libraries into a snapshot first (`build_snapshot.py`), which takes it to ~1-3s.

Watch the interpreter defaults—`timeout` is 5s, which a fan-out will blow through immediately. `max_result_chars` (4000) truncates what comes back to the model.

Subagents via LangChain’s dynamic subagents (https://docs.langchain.com/oss/python/deepagents/dynamic-subagents), dispatched from interpreter code with `task()`. Give them a `response_schema` so the synthesis step gets structured objects instead of prose to parse. Cheap model on the leaves (Haiku for abstract Q&A, Sonnet at the root) is worth showing—it’s the cost story for a client running thousands of abstracts.

Both the interpreter and dynamic subagents are beta; flag that to clients.

Web search beyond PubMed will come later.