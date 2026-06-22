# Talon — Agent Transition Document (2026-06-22)

Handoff for the next agent picking up Talon. Read `MEMORY.md` and the global
`~/.claude/CLAUDE.md` first for standing context and the user's writing rules;
this doc covers the live state and open threads as of this session.

---

## 1. Orientation

Talon is a local PyQt6 desktop AI assistant. Backend is KoboldCpp serving a
**Qwen3.5-9B hybrid reasoning model** (+ vision mmproj) on `localhost:5001`.
Memory is SQLite + ChromaDB. Capabilities are plugin "talents". There is a
voice path, a Signal remote, a scheduler, and a document-RAG / deep-search
stack.

Key files:
- `core/assistant.py` — orchestrator, `process_command`, routing, the agentic tool loop (`_run_tool_loop` / `_handle_tool_loop`)
- `core/llm_client.py` — LLM wrapper. `generate(think=...)` and `chat(reasoning_effort=...)`
- `core/mcp_client.py` — **new this session.** MCP client (`MCPManager`)
- `core/security.py` — input/output filters; `_API_KEY_RE` egress secret scan
- `core/conversation.py` — conversation + factual RAG + deep-search synthesis
- `talents/base.py` — `BaseTalent`, `to_tool_schema()`, optional `tool_parameters`
- `talents/job_search.py`, `talents/signal_remote.py`, `talents/email_talent.py`

## 2. Repo & deploy topology (important)

- **Dev / source of truth:** `C:\Users\zenra\PycharmProjects\talon-assistant`, on branch `main`.
- **Prod (what the user runs):** `C:\Users\zenra\OneDrive\Desktop\talon-assistant`, runs `main`, deploys via `git pull`.
- Each copy has its own gitignored `data/`, `config/settings.json`, etc. They diverge per copy.
- The Bash tool executes **on the user's machine**, so it can read/write the prod copy's files and reach KoboldCpp on 5001.
- **`tool-enabled` branch is merged into `main` and deleted.** Do not look for it. All tool-calling/MCP work lives on `main`.
- Convention: **commit AND push to `main` after each logical fix**, with a why-focused message ending in the `Co-Authored-By` trailer.
- A separate `main` worktree (`talon-main-review`) was removed this session; `main` lives in the primary dir now.

## 3. What shipped this session (all on `main`)

- **Native tool calling + hybrid router.** The model picks/chains tools itself via `/v1/chat/completions`. Gated behind `llm.tool_calling` (default off; GUI toggle in Settings → LLM, hot-swappable). The agentic loop only fires for genuinely multi-step commands (`_MULTI_STEP_RE`); simple single-intent commands stay on the fast keyword/LLM router. The old `planner` talent is the tool_calling=off fallback, kept on purpose.
- **Per-call thinking control.** `generate(think=True/False)` toggles the empty-think prefill on the raw path; `chat(reasoning_effort=...)` sets none/minimal/low/medium/high on the chat path. Default suppressed everywhere. Only deep-search synthesis opts into `think=True`.
- **MCP client support** (`core/mcp_client.py`). Connects to external MCP servers, folds their tools into the loop namespaced `mcp__<server>__<tool>`. Config in `config/mcp_servers.json` (gitignored; Claude-Desktop format). asyncio loop on a daemon thread, one worker coroutine per server, clean shutdown via atexit (fixes a Windows segfault). Fail-soft: no config / missing SDK / dead server just means no MCP tools. Gated on `tool_calling`.
- **Router reach for MCP** (`95c3e4b`): a command matching no fast-path talent drops into the loop when MCP is enabled, so MCP tools are reachable for single commands too.
- **Email** (`talents/email_talent.py`): typed tool params (action/to/subject/body/attachment_path/query/folder) so the recipient stops getting dropped in chains; explicit tool-call sends go immediately (no compose-window draft).
- **Security egress filter** (`18887b9`): `_API_KEY_RE` was a generic `{20,}[-_]{8,}` catch-all that flagged hash-named files. Replaced with real provider signatures (OpenAI/Anthropic/Google/GitHub/GitLab/Slack/AWS/JWT/PEM).
- **Logging fix** (`8eac7f4`/`fa9f15e`): the console log handler crashed on non-ASCII on Windows (cp1252). Reconfigured stdout `errors="replace"`.
- **claude -p error logging** (`299ce60`): `job_search` now logs stdout AND stderr on a `claude -p` failure (it was logging only stderr, which the CLI leaves empty, hiding the real error).

## 4. Environment / config facts

- **KoboldCpp 1.114.1** on `localhost:5001`, `llm_server.mode=external` (Talon connects, doesn't manage it). `reasoning_effort` is honored; `chat_template_kwargs` need `--jinja` (not enabled, not needed).
- Generation ~25 tok/s; each tool-loop chat turn ~4.5s. **Tool-loop latency is round-trip count × model speed, not thinking** (routing turns emit ~57 tokens with no reasoning). This is why routing is hybrid.
- **`pip install mcp`** done in dev; prod needs it too for MCP. Node/npx available; `uv`/`uvx` not.
- Prod has **`tool_calling` ON**; MCP filesystem server scoped to `C:\Users\zenra\Downloads`.
- **Credentials live in the Windows keyring** (Credential Manager), per-user, shared across copies. Not in files.
- ChromaDB embedding model is **bge-base (768-dim)**; `migrate_embeddings.py` re-embeds after a model change (already run on the dev copy this session).
- Machine date during this session: **2026-06-22**.

## 5. Open threads / next steps

1. **Rescore 99 job_search rows (likely in progress).** Today's (2026-06-22) 99 jobs are all `fit_score=0` — the fit batch failed while claude was briefly down (transient). Resolution: in Talon run **`run fit analysis`** (a.k.a. "score my jobs"); `_handle_score_existing` targets exactly `archived=0 AND status='new' AND fit_score=0`, which is precisely those 99. The user was about to trigger it. **Verify** afterward that the scores become a real spread, not all-zero (prod DB: `C:\Users\zenra\OneDrive\Desktop\talon-assistant\data\job_tracker.db`).
2. **Archived cleanup: nothing to do.** 0 archived rows are older than 25 days by either `date_found` or `archived_at` (oldest archived found 2026-06-04). Auto-archive is recent; old listings were never archived.
3. **Spawned task `task_8897e1dc`** — "Sweep talents for swallowed subprocess errors." Many talents log only stderr (or nothing) on subprocess failure; CLIs like `claude -p` and `signal-cli` write errors to stdout, so failures look blank. Already fixed: `job_search` (both `claude -p` calls) and the `signal_remote` issue is documented. Sweep the rest.
4. **#27 typed tool parameters** (in_progress): framework + email migrated. Migrate other talents **reactively** — only chain-target talents that carry cross-clause data (signal_remote, file/notes saving) and only when a real chain is seen dropping data. Single-intent talents (hue/weather/etc.) do NOT need it; they're fast-pathed and their data is self-contained.
5. **`file_organizer` is disabled** (user turned it off). It had a bug: "list the files in `<path>`" fell to its generic help text instead of listing. It also overlaps with the MCP filesystem server. Fix the list-classification bug only if re-enabling.
6. **`signal_remote` logging gap:** only logs signal-cli stderr on non-zero exit, so per-envelope exceptions (which exit 0) are swallowed. Part of the subprocess sweep above.

## 6. Gotchas & lessons (save the next agent the pain)

- **The hybrid 9B reasons unpredictably.** Most session bugs early on traced to its `<think>` output. Thinking is suppressed by default; turn it on only where it earns the latency (deep-search synthesis). Reasoning does NOT help routing/tool selection (those turns don't reason).
- **Talents swallow subprocess errors** — log stdout too, not just stderr. This bit Signal and job_search.
- **Signal was fixed by updating signal-cli**, not by code: 0.14.0 NPE'd on `getServerGuid` (Signal server change); 0.14.5 fixed it. signal_remote requires a **linked** signal-cli (Note-to-Self only), prefix `Talon: `.
- **MCP servers run with the user's credentials** and see whatever you pass them. Scope filesystem access tightly. The user is a CISO; lead with the security framing.
- **Verify with data, don't guess.** The user explicitly values this. Own mistakes plainly; don't fixate on one hypothesis; don't blame-narrate. Don't claim something works without checking.
- **Writing rules (from CLAUDE.md):** no em dashes, no tricolons/parallel triplets, plain language. Applies to any drafted text (LinkedIn posts, emails, docs).

## 7. User

Aaron Lafferty. Target roles: CISO / VP Security / Director of Security (DFW or remote, $200k+). Builds Talon as production AI tooling and a hands-on way to stay current. Active job hunt (the job_search/job_tracker talents are real, in-use). Preferences: concise responses; commit + push after each fix; own errors and move on.
