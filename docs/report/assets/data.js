window.BENCHMARK_DATA = {
  meta: {
    title: "Hermes memory provider benchmark",
    completed: "August 2026",
    personas: 30,
    questionsPerWave: 3750,
    dialogueTurns: 71060,
    conflictCounts: { dynamic: 2946, static: 360, conditional: 444 },
    judge: "Gemma 4 12B",
    noiseBand: "±0.02–0.03 macro score"
  },
  providers: [
    {
      id: "honcho",
      name: "Honcho",
      short: "Honcho",
      tag: "v5ftc",
      role: "featured",
      macro: 0.477,
      dynamic: 0.643,
      static: 0.181,
      conditional: 0.606,
      seh3: 0.808,
      micro: 0.595,
      outcomes: { correct: 2251, partial: 501, blank: 726, incorrect: 272 },
      tokens: { cached: 5528.9, uncached: 7588.6, outputTotal: 42546584, caveat: "Clean run; all 30 personas exited successfully." },
      tested: "Honcho 3.0.9 with honcho-ai 2.2.0. Hybrid recall combined a session summary, user and assistant profiles, peer cards, and a model-generated reconciliation called Dialectic. Dream consolidation was forced after each session.",
      verdict: "The best overall and changing-fact result. A FastAPI server, a deriver worker (the extraction worker), and PostgreSQL/pgvector: three components that each need monitoring. The deriver can produce repetition-loop memories and silently fail specialist jobs; Dream consolidation failures appear only in the deriver log. Derived context (profiles, peer cards, Dialectic) can contradict itself, and a small answer model may follow the wrong section.",
      process: "The database migrations assumed 1,536-dimensional vectors, so the vendor's configuration script had to resize the columns and rebuild both vector-search indexes (HNSW) for the shared 768-dimensional embedder. The deriver also stored repetition loops until its output cap was reduced. Hybrid recall remained plugin-shaped: a large assembled context block rather than a ranked top-five list.",
      timeout: "The SDK retries timeout and network failures even for message-adding write requests, and that call has no idempotency key. If the server commits before the response is lost, a retry could add the messages twice; this is a code-path risk, not an observed duplicate. A successful Dream request also did not prove its specialist jobs succeeded because failures appeared only in the deriver log.",
      caveat: "A deployment runs an API server, a deriver worker, PostgreSQL with pgvector, and an embedder endpoint. The Dream consolidation scheduler waits for idle time, so a busy assistant may never trigger it without explicit scheduling. The deriver can produce repetition-loop observations on small models (23 to 33% of observations in early tests hit the token cap with a single sentence repeated hundreds of times; a presence penalty and output cap reduce this to about 5%). Dream failures report success and the error counter stays at zero; actual failures appear only in the deriver log. The plugin ships with context truncation off, so a user with non-trivial history gets roughly 25,000 tokens injected on every turn. A dimension mismatch between the default schema (1,536) and a smaller embedder requires running the vendor's configuration script before the deriver or API will start.",
      repo: "https://github.com/plastic-labs/honcho"
    },
    {
      id: "mem0",
      name: "mem0",
      short: "mem0",
      tag: "v5ftc",
      role: "featured",
      macro: 0.392,
      dynamic: 0.250,
      static: 0.090,
      conditional: 0.836,
      seh3: 0.672,
      micro: 0.304,
      outcomes: { correct: 1605, partial: 506, blank: 922, incorrect: 717 },
      tokens: { cached: 7489.0, uncached: 1829.8, outputTotal: 17148227, caveat: "Token window is inflated by about 1% after one persona restart." },
      tested: "mem0ai 2.0.14 through its self-hosted Python interface, with Qdrant 1.18.3 as the vector database.",
      verdict: "The simplest integration stack among the shortlisted providers: a Python SDK, a Qdrant vector database, and the shared model endpoints. Best conditional memory (0.836), second-best overall. The main deployment concern is that version 2.0.14 no longer runs the older update/delete reconciliation pass; new facts are added alongside old ones, so stale information stays in the store and the answer model must reconcile competing facts.",
      process: "The 2.x API changed entity scoping, search arguments, and batch embedding behavior. The adapter had to omit an unsupported dimensions parameter on both single and batched embedding calls, bake the sparse-search models into the image, and remove an inherited OpenRouter key that otherwise redirected internal calls. Version 2.0.14 is ADD-only: it extracts new facts but no longer runs the older ADD/UPDATE/DELETE/NONE reconciliation pass.",
      timeout: "A failed extraction now raises instead of being quietly logged, so the adapter records the failed add and continues rather than killing the whole persona. The Python client also leaves worker threads alive after the work is finished; the harness flushes results and exits the process explicitly. No recurring HTTP-timeout failure survived into the final run.",
      caveat: "A conventional deployment: pip install the SDK, run Qdrant, and point extraction at a model endpoint. The 2.x release is a major API change from 1.x (entity scoping, search arguments, and batch embedding calls all changed). Old and new facts can remain together for retrieval, leaving the answer model to resolve them. Extraction can produce self-contradictory memories (observed: 'the birth year 1936 was a typo since the actual birth year is 1936'). The 2.x docstring still describes the 0.1.x two-phase ADD/UPDATE/DELETE behavior, but the code runs ADD only.",
      repo: "https://github.com/mem0ai/mem0"
    },
    {
      id: "supermemory",
      name: "Supermemory",
      short: "Supermemory",
      tag: "v5ftc",
      role: "featured",
      macro: 0.288,
      dynamic: 0.144,
      static: 0.026,
      conditional: 0.694,
      seh3: 0.537,
      micro: 0.197,
      outcomes: { correct: 1075, partial: 585, blank: 1463, incorrect: 627 },
      tokens: { cached: 1796.0, uncached: 590.0, outputTotal: 18328682, caveat: "Token window is inflated by roughly 10–20% after nine failed attempts." },
      tested: "The self-hosted 0.0.5 server with its extraction model, ranked search results, and the separate profile block used by the Hermes integration. This was a deliberate old-version pin: 0.0.6 shipped without the Rivet workflow module and 0.0.7-rc.2 also left submitted documents stuck in the queue.",
      verdict: "Good conditional recall (0.694), but two consecutive self-hosted releases after the tested version (0.0.5) shipped broken: 0.0.6 omitted the Rivet workflow module needed for memory extraction, and 0.0.7-rc.2 left submitted documents stuck in the queue. The ranked-memory list was empty for 904 of 3,750 questions. The server dispatcher can die while HTTP stays healthy, so health checks alone do not prove the system is working.",
      process: "Ingest is asynchronous, so every accepted document had to reach done before its questions were asked. The memory agent also required Qwen's tool-call parser, which converts model-written tool syntax into actual function calls; without it the server finalized chunks while silently falling back from extracted memories. Under the simulated clock, node-cron tried to replay years of missed schedules and allocated an estimated 73 GB, so the server was respawned after every session on the same data directory.",
      timeout: "Transport failures and read timeouts were retried, but HTTP 5xx responses were not. A timed-out document POST may already have been accepted: re-submitting it could duplicate memory. When the server died during drain, the harness refused to resubmit, marked the persona invalid, and reran it from a clean checkpoint. The 0.0.5 dispatcher could also keep returning HTTP 200 while leaving every new document queued.",
      caveat: "A single native server binary (Bun-based) with an internal extraction model and embedded storage. Simple to start, but the self-hosted release history is unstable: the tested 0.0.5 is the newest Linux build that completed ingestion during this campaign. Ingest is asynchronous, so a deployer must poll for completion before the memories become searchable. The server ships only via GitHub Releases, and the api.github.com lookup is rate-limited; pin a version rather than pulling latest. Nine failed attempts during the benchmark campaign made workload roughly 10 to 20% high.",
      repo: "https://github.com/supermemoryai/supermemory"
    },
    {
      id: "retaindb",
      name: "RetainDB server",
      short: "RetainDB",
      tag: "v5ftc",
      role: "featured",
      macro: 0.270,
      dynamic: 0.279,
      static: 0.035,
      conditional: 0.495,
      seh3: 0.544,
      micro: 0.281,
      outcomes: { correct: 1147, partial: 792, blank: 1323, incorrect: 488 },
      tokens: { cached: 468.3, uncached: 3153.8, outputTotal: 52781967, caveat: "No restarts; about 0.6% foreign traffic remains in the token window." },
      tested: "RetainDB server 1.0.0 at a pinned upstream revision, with Hermes-shaped ingest, a vendor option that promotes user-specific session memories, and a documented patch layer required to build and run that revision.",
      verdict: "No clear category lead. Seven upstream defects were found during integration: the Prisma schema covered only 27 of 67 tables, first ingest failed on a fresh deploy, temporal dates returned null despite correct database values, the semantic cache had no tenant isolation, and the best-confidence search results were discarded by an early-exit path. Five of these carry patches in the test image.",
      process: "Seven upstream defects were found and five are patched in the test image: Prisma/schema compatibility, seed rows, date propagation, cache isolation, and malformed high-confidence search results. Approximate vector indexes (IVFFlat) created on empty tables were removed in favor of exact search at this scale. The quality profile also needed its post-vector budget raised from 120 to 2,000 ms; otherwise 117 of 122 smoke questions silently skipped graph and temporal stages.",
      timeout: "The generic write helper retries server errors, connection errors, and timeouts without an idempotency key. A response lost after commit could therefore replay a whole-session ingest; this is a source-code risk, not an observed duplicate. Separately, lifecycle waits now use the scheduler's own completion marker so a slow summary cannot be mistaken for a finished session.",
      caveat: "A Node.js server with Prisma ORM and PostgreSQL. The shipped schema covers 27 of 67 tables the migrations create, requiring a full introspected schema swap before the server starts. First ingest fails on a fresh deploy because auto-created project references point at empty seed tables. The search budget defaults to 120 ms, which is too short for a remote embedder and silently skips the graph and temporal scoring stages. The semantic cache keys only on query-embedding similarity with no tenant isolation, so one user's cached results can serve another's query. The SIGTERM handler overrides Node's default, so containers hang after kill. The local edition (npm @retaindb/local) is a separate product; its search is O(n^2) and reached about 245 seconds per query at 2,897 memories.",
      repo: "https://github.com/RetainDB/RetainDB"
    },
    {
      id: "openviking",
      name: "OpenViking",
      short: "OpenViking",
      tag: "v5ftovk",
      role: "featured",
      macro: 0.132,
      dynamic: 0.143,
      static: 0.067,
      conditional: 0.187,
      seh3: 0.321,
      micro: 0.141,
      outcomes: { correct: 614, partial: 356, blank: 2516, incorrect: 264 },
      tokens: { cached: 364.0, uncached: 1037.7, outputTotal: 19345376, caveat: "Token window is inflated by roughly 15–20% after nine failed attempts." },
      tested: "OpenViking 0.4.12 through Hermes' pre-response memory lookup: a session-start profile block followed by model-planned search over its memory tree.",
      verdict: "A single self-contained server with embedded storage and no external database. Low wrong-answer rates came primarily from abstention: 67.1% of all answers were blank. The search planner can decide that no search is needed and return only the profile block, and the plugin does not fall back on an empty result. Nine failed attempts were needed to complete 30 personas, caused by workspace I/O failures that left commit tasks permanently pending.",
      process: "The planner may decide that no search is needed and return only the session-start block; Hermes falls back to deterministic find after a request failure, not after an empty plan. The campaign added fail-closed ingest checks, liveness probes, unique persona workspaces, and container-local working directories after lock-file races appeared on both Docker Desktop and native Linux.",
      timeout: "OpenViking has separate language-model, HTTP, task-drain, and shutdown timeouts. Raising only the HTTP limit did not extend the internal model-request deadline. Worse, a queue-worker exception could leave a commit task pending without acknowledging it; the client then waited the full 1,800 seconds before invalidating the persona. A timeout here means unknown memory state, so the final harness did not score or blindly retry that persona.",
      caveat: "A single server binary with an embedded workspace directory (no external database). Each instance uses a single-process lock file, so concurrent workers need separate workspaces. The .openviking.pid lock persists across container recreation, requiring manual cleanup before relaunching. Workspace I/O on Docker bind mounts caused unrecoverable commit task failures on both Windows and native Linux; putting the workspace on container-local storage was the fix. Every configuration model uses extra: forbid, so an unknown field (such as the documented telemetry.enabled) exits the server at boot. Nine failed attempts make workload roughly 15 to 20% high.",
      repo: "https://github.com/volcengine/OpenViking"
    },
    {
      id: "mnemosyne",
      name: "Mnemosyne",
      short: "Mnemosyne",
      tag: "v5ftc",
      role: "featured",
      macro: 0.116,
      dynamic: 0.344,
      static: -0.204,
      conditional: 0.207,
      seh3: 0.341,
      micro: 0.275,
      outcomes: { correct: 910, partial: 722, blank: 1880, incorrect: 238 },
      tokens: { cached: 0.0, uncached: 116.3, outputTotal: 9825512, caveat: "Clean run, but this arm used no language model during extraction." },
      tested: "Mnemosyne 3.14.0 with the shipped 168-hour working-memory lifetime, forced consolidation after each session, and no language model extracting facts during ingest.",
      verdict: "The simplest provider to install: embedded Python storage with no database, no daemon, and no server process. The fewest measured tokens of any provider, well suited when model-token workload is the primary constraint. The tradeoff is a 50.1% blank rate and a negative static score (-0.204), meaning it asserted false static facts more often than it answered correctly.",
      process: "The plugin's seven-day working-memory lifetime is shorter than the dataset's median 29-day gap, so the normal every-ten-turn auto-sleep rarely fired before old rows expired. The run therefore called the vendor's sleep operation at each session boundary. Bookkeeping proposal rows could still outrank real turns; the plugin overlay dropped them without backfilling, leaving sparse recall.",
      timeout: "No recurring HTTP transport failure survived into the final wave. The dangerous failures were quieter: a low output cap produced empty consolidation proposals with no error, and background sleep deadlines could return before useful consolidation existed. The final run raised the sleep budget and drained each forced session sleep before asking questions.",
      caveat: "No external infrastructure: pip install, point at an embedding endpoint, and start. The configuration surface is small and mostly hardcoded. The main deployment concern is that the shipped consolidation (sleep) produces lossy summaries, and the seven-day working-memory TTL can delete backdated data before consolidation runs. In Docker, output files are written to a container-local path, so row counts on the host always read zero and must be checked via docker exec. The embedding model needs network access to Hugging Face; unreachable CDNs fail with no retry.",
      repo: "https://github.com/mnemosyne-oss/mnemosyne"
    },
    {
      id: "hindsight",
      name: "Hindsight",
      sourceName: "Hindsight (2nd arm)",
      short: "Hindsight",
      tag: "v5ftcall",
      role: "featured",
      macro: 0.281,
      dynamic: 0.455,
      static: 0.114,
      conditional: 0.275,
      seh3: 0.452,
      micro: 0.401,
      outcomes: { correct: 1808, partial: 507, blank: 877, incorrect: 558 },
      tokens: { cached: 384.6, uncached: 1942.2, outputTotal: 21469851, caveat: "Token window is inflated by roughly 3% after one late restart." },
      tested: "Hindsight 0.8.6 with embedded PostgreSQL and its pgvector search extension. Recall included all three native memory types: observations (consolidated patterns), world facts, and experiences. It used the provider's full 4,096-token budget rather than a shared top-five cut.",
      verdict: "Second-best dynamic result (0.455) at roughly one third of Honcho's workload. A large exposed configuration surface: extraction temperature, output cap, mission prompts, reranker provider, recall types, recency decay, and dedup threshold. That flexibility is valuable for tuning but requires careful attention to defaults: the observations-only Hermes default scored 0.218 versus 0.281 with all types enabled.",
      process: "Version 0.8.6 fixed part of an append-mode date-loss bug, but some timestamps could still fall back to the daemon clock. The daemon and its embedded PostgreSQL therefore ran in the same simulated-time domain. Near-duplicates remained because the documented maximal marginal relevance (MMR) dedup step was not implemented despite two docstrings promising it, and the 0.97 merge threshold left many paraphrases distinct.",
      timeout: "Early high-concurrency runs exposed ambiguous timeouts: the client gave up after 480 seconds while some retain calls continued server-side and later recorded success at 796 to 1,030 seconds. Consolidation drains could time out separately and leave work carrying into the next session. The final setup reduced concurrency, raised bounded timeouts, and failed closed when the store could not be proven settled.",
      caveat: "The infrastructure stack is a daemon process, PostgreSQL with pgvector, an embedding service, and optionally a GPU reranker. The pg0 embedded mode simplifies setup but requires careful HOME directory handling when multiple containers share a volume. The documented MMR diversity stage (which would suppress near-duplicate search results) does not exist in the code; near-duplicate paraphrases occupy multiple high positions. The median recall was about 126 memories per question, so context budgets need planning. The dedup threshold (cosine 0.97) leaves many paraphrases distinct. The 0.8.4 to 0.8.6 releases fixed a timestamp bug where the append path dropped the caller's timestamp and stamped everything with the wall clock; a related issue remains open.",
      repo: "https://github.com/vectorize-io/hindsight"
    }
  ],
  conflictOutcomes: {
    honcho: {
      dynamic: { N: 2946, correct: 1836, partial: 447, blank: 499, incorrect: 164 },
      static: { N: 360, correct: 131, partial: 54, blank: 82, incorrect: 93 },
      conditional: { N: 444, correct: 284, partial: 0, blank: 145, incorrect: 15 }
    },
    mem0: {
      dynamic: { N: 2946, correct: 1124, partial: 439, blank: 777, incorrect: 606 },
      static: { N: 360, correct: 106, partial: 67, blank: 80, incorrect: 107 },
      conditional: { N: 444, correct: 375, partial: 0, blank: 65, incorrect: 4 }
    },
    supermemory: {
      dynamic: { N: 2946, correct: 672, partial: 520, blank: 1245, incorrect: 509 },
      static: { N: 360, correct: 87, partial: 65, blank: 98, incorrect: 110 },
      conditional: { N: 444, correct: 316, partial: 0, blank: 120, incorrect: 8 }
    },
    retaindb: {
      dynamic: { N: 2946, correct: 831, partial: 713, blank: 1037, incorrect: 365 },
      static: { N: 360, correct: 78, partial: 79, blank: 98, incorrect: 105 },
      conditional: { N: 444, correct: 238, partial: 0, blank: 188, incorrect: 18 }
    },
    hindsight: {
      dynamic: { N: 2946, correct: 1589, partial: 445, blank: 441, incorrect: 471 },
      static: { N: 360, correct: 91, partial: 62, blank: 126, incorrect: 81 },
      conditional: { N: 444, correct: 128, partial: 0, blank: 310, incorrect: 6 }
    },
    openviking: {
      dynamic: { N: 2946, correct: 488, partial: 318, blank: 1914, incorrect: 226 },
      static: { N: 360, correct: 41, partial: 38, blank: 245, incorrect: 36 },
      conditional: { N: 444, correct: 85, partial: 0, blank: 357, incorrect: 2 }
    },
    mnemosyne: {
      dynamic: { N: 2946, correct: 768, partial: 683, blank: 1400, incorrect: 95 },
      static: { N: 360, correct: 41, partial: 39, blank: 146, incorrect: 134 },
      conditional: { N: 444, correct: 101, partial: 0, blank: 334, incorrect: 9 }
    }
  },
  sessionBins: ["1–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31–35", "36–40", "41–45", "46–50", "51–55"],
  sessionSeries: {
    honcho: [
      [40,31,1,6,2],[300,214,24,49,13],[347,225,36,69,17],[391,237,49,77,28],[442,250,60,86,46],[397,234,65,78,20],[408,229,59,82,38],[404,233,73,64,34],[395,240,56,71,28],[438,262,55,87,34],[188,96,23,57,12]
    ],
    mem0: [
      [40,18,7,12,3],[300,127,44,104,25],[347,148,55,96,48],[391,160,57,112,62],[442,169,57,122,94],[397,168,67,91,71],[408,164,50,98,96],[404,171,57,80,96],[395,157,57,89,92],[438,213,43,85,97],[188,110,12,33,33]
    ],
    supermemory: [
      [40,10,9,21,0],[300,66,55,154,25],[347,97,74,138,38],[391,108,54,163,66],[442,123,62,171,86],[397,126,61,143,67],[408,105,60,162,81],[404,98,65,163,78],[395,124,62,126,83],[438,141,64,154,79],[188,77,19,68,24]
    ],
    retaindb: [
      [40,7,14,15,4],[300,88,87,110,15],[347,98,90,131,28],[391,116,98,133,44],[442,139,100,150,53],[397,110,74,162,51],[408,137,84,140,47],[404,117,82,131,74],[395,123,71,136,65],[438,137,71,154,76],[188,75,21,61,31]
    ],
    hindsight: [
      [40,17,9,12,2],[300,158,39,73,30],[347,180,55,75,37],[391,189,63,83,56],[442,221,57,93,71],[397,198,48,89,62],[408,191,60,81,76],[404,197,56,88,63],[395,175,53,108,59],[438,202,47,111,78],[188,80,20,64,24]
    ],
    openviking: [
      [40,3,5,29,3],[300,41,37,213,9],[347,57,37,227,26],[391,59,36,266,30],[442,79,49,283,31],[397,57,41,266,33],[408,63,31,284,30],[404,75,40,256,33],[395,69,31,270,25],[438,84,35,282,37],[188,27,14,140,7]
    ],
    mnemosyne: [
      [40,11,11,16,2],[300,82,75,137,6],[347,82,67,184,14],[391,92,97,185,17],[442,105,83,231,23],[397,99,79,185,34],[408,102,73,201,32],[404,99,79,201,25],[395,92,69,206,28],[438,106,60,231,41],[188,40,29,103,16]
    ]
  }
};
