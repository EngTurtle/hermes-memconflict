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
      verdict: "Highest macro score (0.477) and the strongest dynamic result (0.643). Its lead over mem0 is larger than the measured judge-noise band.",
      process: "The database migrations assumed 1,536-dimensional vectors, so the vendor's configuration script had to resize the columns and rebuild both vector-search indexes (HNSW) for the shared 768-dimensional embedder. The deriver also stored repetition loops until its output cap was reduced. Hybrid recall remained plugin-shaped: a large assembled context block rather than a ranked top-five list.",
      timeout: "The SDK retries timeout and network failures even for message-adding write requests, and that call has no idempotency key—a unique request token the server could use to reject a duplicate. If the server commits before the response is lost, a retry could add the messages twice; this is a code-path risk, not an observed duplicate. A successful Dream request also did not prove its specialist jobs succeeded because failures appeared only in the deriver log.",
      caveat: "The benchmark manually triggered Honcho's shipped Dream consolidation after every session because the normal scheduler waits for idle time. That makes this a quality-oriented deployment configuration and an upper-bound workload estimate. Honcho injected roughly 25,000 tokens per question and alone required a 49,152-token judge window.",
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
      verdict: "The strongest conditional result (0.836) and second-highest macro score in a conventional SDK-plus-Qdrant stack, but 20.6% of dynamic answers were wrong.",
      process: "The 2.x API changed entity scoping, search arguments, and batch embedding behavior. The adapter had to omit an unsupported dimensions parameter on both single and batched embedding calls, bake the sparse-search models into the image, and remove an inherited OpenRouter key that otherwise redirected internal calls. Version 2.0.14 is ADD-only: it extracts new facts but no longer runs the older ADD/UPDATE/DELETE/NONE reconciliation pass.",
      timeout: "A failed extraction now raises instead of being quietly logged, so the adapter records the failed add and continues rather than killing the whole persona. The Python client also leaves worker threads alive after the work is finished; the harness flushes results and exits the process explicitly. No recurring HTTP-timeout failure survived into the final run.",
      caveat: "Old and new facts can remain together for retrieval, leaving the answer model to resolve them. One restarted persona inflates token counts by about 1%; it does not affect banked answers. The observed 80.4% prompt-cache rate depends on repeated local templates and may not transfer to another host.",
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
      verdict: "Good conditional recall (0.694), but the ranked-memory list was empty for 904 of 3,750 questions.",
      process: "Ingest is asynchronous, so every accepted document had to reach done before its questions were asked. The memory agent also required Qwen's tool-call parser, which converts model-written tool syntax into actual function calls; without it the server finalized chunks while silently falling back from extracted memories. Under the simulated clock, node-cron tried to replay years of missed schedules and allocated an estimated 73 GB, so the server was respawned after every session on the same data directory.",
      timeout: "Transport failures and read timeouts were retried—up to 30 times in the clock-synced preset—but HTTP 5xx responses were not. More importantly, a timed-out document POST may already have been accepted: re-submitting it could duplicate memory. When the server died during drain, the harness refused to resubmit, marked the persona invalid, and reran it from a clean checkpoint. The 0.0.5 dispatcher could also keep returning HTTP 200 while leaving every new document queued.",
      caveat: "This score should not be read as the current Supermemory release. It is the newest self-hosted Linux build that completed a document in this campaign. The answer model could still see the profile block when ranked recall was empty, while the evidence judge only saw the ranked list. Nine failed attempts make workload roughly 10–20% high.",
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
      verdict: "A 0.270 macro score with no clear category strength, plus the largest measured output-token total.",
      process: "Seven upstream defects were found and five are patched in the test image: Prisma/schema compatibility, seed rows, date propagation, cache isolation, and malformed high-confidence search results. Approximate vector indexes (IVFFlat) created on empty tables were removed in favor of exact search at this scale. The quality profile also needed its post-vector budget raised from 120 to 2,000 ms; otherwise 117 of 122 smoke questions silently skipped graph and temporal stages.",
      timeout: "The generic write helper retries server errors, connection errors, and timeouts without an idempotency key—a unique token that would let the server reject a duplicate request. A response lost after commit could therefore replay a whole-session ingest; this is a source-code risk, not an observed duplicate. Separately, lifecycle waits now use the scheduler's own completion marker so a slow summary cannot be mistaken for a finished session.",
      caveat: "This is best-effort deployable, not an untouched-plugin result: the promotion field used here is vendor-exposed but absent from the pinned Hermes plugin. Extraction sometimes kept that a move happened while losing the destination. RetainDB local is a separate product and was excluded after one search reached about 245 seconds at 2,897 memories.",
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
      verdict: "Low committed-error rates mostly came from abstention: 67.1% of all answers were blank.",
      process: "The planner may decide that no search is needed and return only the session-start block; Hermes falls back to deterministic find after a request failure, not after an empty plan. The campaign added fail-closed ingest checks, liveness probes, unique persona workspaces, and container-local working directories after lock-file races appeared on both Docker Desktop and native Linux.",
      timeout: "OpenViking has separate language-model, HTTP, task-drain, and shutdown timeouts. Raising only the HTTP limit did not extend the internal model-request deadline. Worse, a queue-worker exception could leave a commit task pending without acknowledging it; the client then waited the full 1,800 seconds before invalidating the persona. A timeout here means unknown memory state, so the final harness did not score or blindly retry that persona.",
      caveat: "The deterministic find arm was useful for diagnosis but is not part of this comparison. Five successful reruns used a 1,800-second HTTP timeout instead of 600, while the internal model-request timeout stayed at 600; this is a disclosed mixed-configuration deviation, not proof that every failure needed the larger value. Nine failed attempts make workload roughly 15–20% high.",
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
      verdict: "The fewest measured tokens and a low dynamic wrong-answer rate, but 50.1% of all answers were blank and the static score was −0.204.",
      process: "The plugin's seven-day working-memory lifetime is shorter than the dataset's median 29-day gap, so the normal every-ten-turn auto-sleep rarely fired before old rows expired. The run therefore called the vendor's sleep operation at each session boundary. Bookkeeping proposal rows could still outrank real turns; the plugin overlay dropped them without backfilling, leaving sparse recall.",
      timeout: "No recurring HTTP transport failure survived into the final wave. The dangerous failures were quieter: a low output cap produced empty consolidation proposals with no error, and background sleep deadlines could return before useful consolidation existed. The final run raised the sleep budget and drained each forced session sleep before asking questions.",
      caveat: "The static result is the deciding risk: 37.2% wrong versus 11.4% correct. Sleep summaries were lossy, and this arm used no language model during extraction, so its workload belongs to a lower-cost configuration class rather than a like-for-like implementation comparison.",
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
      verdict: "A 0.281 macro score and the second-best dynamic result (0.455), but conditional recall remained weak and context was unusually wide.",
      process: "Version 0.8.6 fixed part of an append-mode date-loss bug, but some timestamps could still fall back to the daemon clock. The daemon and its embedded PostgreSQL therefore ran in the same simulated-time domain. Near-duplicates remained because the documented maximal marginal relevance (MMR) step—which should suppress repetitive search results—was not implemented, and the 0.97 merge threshold left many paraphrases distinct.",
      timeout: "Early high-concurrency runs exposed ambiguous timeouts: the client gave up after 480 seconds while some retain calls continued server-side and later recorded success at 796–1,030 seconds. Consolidation drains could time out separately and leave work carrying into the next session. The final setup reduced concurrency, raised bounded timeouts, and failed closed when the store could not be proven settled.",
      caveat: "The median recall contained roughly 126 memories per question. That broad context helped changing-fact coverage but increased answer-context cost and made fixed top-k evidence metrics misleading. One persona hit a separate 600-second PostgreSQL recall timeout and passed unchanged on a near-solo rerun; its failed near-complete attempt inflates workload by about 3%.",
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
