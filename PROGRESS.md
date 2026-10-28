# APEX progress

Research and offline simulation only. Live trading is disabled (`live_trading_enabled=False`), risk is
0.5% per trade, and at most one position is allowed. No module places broker orders or reads an account.

## Latest development checkpoint
- The current project files contain the regime and liquidity/market-structure research agents, their coordinator
  and CLI integration, and tests for causal features, gap handling, deterministic output, walk-forward selection,
  parameter sensitivity and sample/effect reporting. `PROGRESS.md` did not identify a "Task 52" checkpoint;
  the checked-in implementation and tests were used as the source of truth.
- Added a local, read-only dashboard at `dashboard.py`. It uses only Python's standard library and binds to
  `127.0.0.1`. It reads `logs/research-latest.txt`, which the research CLI now writes after a successful load.
  Missing daily/weekly results, persisted simulated positions and per-trade history are explicitly shown as
  unavailable; the dashboard never fabricates them.
- A CLI smoke run used only the first 1,200 candles of the existing 12-month CSV (2025-10-26 through 2025-10-31).
  There were no out-of-sample trades/folds in this short slice; regime and liquidity/structure hypotheses were
  inconclusive and did not pass their criteria. This is a pipeline check, not strategy evidence.
- Existing candidate simulation integration remains in `research/candidates.py`, `research/agents.py`,
  `research/walk_forward.py` and `research/backtest.py`: causal signal functions are selected on training windows,
  then simulated with the configured cost/risk model and returned as recorded trade results. It is not a durable
  position/trade ledger. No duplicate execution engine was added.
- Full-year run completed on the exact dataset below. Two full executions completed with exit code 0 and matching
  research metrics; the second report is the dashboard's latest. A duplicate execution occurred because the first
  background run's state was not visible through the task-output tool. Preserve both archives and logs.
- Both CLI reports record their own `perf_counter` runtime; the external runner's elapsed timer was longer:
  run `20261027T225430841617Z` CLI 9,523.104 s (2:38:43.104), external 9,957.592 s (2:45:57.592);
  run `20261027T235121675596Z` CLI 12,701.105 s (3:31:41.105), external 12,768.147 s (3:32:48.147).
  This is substantially slower than the previous ~10-minute estimate and blocks efficient repeated research.
- Full-year evidence summary: 14 regime variants and 64 sweep/structure definitions, each with 9 walk-forward folds;
  regime OOS expansion separation was large but failed the minimum-duration criterion; selected sweep/structure
  association was weak and failed OOS significance and the existing-definition comparison. Candidate simulation
  produced 1,394 OOS trades at +0.02R expectancy and PF 1.03 under placeholder costs, but t=0.54, only 4/8
  trading folds were positive, and the result remains inconclusive and experimental. The user's setup remains
  incomplete. See the run archives for full fold-level output.
- Verification: all 250 tests passed in focused batches covering every test module. The combined one-shot suite
  had previously exceeded the execution service timeout; focused batches were used. One initial test run exposed
  an incorrect expected sort order in the new report-store test; the test was corrected and the final batches pass.
  CLI report/archive ordering and failure behavior were tested; dashboard HTTP startup and read-only POST rejection
  were also tested.

## Completed modules
| Area | Module | State |
|---|---|---|
| Data | `data/mt5_history_collector.py` | Read-only M5 collector (bounded probes, staged, no-overwrite). Caller owns MT5 init/shutdown. |
| Data | `data/historical_data_loader.py`, `data/dataset_audit.py` | CSV loader and audit (gaps are observations, never filled). |
| Research | `research/market_features.py`, `market_structure.py`, `liquidity_sweeps.py` | Causal features, confirmed swings/breaks, candidate sweeps. |
| Research | `research/price_expansion.py` | Predictors (causal) kept apart from expansion labels (future); baseline flags. |
| Research | `research/regimes.py`, `liquidity_structure.py`, `study_agents.py` | Point-in-time state/event studies; future labels evaluation-only; train-selected, chronological OOS; sensitivity and stability reporting. |
| Research | `research/backtest.py`, `walk_forward.py`, `rule_spec.py` | Cost-aware simulator, walk-forward with embargo, versioned specs with promotion gate. |
| Research | `research/candidates.py` | Explicit candidates; incomplete ones are rejected with reasons; variant count recorded. |
| Research | `research/coordinator.py`, `agents.py`, `run_suite.py` | Coordinator + Price Expansion agent + Candidate Strategy agent. |

## Data on disk (`data/historical/`, never modified)
- `XAUUSDm_M5_20250901T000000Z_20251001T000000Z.csv`, `..20251020T000000Z_20251027T000000Z.csv`, `..20251020T120000Z_20251020T130000Z.csv` (originals)
- `XAUUSDm_M5_20260925T000000Z_20261025T000000Z.csv` (1 month)
- `XAUUSDm_M5_20251025T000000Z_20261025T000000Z.csv` (12 months, 70,588 candles; contains the same 5,772 candles as the 1-month file, verified identical)

## Full-year research validation (2026-10-27)
- Input: `data\historical\XAUUSDm_M5_20251025T000000Z_20261025T000000Z.csv`, 4,733,839 bytes, 70,588
  data rows; SHA-256 before and after execution (identical):
  `ae7e9453de78ebe3f8996aca64c91f55a79b8e181907338dc5cc9b4f921b0073`.
  CSV candle-open range: `2025-10-26T22:05:00+00:00` through `2026-10-23T20:55:00+00:00` (361 calendar
  days, 13 calendar months). The CSV was only read by the CLI.
- Run archives: `logs\research-full-year-20261027T225430841617Z.txt` and
  `logs\research-full-year-20261027T235121675596Z.txt`. Both process records show the exact requested CLI
  command and exit code 0; all four agents returned `ok`. The first run's stdout is retained in
  `logs\research-full-year-cli-20261027T201522.log`. The second run's stdout log at
  `logs\research-full-year-cli-20261027T201906.log` was restored from its complete archived report after the
  output file was removed during run monitoring; its process record remains in
  `logs\research-full-year-process-20261027T201906.json`. The second archive is also `logs\research-latest.txt`;
  it retains the run ID, input path, candle count, data range and runtime.
- Price-expansion study: 635,292 future-outcome labels over 70,588 candles. At the 15-minute/2-ATR horizon the
  base expansion rate was 16.6% (69,796 labelled rows; 776 gap, 13 ATR-unavailable and 3 incomplete rows
  excluded). Untuned tick-volume-spike and large-closed-candle flags had above-baseline lift in both chronological
  halves, but that half-split is descriptive, not a tuned independent strategy test; these are investigation leads,
  not tradeable edge evidence.
- Regime study: 14 variants, 9 folds; `compression_q25_75` selected in all 9. On thinned OOS rows, hot/cold
  expansion was 25.5%/9.8%, +15.7 percentage points (z=19.42), versus the fixed large-candle baseline's
  +13.4pp (z=15.69). Direction agreed in 9/9 folds and the baseline margin passed, but median regime duration was
  only 3 candles versus the required 6. The agent therefore did not justify regime labelling as a strategy basis.
- Liquidity/structure study: 64 definitions, 9 folds. The reference detector cross-check was identical at
  1,674,091 events. The selected OOS result was 4,371 events (3,097 thinned), 8.7% aligned versus 8.2% expected,
  z=0.56; the existing definition had z=0.65 and +1.29pp excess versus selected +0.75pp. Fold-direction consistency
  passed at 8/9, but OOS significance and beating the existing definition failed. The reference unlimited/repeat
  definition was highly active (99.6% of candles had a recent sweep; 9,509 mean live swings per candle), making
  that variant operationally uninformative. The larger full-sample break associations are descriptive and were not
  validated as a trade rule.
- Candidate study: 57 variants generated; 56 runnable (48 strategy candidates and 8 null controls); the owner's
  `user_sweep_mss_ifvg_keylevel` was rejected with 12 reasons: 10 unresolved setup definitions, no exit rule and
  no implemented signal family. Across 9 folds there were 1,394
  OOS trades, +0.02R expectancy, PF 1.03, 13.0% closed-trade drawdown and 0.03R average modeled costs. The
  expectancy t-statistic was 0.54 (threshold 2); 4/8 folds with trades were positive. The 6,095 null-control
  trades had -0.04R expectancy, PF 0.90 and 77.8% drawdown. The placeholder promotion bar was met, but evidence
  was not distinguishable from zero; candidates remain experimental.
- Limitations: one instrument/timeframe and about one year; 57 candidate variants and 64 event definitions were
  tried; observations/folds are dependent. CSVs have no spread, M1, commission, swap or margin data; modeled
  spread/slippage are placeholders. Bar execution uses next-open entries and stop-first intrabar ties, with
  incomplete treatment of gaps and no lot rounding. Positive descriptive or pooled metrics do not establish
  profitability.

## Run the research suite
`.\.venv\Scripts\python.exe -m research.run_suite data\historical\XAUUSDm_M5_20251025T000000Z_20261025T000000Z.csv`
(two measured runs took 2:38:43 and 3:31:41; allow several hours until runtime is improved;
`--limit-candles N` for a quick check). Completed runs are saved as
timestamped `logs\research-<label>-<run-id>.txt` archives and atomically published to
`logs\research-latest.txt` for the dashboard. The report records its dataset, run ID and runtime; the historical
CSV remains read-only.

## Run the local dashboard (Windows)
From `C:\APEX`, run:

`.\.venv\Scripts\python.exe dashboard.py`

Open `http://127.0.0.1:8765/`. The dashboard is local and read-only; stop it with Ctrl+C. Run the research suite
first to populate its latest report. It displays the report's historical date range, not live-data freshness.
Only monthly walk-forward performance is shown when present in the report; daily and weekly performance, open
simulated positions and individual trade history are not currently persisted.

## Known limitations
- No spread or M1 data; costs are placeholders (spread 0.30, slippage 0.05 per side).
- Session and timezone rules, key levels, FVG/IFVG and MSS/CHOCH definitions are unresolved, so the owner's
  hypothesis is registered as a candidate but rejected as incomplete.
- Regime buckets are reporting-only (causal realized-volatility ratio), not a validated model.
- Collector note: the position-based chunk fetch can race with a live bar rolling over during a multi-minute
  collection. A single-call fetch (`max_chunk_size=100_000`) avoids it.
- The dashboard is an offline research/status view, not an always-on market monitor. It has no live feed,
  persisted simulated-position ledger, or daily/weekly performance store.
- Simulated execution is an OHLC-bar approximation: intrabar order is unknown (stop-first tie handling), gaps
  are not filled at entry, positions can span gaps, and costs are placeholders. Broker volume/margin/swaps and
  measured spread/slippage are not modelled.

## Repository audit (2026-10-28)

This is a code-and-evidence audit, not a live-system test. No MT5 connection, data download, order, or account-setting
change was made. The safety baseline remains `live_trading_enabled=False`, `risk_per_trade=0.005` and
`max_open_positions=1` in `/home/runner/work/apex/apex/config/settings.py:18-24`.

### Agent-by-agent implementation and verification

| Agent/component | Classification | Evidence and verification | Boundary |
|---|---|---|---|
| `PriceExpansionAgent` | Implemented and verified as descriptive research | `/home/runner/work/apex/apex/research/agents.py:47-139`; `/home/runner/work/apex/apex/tests/test_coordinator.py`, `TestAgents.test_price_expansion_agent_reports_without_changing_inputs`; full-year archived report has `[ok]`. | Measures future expansion outcomes against causal, fixed baseline flags; it does not generate an executable strategy. |
| `MarketRegimeAgent` | Implemented and verified as offline study code | `/home/runner/work/apex/apex/research/study_agents.py:42-122`, calling causal feature and chronological study code in `/home/runner/work/apex/apex/research/regimes.py`; `/home/runner/work/apex/apex/tests/test_research_studies.py`, `TestAgentsAndCoordinator.test_agents_run_and_are_deterministic` and `.test_regime_agent_reports_variant_count_and_exclusions`; full-year report has `[ok]`. | The 2025–26 study failed its minimum regime-duration criterion (median 3 candles vs 6); it is not a validated regime model. |
| `LiquidityStructureAgent` | Implemented and verified as offline study code | `/home/runner/work/apex/apex/research/study_agents.py:152-274` and `/home/runner/work/apex/apex/research/liquidity_structure.py`; `/home/runner/work/apex/apex/tests/test_research_studies.py`, `TestAgentsAndCoordinator.test_liquidity_agent_reports_cross_check_and_variant_count` and `.test_agents_run_and_are_deterministic`; full-year report has `[ok]`. | Studies confirmed swings, structural breaks and candidate sweeps. It explicitly does not implement the complete user sequence or FVG/IFVG/key-level rules. |
| `CandidateStrategyAgent` | Partially implemented | `/home/runner/work/apex/apex/research/agents.py:159-304`, `/home/runner/work/apex/apex/research/candidates.py:115-264` and `/home/runner/work/apex/apex/research/walk_forward.py:132-185`; `/home/runner/work/apex/apex/tests/test_coordinator.py`, `TestAgents.test_candidate_agent_counts_variants_rejects_and_keeps_candidates_experimental`; also `/home/runner/work/apex/apex/tests/test_candidates.py` and `/home/runner/work/apex/apex/tests/test_backtest.py`. Full-year report: 57 variants, 1,394 OOS trades, +0.02R expectancy, t=0.54; promotion remains inconclusive. | Searches a finite, predefined grid; it does not invent arbitrary features or hypotheses. The owner-provided setup is rejected as incomplete, and runnable candidates remain experimental. |
| Coordinator and CLI | Partially implemented | `/home/runner/work/apex/apex/research/coordinator.py:78-180` and `/home/runner/work/apex/apex/research/run_suite.py:21-62`; `/home/runner/work/apex/apex/tests/test_coordinator.py`, `TestCoordinator.test_runs_agents_in_order_and_isolates_failures`; `/home/runner/work/apex/apex/tests/test_research_studies.py`, `TestAgentsAndCoordinator.test_failures_are_isolated_and_order_is_kept`; `/home/runner/work/apex/apex/tests/test_run_suite.py`. | Shares one validated immutable context and isolates agent exceptions, but runs sequentially. There is no scheduled agent service, durable task state, retry/recovery workflow, or shared persistent result store beyond research reports. |

### Research-area matrix

| Area | Classification | Evidence and current boundary |
|---|---|---|
| 1. Collection, validation, gaps, multiple timeframes | Partially implemented | Candle/CSV validation and non-mutating gap audits are in `/home/runner/work/apex/apex/core/market_data.py:30-102`, `/home/runner/work/apex/apex/data/historical_data_loader.py:50-102` and `/home/runner/work/apex/apex/data/dataset_audit.py:72-163`. `/home/runner/work/apex/apex/data/mt5_history_collector.py:22-28,403-482` accepts an injected rates API and collects bounded M5 history. Tests: `/home/runner/work/apex/apex/tests/test_market_data.py`, `/home/runner/work/apex/apex/tests/test_dataset_audit.py`, `/home/runner/work/apex/apex/tests/test_historical_data_loader.py`, `/home/runner/work/apex/apex/tests/test_mt5_history_collector.py`. No multi-timeframe processing or direct broker/account adapter is present; collection tests use a fake API, not a live terminal. |
| 2. Liquidity, structure, FVG/IFVG, levels, candle patterns | Partially implemented | Confirmed swings, close-basis breaks and candidate sweeps exist in `/home/runner/work/apex/apex/research/market_structure.py` and `/home/runner/work/apex/apex/research/liquidity_sweeps.py`; study variants are in `/home/runner/work/apex/apex/research/liquidity_structure.py`. `/home/runner/work/apex/apex/tests/test_market_structure.py`, `/home/runner/work/apex/apex/tests/test_liquidity_sweeps.py` and `/home/runner/work/apex/apex/tests/test_research_studies.py` verify definitions, causality and a detector cross-check. FVG/IFVG, key-level retests, session rules and the complete user sequence are unresolved/not implemented; these studies are associations, not trade rules. |
| 3. Independent pattern discovery and hypotheses | Partially implemented | `/home/runner/work/apex/apex/research/candidates.py:155-247` enumerates specified momentum, large-candle fade, generic sweep and null-control families. `/home/runner/work/apex/apex/tests/test_candidates.py`, `TestGeneration.test_default_grid_counts_every_variant_and_rejects_the_unresolved_hypothesis` verifies 57 fixed variants and the incomplete user candidate rejection. This is bounded parameter search over authored rule families, not autonomous discovery of novel patterns or hypotheses. |
| 4. Volatility, trend, compression, momentum, expansion, regimes | Partially implemented | Causal ATR, realized-volatility, compression, momentum and range features are built in `/home/runner/work/apex/apex/research/price_expansion.py`; expanding-rank regime features are in `/home/runner/work/apex/apex/research/regimes.py`. `/home/runner/work/apex/apex/tests/test_price_expansion.py` and `/home/runner/work/apex/apex/tests/test_research_studies.py`, `TestRegimeBuildingBlocks` cover causality, warm-up and gaps. Feature studies are exploratory; regime results did not meet all criteria and there is no validated tradeable regime model. |
| 5. Generation, comparison, backtest, walk-forward, OOS | Partially implemented | `/home/runner/work/apex/apex/research/backtest.py:1-34,215-335` simulates one position using OHLC bars; `/home/runner/work/apex/apex/research/walk_forward.py:99-185` selects on train windows and evaluates later windows with an embargo. Tests: `/home/runner/work/apex/apex/tests/test_candidates.py`, `/home/runner/work/apex/apex/tests/test_backtest.py`, `/home/runner/work/apex/apex/tests/test_walk_forward.py`, `/home/runner/work/apex/apex/tests/test_coordinator.py`. The full-year archive records 9 folds and 1,394 pooled OOS trades, but only one symbol/timeframe and placeholder costs; no multi-timeframe or cross-market generalization. |
| 6. Significance, uncertainty, overfit and null controls | Partially implemented | `/home/runner/work/apex/apex/research/study_stats.py` thins overlapping observations and computes proportion z-statistics; candidate reporting counts variants, computes a trade-return t-statistic, compares null controls and applies a promotion gate (`/home/runner/work/apex/apex/research/agents.py:226-271`, `/home/runner/work/apex/apex/research/rule_spec.py`). `/home/runner/work/apex/apex/tests/test_research_studies.py`, `TestStudyStats`; `/home/runner/work/apex/apex/tests/test_rule_spec.py`; and `/home/runner/work/apex/apex/tests/test_coordinator.py`, `TestAgents` cover these safeguards. There are no robust confidence intervals or multiple-testing correction; thinning does not make all observations/folds independent. The full-year candidate t-statistic was 0.54 and results remain inconclusive. |
| 7. Continuous monitoring and daily opportunity detection | Planned but not implemented | `/home/runner/work/apex/apex/research/run_suite.py` runs only on explicit CLI invocation; the collector is bounded, caller-initialized, read-only M5 history collection. There is no live feed, scheduler, daily opportunity scanner or freshness loop. |
| 8. Strategy selection, performance, drift, revalidation | Partially implemented | `/home/runner/work/apex/apex/research/walk_forward.py` selects a rule independently within historical train folds; `/home/runner/work/apex/apex/research/rule_spec.py` stores experimental specs and gates promotion for human review. No deployed strategy registry, ongoing performance monitor, drift detector or automatic revalidation exists. |
| 9. News and social sentiment | Missing or blocked | No news/social source, API client, feed configuration or sentiment research module is present. `PriceExpansionAgent` explicitly lists news context as unmodelled (`/home/runner/work/apex/apex/research/agents.py:126-133`). |
| 10. Coordinator, scheduling, task status, shared results, failures | Partially implemented | The coordinator shares a per-run context, preserves deterministic order and returns per-agent errors; tests above verify failure isolation. It has no scheduler, persistent task/status API, cross-run result database, retries or restart recovery. Research report archives are the only durable run output (`/home/runner/work/apex/apex/research/report_store.py`). |
| 11. Dashboard explanations and opportunities | Partially implemented | `/home/runner/work/apex/apex/dashboard.py:30-121,124-205` displays saved agent statuses/errors, candidate validation, research text and reported drawdown; tests in `/home/runner/work/apex/apex/tests/test_dashboard.py` verify escaping, unavailable data and GET-only behavior. It is local/read-only and has no daily opportunity feed, trade ledger, daily/weekly performance or approval workflow; no candidate is auto-approved. |

### Evidence, tests, and blockers

- Audit verification ran 254 relevant unit tests successfully: 115 data/feature tests and 139 research/risk/dashboard tests. Commands were `python -m unittest tests.test_market_data tests.test_dataset_audit tests.test_historical_data_loader tests.test_mt5_history_collector tests.test_market_features tests.test_market_structure tests.test_liquidity_sweeps tests.test_price_expansion` and `python -m unittest tests.test_candidates tests.test_backtest tests.test_walk_forward tests.test_research_studies tests.test_rule_spec tests.test_coordinator tests.test_pipeline tests.test_run_suite tests.test_risk_manager tests.test_settings tests.test_dashboard`. No full-year research rerun was performed; the existing archives/logs above were inspected and preserved.
- The full-year 2026-10-27 report (`/home/runner/work/apex/apex/logs/research-full-year-20261027T235121675596Z.txt`) reports all four agents `[ok]`, but not a successful strategy: the liquidity result failed significance/comparison criteria; candidate performance was inconclusive; the regime study failed duration. “Agent ok” means the run completed, not that an edge was approved.
- Existing evidence is limited to XAUUSDm M5 and about one year. Historical CSVs have no M1 execution data, measured spread, commission, swaps or margin. Costs and volume value are configured inputs/placeholders. No dependency manifest or broker account/order/symbol-spec adapter is present; the collector protocol only retrieves rates. No task scheduler, news/social integration or persisted trade/risk ledger exists.
- Independent discovery is limited to the authored candidate families and their fixed parameter grids. APEX can test those candidates against historical data, but cannot currently discover and validate arbitrary new strategy concepts without predefined feature/signal families.
- Daily monitoring and opportunity detection are not operational. Research must be invoked manually against local historical CSVs; dashboard timestamps describe the report/data, not market freshness. Rejected-rule reasons and research results can be viewed, but detected live opportunities and approved live strategies cannot be reported because neither workflow exists.

### Multiple-entry and portfolio-risk readiness

**Not ready for multiple simultaneous trades; do not enable them.** Safety configuration is unchanged: live trading is disabled, per-trade risk is 0.5%, and maximum positions is one (`/home/runner/work/apex/apex/config/settings.py:18-24`). The simulator explicitly rejects `max_open_positions != 1` (`/home/runner/work/apex/apex/research/backtest.py:91-114`).

| Required control | Current state |
|---|---|
| Read balance, equity, free margin, open positions, pending orders | Missing. No account/positions/orders API is implemented. |
| XAUUSDm stop-loss cash risk and broker-valid size | Partial theory only: `/home/runner/work/apex/apex/risk/risk_manager/__init__.py:7-70` sizes from caller-supplied balance, risk fraction and value-per-price-unit. No broker tick/contract/currency specifications or quote conversion; no min/max/step rounding. |
| Per-trade and aggregate risk budgets | Per-trade cap only; no aggregate, pending-order reservation or portfolio exposure accounting. |
| Correlation, overlapping setups and existing exposure | Missing. No correlation model, setup deduplication or account exposure inventory. |
| Drawdown and available-margin limits | Missing at execution. Backtest drawdown is closed-trade-only; live equity drawdown and margin are unavailable and unmodelled. |
| Immediate pre-execution recomputation and budget rejection/reduction | Missing. There is no execution path or order preflight. |
| Broker volume, margin and execution constraints | Missing. The simulator explicitly omits lot rounding, broker minimum volume, swaps and margin (`/home/runner/work/apex/apex/research/backtest.py:24-33`). |
| Partial fills, rejects, duplicates, connection loss, uncertain status | Missing. No order lifecycle or reconciliation code. |
| Restart-safe trade/risk persistence | Missing. Reports persist; positions, order intents, exposure and risk reservations do not. |

A future, disabled-by-default design should use a broker adapter that supplies a timestamped account snapshot and live symbol specifications; size from tick value/size and contract/currency conversion, then round volume down to a valid step and reject if below minimum or if the resulting loss exceeds budget. Before submission, reserve risk for open positions and pending intents, include conservative costs and stop-gap allowance, and enforce per-entry, aggregate stop-risk, drawdown and free-margin limits. Treat correlated/same-underlying positions conservatively, deduplicate overlapping setups, and fail closed on stale/missing snapshots. Recompute the complete budget immediately before execution. Persist idempotent order intents and lifecycle state; reconcile uncertain submissions before retrying and explicitly handle partial fill/reject/cancel states. A first multi-entry trial should not raise total risk above the current one-entry baseline unless separately approved.

Required tests before considering such a change: fake-account/spec calculations for long/short stops and currency conversion; min/max/step volume rounding and below-minimum rejection; open and pending exposure plus aggregate-budget limits; correlated/duplicate setup policy; drawdown and free-margin rejects; stale snapshot and last-moment risk-change fail-closed behavior; margin/stop-level/execution rejects; idempotency and reconciliation for timeout/uncertain status; partial fills and rejected orders; restart persistence/recovery; and regression tests proving the default remains live-disabled, 0.5% risk and one position. All broker interactions must be simulated with fakes in unit tests.

Compare one-position and future multi-position configurations on identical signals, data windows, chronological folds and out-of-sample periods, with the same risk budget and realistic measured spread, slippage, commission, swaps, gaps and margin. Report expectancy and profit factor alongside trade count, exposure, equity/intratrade drawdown, time-under-water and margin utilization; include uncertainty estimates and leave the final OOS period untouched. A higher return alone is not evidence of improvement.

## Next highest-value step
Profile and optimize the full-year offline research suite so evidence iteration is practical. Acceptance criteria:
using the unchanged 70,588-candle input and default configuration, produce the same deterministic candidate/event
counts, folds, selections, and reported OOS metrics as the preserved baseline archives; add a focused regression
test for those parity invariants; and complete one measured full-year run in 60 minutes or less. This is research
only: do not connect to MT5, alter the historical data, enable live trading, increase risk or raise the one-position
limit. Afterward, prioritize measured execution-cost data and independent review before any trading-readiness claims.
