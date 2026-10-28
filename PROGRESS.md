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

## Next highest-value step
Profile and optimize the full-year liquidity/structure study, preserving its event counts and OOS metrics under
determinism tests; the observed 2.6-3.5 hour suite runtime makes evidence iteration impractical. Then improve
execution-cost evidence with M1 and measured spread data, and formalise session/timezone, key levels, MSS/CHOCH
and FVG/IFVG to complete the owner's candidate. A persistent simulated-position/per-trade store and continuous
market-data monitor remain future work. Broker execution must remain disabled until strategy evidence, execution
realism, operational safeguards and human review are independently established.
