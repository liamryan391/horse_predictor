import { StrictMode, useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BookOpen,
  CalendarDays,
  ClipboardList,
  Database,
  ExternalLink,
  Moon,
  RefreshCw,
  Scale,
  ShieldCheck,
  Sun,
  Trophy,
  Trash2,
  UserRound,
  WalletCards,
  type LucideIcon
} from "lucide-react";
import {
  betProfit,
  createBet,
  parseStoredBets,
  summarizeBets,
  type Bet,
  type BetStatus
} from "./journal";
import "./styles.css";

type Page = "workspace" | "journal" | "methodology" | "responsible-use" | "contact";
type WorkspaceView = "rankings" | "race-card" | "evaluation" | "trends";

type DatabaseSummary = {
  environment: string;
  engine: string;
  driver: string;
  database: string | null;
};

type PageMeta = {
  limit: number;
  offset: number;
  returned: number;
  total: number;
};

type Summary = {
  requestId: string;
  database: DatabaseSummary;
  historicalRuns: number;
  currentRunners: number;
  lastRefresh: string | null;
  dataFreshness: DataFreshness;
};

type DataFreshness = {
  status: string;
  lastRefresh: string | null;
  ageHours: number | null;
  maxAgeHours: number;
};

type RaceRunner = {
  race_date: string | null;
  track: string | null;
  distance: number | null;
  surface: string | null;
  horse: string | null;
  jockey: string | null;
  owner: string | null;
  trainer: string | null;
  odds: number | null;
  draw: number | null;
  speed_rating: number | null;
  class_rating: number | null;
  weather: string | null;
};

type Prediction = RaceRunner & {
  model_odds: number | null;
  win_probability: number | null;
  implied_probability: number | null;
  value_edge: number | null;
  suggested_rank: number | null;
  field_size: number | null;
  odds_rank: number | null;
  relative_speed_rating: number | null;
  relative_class_rating: number | null;
};

type Meeting = {
  race_date: string | null;
  track: string | null;
  races: number;
  runners: number;
  first_distance: number | null;
  last_distance: number | null;
};

type ModelStatus = {
  requestId: string;
  modelVersionId: number | null;
  servingMode: string;
  artifactUri: string | null;
  trainingRows: number;
  winnerRate: number;
  trainingStart: string | null;
  trainingEnd: string | null;
  featureCount: number;
  features: string[];
  historicalRows: number;
  evaluation: ModelEvaluation;
};

type ModelRegistryRow = {
  id: number;
  name: string;
  algorithm: string;
  status: string;
  featureCount: number;
  trainingStart: string | null;
  trainingEnd: string | null;
  artifactUri: string | null;
  artifactSha256: string | null;
  featureSchemaHash: string | null;
  codeCommitSha: string | null;
  artifactReady: boolean;
  createdAt: string | null;
  updatedAt: string | null;
  metrics: EvaluationMetrics;
};

type PredictionRun = {
  id: number;
  modelVersionId: number | null;
  raceId: number | null;
  runAt: string | null;
  source: string;
  notes: string | null;
  runnerCount: number;
  topRunner: string | null;
  topWinProbability: number | null;
  topValueEdge: number | null;
};

type EvaluationMetrics = {
  runner_log_loss?: number | null;
  runner_brier_score?: number | null;
  market_log_loss?: number | null;
  market_brier_score?: number | null;
  calibration_mae?: number | null;
  top_pick_win_rate?: number | null;
  market_top_pick_win_rate?: number | null;
  mean_winner_rank?: number | null;
  market_mean_winner_rank?: number | null;
  fixed_stake_bets?: number | null;
  fixed_stake_profit?: number | null;
  fixed_stake_roi?: number | null;
};

type ModelEvaluation = {
  status: string;
  message: string | null;
  trainingRows: number;
  validationRows: number;
  trainingRaces: number;
  validationRaces: number;
  evaluationStart: string | null;
  evaluationEnd: string | null;
  metrics: EvaluationMetrics;
  leakageFeatures: string[];
};

type TrendRow = {
  jockey?: string;
  trainer?: string;
  owner?: string;
  runs: number;
  wins: number;
  avg_odds: number | null;
  win_rate: number | null;
};

type Trends = {
  requestId: string;
  jockey: TrendRow[];
  trainer: TrendRow[];
  owner: TrendRow[];
};

type ProductSafeguards = {
  requestId: string;
  responsibleUseNotice: string;
  limitations: string[];
  dataLicensingNotice: string;
  privacyNotice: string;
  termsNotice: string;
  links: Record<string, string | null>;
};

const navigation: { page: Page; label: string; icon: LucideIcon }[] = [
  { page: "workspace", label: "Workspace", icon: BarChart3 },
  { page: "journal", label: "Bet Journal", icon: WalletCards },
  { page: "methodology", label: "Methodology", icon: BookOpen },
  { page: "responsible-use", label: "Responsible Use", icon: ShieldCheck },
  { page: "contact", label: "Contact", icon: UserRound }
];

const workspaceViews: { view: WorkspaceView; label: string; icon: LucideIcon }[] = [
  { view: "rankings", label: "Rankings", icon: Trophy },
  { view: "race-card", label: "Race Card", icon: ClipboardList },
  { view: "evaluation", label: "Evaluation", icon: ShieldCheck },
  { view: "trends", label: "Trends", icon: Activity }
];

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_PREFIX = "/api/v1";
const TRACK_FILTER_KEY = "horse-predictor-track-filter";
const THEME_KEY = "horse-predictor-theme";
const BETS_KEY = "horse-predictor-bets";
const DEFAULT_SAFEGUARDS: ProductSafeguards = {
  requestId: "-",
  responsibleUseNotice: "Horse Predictor is decision-support software, not betting advice or a guaranteed-return system.",
  limitations: [
    "Predictions depend on provider coverage, data freshness, and historical data quality.",
    "Holdout metrics can drift when tracks, fields, weather, or provider schemas change.",
    "Value edges are model estimates and should be reviewed alongside bankroll controls and market context."
  ],
  dataLicensingNotice: "Only display race-card, odds, and result data that the operator is licensed to use.",
  privacyNotice:
    "The current bet journal stores entries in browser local storage; do not enter personal data until account storage and a published privacy policy are configured.",
  termsNotice: "Production launch requires published terms of use and no guaranteed-profit claims in product or marketing copy.",
  links: {}
};

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

function formatInteger(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Math.round(value).toLocaleString();
}

function formatSignedNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const formatted = value.toFixed(digits);
  return value > 0 ? `+${formatted}` : formatted;
}

function formatDate(value: string | null | undefined) {
  if (!value) return "-";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString();
}

function formatFreshness(value: DataFreshness | null | undefined) {
  if (!value) return "-";
  if (value.status === "fresh" && value.ageHours !== null) return `Fresh (${formatNumber(value.ageHours, 1)}h)`;
  if (value.status === "stale" && value.ageHours !== null) return `Stale (${formatNumber(value.ageHours, 1)}h)`;
  return value.status;
}

function localStorageValue(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`);
  if (!response.ok) {
    const detail = await response.text();
    let message = detail || `Request failed: ${response.status}`;
    try {
      const payload = JSON.parse(detail) as { error?: { detail?: string; requestId?: string } };
      const requestId = payload.error?.requestId ? ` (${payload.error.requestId})` : "";
      message = `${payload.error?.detail ?? `Request failed: ${response.status}`}${requestId}`;
    } catch {
      // Keep the raw response body if the server did not return the standard error envelope.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function Logo() {
  return (
    <div className="brand-mark" aria-label="Horse Predictor logo">
      <svg viewBox="0 0 64 64" role="img" aria-hidden="true">
        <path d="M12 44c10 0 15-6 20-15 3-5 7-9 14-10l5 5-5 3 4 8-8-2-8 15H20l6-10c-4 4-8 6-14 6Z" />
        <path d="M18 23h16M13 31h13M9 38h12" />
      </svg>
    </div>
  );
}

function App() {
  const [page, setPage] = useState<Page>("workspace");
  const [theme, setTheme] = useState(() => localStorageValue(THEME_KEY, "light"));

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setPage("workspace")} aria-label="Open workspace">
          <Logo />
          <span>
            <strong>Horse Predictor</strong>
            <small>Racing Intelligence</small>
          </span>
        </button>
        <nav className="nav-tabs" aria-label="Primary navigation">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.page}
                className={page === item.page ? "active" : ""}
                onClick={() => setPage(item.page)}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>
        <button
          className="icon-action square-action"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Use light theme" : "Use dark theme"}
          aria-label={theme === "dark" ? "Use light theme" : "Use dark theme"}
        >
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </header>

      <main>
        {page === "workspace" && <Workspace />}
        {page === "journal" && <BetJournal />}
        {page === "methodology" && <MethodologyPage />}
        {page === "responsible-use" && <ResponsibleUsePage />}
        {page === "contact" && <ContactPage />}
      </main>
    </div>
  );
}

function Workspace() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [raceCard, setRaceCard] = useState<RaceRunner[]>([]);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [modelRegistry, setModelRegistry] = useState<ModelRegistryRow[]>([]);
  const [predictionRuns, setPredictionRuns] = useState<PredictionRun[]>([]);
  const [trends, setTrends] = useState<Trends | null>(null);
  const [track, setTrack] = useState(() => localStorageValue(TRACK_FILTER_KEY, "All tracks"));
  const [search, setSearch] = useState("");
  const [view, setView] = useState<WorkspaceView>("rankings");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [summaryResult, predictionResult, modelResult, modelRegistryResult, predictionRunsResult, trendResult, meetingResult, raceCardResult] =
        await Promise.all([
          apiGet<Summary>("/summary"),
          apiGet<{ predictions: Prediction[]; page: PageMeta }>("/predictions?limit=500"),
          apiGet<ModelStatus>("/model"),
          apiGet<{ models: ModelRegistryRow[]; page: PageMeta }>("/model/registry?limit=10"),
          apiGet<{ runs: PredictionRun[]; page: PageMeta }>("/prediction-runs?limit=5"),
          apiGet<Trends>("/trends"),
          apiGet<{ meetings: Meeting[]; page: PageMeta }>("/meetings?limit=200"),
          apiGet<{ raceCard: RaceRunner[]; page: PageMeta }>("/race-card?limit=500")
        ]);
      setSummary(summaryResult);
      setPredictions(predictionResult.predictions);
      setModel(modelResult);
      setModelRegistry(modelRegistryResult.models);
      setPredictionRuns(predictionRunsResult.runs);
      setTrends(trendResult);
      setMeetings(meetingResult.meetings);
      setRaceCard(raceCardResult.raceCard);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load racing data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadData();
  }, []);

  useEffect(() => {
    localStorage.setItem(TRACK_FILTER_KEY, track);
  }, [track]);

  const tracks = useMemo(() => {
    const unique = new Set([...predictions, ...raceCard].map((item) => item.track).filter(Boolean) as string[]);
    return ["All tracks", ...Array.from(unique).sort()];
  }, [predictions, raceCard]);

  const filteredPredictions = useMemo(() => {
    return predictions.filter((item) => {
      const matchesTrack = track === "All tracks" || item.track === track;
      const query = search.trim().toLowerCase();
      const matchesSearch =
        !query ||
        [item.horse, item.jockey, item.trainer, item.owner].some((value) => value?.toLowerCase().includes(query));
      return matchesTrack && matchesSearch;
    });
  }, [predictions, search, track]);

  const filteredRaceCard = useMemo(() => {
    return raceCard.filter((item) => {
      const matchesTrack = track === "All tracks" || item.track === track;
      const query = search.trim().toLowerCase();
      const matchesSearch =
        !query ||
        [item.horse, item.jockey, item.trainer, item.owner].some((value) => value?.toLowerCase().includes(query));
      return matchesTrack && matchesSearch;
    });
  }, [raceCard, search, track]);

  const filteredMeetings = useMemo(() => {
    return meetings.filter((item) => track === "All tracks" || item.track === track);
  }, [meetings, track]);

  const topRunner = filteredPredictions[0];
  const secondRunner = filteredPredictions[1];
  const dataQuality = model?.evaluation.status === "ok" ? "Holdout active" : "Needs history";

  return (
    <section className="workspace">
      <div className="workspace-header">
        <div>
          <div className="eyebrow">Race Workspace</div>
          <h1>Today&apos;s racing desk</h1>
          <p>Rankings, race cards, model checks, and betting records in one operating view.</p>
        </div>
        <button className="icon-action" onClick={() => void loadData()} disabled={loading} title="Refresh data">
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <strong>API error</strong>
          <span>{error}</span>
          <button className="icon-action" onClick={() => void loadData()}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      )}

      <div className="metric-row">
        <Metric label="Historical runs" value={summary?.historicalRuns.toLocaleString() ?? "-"} />
        <Metric label="Current runners" value={summary?.currentRunners.toLocaleString() ?? "-"} />
        <Metric label="Meetings" value={formatInteger(filteredMeetings.length)} />
        <Metric label="Top-pick holdout" value={formatPercent(model?.evaluation.metrics.top_pick_win_rate)} />
        <Metric label="Data quality" value={dataQuality} />
        <Metric label="Freshness" value={formatFreshness(summary?.dataFreshness)} />
        <Metric label="Last refresh" value={formatDate(summary?.lastRefresh)} />
      </div>

      <div className="workspace-layout">
        <aside className="control-panel">
          <div className="panel-title">
            <Database size={18} />
            <span>Controls</span>
          </div>
          <label>
            Track
            <select value={track} onChange={(event) => setTrack(event.target.value)}>
              {tracks.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Runner search
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Horse, jockey, trainer" />
          </label>
          <div className="segmented-control" aria-label="Workspace view">
            {workspaceViews.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.view}
                  className={view === item.view ? "active" : ""}
                  onClick={() => setView(item.view)}
                >
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </aside>

        <section className="workspace-main">
          <MeetingStrip meetings={filteredMeetings} loading={loading} />
          <RunnerComparison first={topRunner} second={secondRunner} />
          {view === "rankings" && <PredictionTable rows={filteredPredictions} loading={loading} />}
          {view === "race-card" && <RaceCardTable rows={filteredRaceCard} loading={loading} />}
          {view === "evaluation" && model?.evaluation && (
            <>
              <EvaluationPanel evaluation={model.evaluation} model={model} />
              <ModelRegistryTable rows={modelRegistry} />
              <PredictionRunTable rows={predictionRuns} />
            </>
          )}
          {view === "trends" && <TrendGrid trends={trends} />}
        </section>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MeetingStrip({ meetings, loading }: { meetings: Meeting[]; loading: boolean }) {
  if (loading) {
    return (
      <div className="meeting-strip">
        {[0, 1, 2].map((item) => (
          <div className="meeting-tile skeleton" key={item} />
        ))}
      </div>
    );
  }

  return (
    <div className="meeting-strip">
      {meetings.length === 0 && <EmptyState title="No meetings found" detail="Change the track filter or refresh race data." />}
      {meetings.map((meeting) => (
        <article className="meeting-tile" key={`${meeting.race_date}-${meeting.track}`}>
          <CalendarDays size={18} />
          <div>
            <strong>{meeting.track ?? "-"}</strong>
            <span>
              {formatDate(meeting.race_date)} / {meeting.runners} runners / {meeting.races} races
            </span>
          </div>
        </article>
      ))}
    </div>
  );
}

function RunnerComparison({ first, second }: { first?: Prediction; second?: Prediction }) {
  return (
    <section className="comparison-panel">
      <div className="panel-title">
        <Scale size={18} />
        <span>Runner comparison</span>
      </div>
      <div className="comparison-grid">
        {[first, second].map((runner, index) => (
          <article className="runner-card" key={runner?.horse ?? index}>
            <span>Rank {runner?.suggested_rank ?? index + 1}</span>
            <strong>{runner?.horse ?? "-"}</strong>
            <dl>
              <div>
                <dt>Win probability</dt>
                <dd>{formatPercent(runner?.win_probability)}</dd>
              </div>
              <div>
                <dt>Value edge</dt>
                <dd className={(runner?.value_edge ?? 0) >= 0 ? "positive" : "negative"}>
                  {formatPercent(runner?.value_edge)}
                </dd>
              </div>
              <div>
                <dt>Model odds</dt>
                <dd>{formatNumber(runner?.model_odds)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}

function EvaluationPanel({ evaluation, model }: { evaluation: ModelEvaluation; model: ModelStatus }) {
  const metrics = evaluation.metrics;
  const calibration = Math.min(Math.max(metrics.calibration_mae ?? 0, 0), 1);

  return (
    <section className="evaluation-panel">
      <div className="evaluation-header">
        <ShieldCheck size={20} />
        <div>
          <h2>Model Evaluation</h2>
          <span>
            {evaluation.status === "ok"
              ? `${evaluation.evaluationStart ?? "-"} to ${evaluation.evaluationEnd ?? "-"}`
              : evaluation.message ?? "Waiting for enough historical races"}
          </span>
        </div>
      </div>
      <div className="evaluation-grid">
        <Metric label="Training rows" value={formatInteger(model.trainingRows)} />
        <Metric label="Features" value={formatInteger(model.featureCount)} />
        <Metric label="Serving" value={model.servingMode === "artifact" ? `Artifact #${model.modelVersionId ?? "-"}` : "In-memory"} />
        <Metric label="Runner log loss" value={formatNumber(metrics.runner_log_loss, 3)} />
        <Metric label="Market log loss" value={formatNumber(metrics.market_log_loss, 3)} />
        <Metric label="Calibration gap" value={formatPercent(metrics.calibration_mae)} />
        <Metric label="Market top pick" value={formatPercent(metrics.market_top_pick_win_rate)} />
        <Metric label="Mean winner rank" value={formatNumber(metrics.mean_winner_rank, 2)} />
        <Metric label="Fixed-stake ROI" value={formatPercent(metrics.fixed_stake_roi)} />
      </div>
      <div className="calibration-bar" aria-label="Calibration gap">
        <span style={{ width: `${calibration * 100}%` }} />
      </div>
    </section>
  );
}

function ModelRegistryTable({ rows }: { rows: ModelRegistryRow[] }) {
  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <Database size={20} />
        <div>
          <h2>Model Registry</h2>
          <span>Persisted candidate and approved evaluation snapshots.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Artifact</th>
              <th>Name</th>
              <th>Training window</th>
              <th>Top-pick holdout</th>
              <th>Market top pick</th>
              <th>Brier score</th>
              <th>Recorded</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <span className={`status-pill ${row.status}`}>{row.status}</span>
                </td>
                <td>
                  <span className={`status-pill ${row.artifactReady ? "approved" : "superseded"}`}>
                    {row.artifactReady ? "Ready" : "Missing"}
                  </span>
                  <span className="table-subtext">{row.artifactSha256 ? `sha ${row.artifactSha256.slice(0, 8)}` : "-"}</span>
                </td>
                <td>
                  <strong>{row.name}</strong>
                  <span className="table-subtext">{row.algorithm} / {row.featureCount} features</span>
                </td>
                <td>
                  {formatDate(row.trainingStart)} to {formatDate(row.trainingEnd)}
                </td>
                <td>{formatPercent(row.metrics.top_pick_win_rate)}</td>
                <td>{formatPercent(row.metrics.market_top_pick_win_rate)}</td>
                <td>{formatNumber(row.metrics.runner_brier_score, 3)}</td>
                <td>{formatDate(row.createdAt)}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={8}>No persisted model snapshots yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PredictionRunTable({ rows }: { rows: PredictionRun[] }) {
  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <ClipboardList size={20} />
        <div>
          <h2>Prediction Runs</h2>
          <span>Persisted scoring snapshots for current race cards.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Recorded</th>
              <th>Model</th>
              <th>Runners</th>
              <th>Top runner</th>
              <th>Win probability</th>
              <th>Value edge</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <strong>#{row.id}</strong>
                  <span className="table-subtext">{row.source}</span>
                </td>
                <td>{formatDate(row.runAt)}</td>
                <td>{row.modelVersionId ? `#${row.modelVersionId}` : "Unlinked"}</td>
                <td>{formatInteger(row.runnerCount)}</td>
                <td>{row.topRunner ?? "-"}</td>
                <td>{formatPercent(row.topWinProbability)}</td>
                <td className={(row.topValueEdge ?? 0) >= 0 ? "positive" : "negative"}>
                  {formatPercent(row.topValueEdge)}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7}>No persisted prediction runs yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PredictionTable({ rows, loading }: { rows: Prediction[]; loading: boolean }) {
  return (
    <DataTable loading={loading} emptyText="No ranked runners found.">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Runner</th>
          <th>Track</th>
          <th>Jockey</th>
          <th>Trainer</th>
          <th>Market odds</th>
          <th>Model odds</th>
          <th>Win probability</th>
          <th>Value edge</th>
        </tr>
      </thead>
      <tbody>
        {!loading &&
          rows.map((row) => (
            <tr key={`${row.race_date}-${row.track}-${row.horse}`}>
              <td>{row.suggested_rank ?? "-"}</td>
              <td>
                <strong>{row.horse ?? "-"}</strong>
              </td>
              <td>{row.track ?? "-"}</td>
              <td>{row.jockey ?? "-"}</td>
              <td>{row.trainer ?? "-"}</td>
              <td>{formatNumber(row.odds)}</td>
              <td>{formatNumber(row.model_odds)}</td>
              <td>{formatPercent(row.win_probability)}</td>
              <td className={(row.value_edge ?? 0) >= 0 ? "positive" : "negative"}>
                {formatPercent(row.value_edge)}
              </td>
            </tr>
          ))}
        {!loading && rows.length === 0 && (
          <tr>
            <td colSpan={9}>No ranked runners found.</td>
          </tr>
        )}
      </tbody>
    </DataTable>
  );
}

function RaceCardTable({ rows, loading }: { rows: RaceRunner[]; loading: boolean }) {
  return (
    <DataTable loading={loading} emptyText="No race-card rows found.">
      <thead>
        <tr>
          <th>Runner</th>
          <th>Track</th>
          <th>Date</th>
          <th>Surface</th>
          <th>Jockey</th>
          <th>Trainer</th>
          <th>Odds</th>
          <th>Draw</th>
          <th>Rating</th>
        </tr>
      </thead>
      <tbody>
        {!loading &&
          rows.map((row) => (
            <tr key={`${row.race_date}-${row.track}-${row.horse}`}>
              <td>
                <strong>{row.horse ?? "-"}</strong>
              </td>
              <td>{row.track ?? "-"}</td>
              <td>{formatDate(row.race_date)}</td>
              <td>{row.surface ?? "-"}</td>
              <td>{row.jockey ?? "-"}</td>
              <td>{row.trainer ?? "-"}</td>
              <td>{formatNumber(row.odds)}</td>
              <td>{formatNumber(row.draw, 0)}</td>
              <td>{formatNumber(row.speed_rating, 0)}</td>
            </tr>
          ))}
        {!loading && rows.length === 0 && (
          <tr>
            <td colSpan={9}>No race-card rows found.</td>
          </tr>
        )}
      </tbody>
    </DataTable>
  );
}

function DataTable({ children, loading, emptyText }: { children: ReactNode; loading: boolean; emptyText: string }) {
  return (
    <div className="table-wrap">
      <table>
        {loading ? (
          <tbody>
            {[0, 1, 2, 3].map((row) => (
              <tr key={row}>
                <td colSpan={9}>
                  <div className="table-skeleton">{emptyText}</div>
                </td>
              </tr>
            ))}
          </tbody>
        ) : (
          children
        )}
      </table>
    </div>
  );
}

function TrendGrid({ trends }: { trends: Trends | null }) {
  return (
    <div className="trend-grid">
      <TrendTable title="Jockey form" rows={trends?.jockey ?? []} labelKey="jockey" />
      <TrendTable title="Trainer form" rows={trends?.trainer ?? []} labelKey="trainer" />
      <TrendTable title="Owner form" rows={trends?.owner ?? []} labelKey="owner" />
    </div>
  );
}

function TrendTable({ title, rows, labelKey }: { title: string; rows: TrendRow[]; labelKey: keyof TrendRow }) {
  return (
    <article className="trend-table">
      <h2>{title}</h2>
      {rows.slice(0, 8).map((row) => (
        <div className="trend-row" key={String(row[labelKey])}>
          <span>{String(row[labelKey] ?? "-")}</span>
          <strong>{formatPercent(row.win_rate)}</strong>
        </div>
      ))}
      {rows.length === 0 && <EmptyState title="No trend rows" detail="Historical data will populate this table." />}
    </article>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function BetJournal() {
  const [bets, setBets] = useState<Bet[]>(() => parseStoredBets(localStorage.getItem(BETS_KEY)));
  const [draft, setDraft] = useState({
    horse: "",
    track: "",
    stake: "10",
    odds: "3.00",
    status: "open" as BetStatus
  });

  useEffect(() => {
    localStorage.setItem(BETS_KEY, JSON.stringify(bets));
  }, [bets]);

  const journal = useMemo(() => summarizeBets(bets), [bets]);

  function submitBet(event: FormEvent) {
    event.preventDefault();
    const nextBet = createBet(draft, crypto.randomUUID(), new Date().toISOString());
    if (!nextBet) {
      return;
    }
    setBets((current) => [nextBet, ...current]);
    setDraft({ horse: "", track: "", stake: "10", odds: "3.00", status: "open" });
  }

  function updateStatus(id: string, status: BetStatus) {
    setBets((current) => current.map((bet) => (bet.id === id ? { ...bet, status } : bet)));
  }

  function removeBet(id: string) {
    setBets((current) => current.filter((bet) => bet.id !== id));
  }

  return (
    <section className="journal-page">
      <div className="workspace-header">
        <div>
          <div className="eyebrow">Bet Journal</div>
          <h1>Position ledger</h1>
          <p>Track open and settled positions separately from model performance.</p>
        </div>
      </div>

      <div className="metric-row">
        <Metric label="Recorded bets" value={formatInteger(journal.count)} />
        <Metric label="Open bets" value={formatInteger(journal.open)} />
        <Metric label="Total staked" value={formatNumber(journal.staked)} />
        <Metric label="Settled profit" value={formatSignedNumber(journal.profit)} />
        <Metric label="Settled ROI" value={formatPercent(journal.roi)} />
      </div>

      <form className="journal-form" onSubmit={submitBet}>
        <label>
          Horse
          <input value={draft.horse} onChange={(event) => setDraft({ ...draft, horse: event.target.value })} />
        </label>
        <label>
          Track
          <input value={draft.track} onChange={(event) => setDraft({ ...draft, track: event.target.value })} />
        </label>
        <label>
          Stake
          <input type="number" min="0" step="0.01" value={draft.stake} onChange={(event) => setDraft({ ...draft, stake: event.target.value })} />
        </label>
        <label>
          Odds
          <input type="number" min="1.01" step="0.01" value={draft.odds} onChange={(event) => setDraft({ ...draft, odds: event.target.value })} />
        </label>
        <label>
          Status
          <select value={draft.status} onChange={(event) => setDraft({ ...draft, status: event.target.value as BetStatus })}>
            <option value="open">Open</option>
            <option value="won">Won</option>
            <option value="lost">Lost</option>
          </select>
        </label>
        <button className="primary-action" type="submit">
          <WalletCards size={18} />
          Add bet
        </button>
      </form>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Horse</th>
              <th>Track</th>
              <th>Stake</th>
              <th>Odds</th>
              <th>Status</th>
              <th>Profit</th>
              <th>Date</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {bets.map((bet) => (
              <tr key={bet.id}>
                <td>
                  <strong>{bet.horse}</strong>
                </td>
                <td>{bet.track || "-"}</td>
                <td>{formatNumber(bet.stake)}</td>
                <td>{formatNumber(bet.odds)}</td>
                <td>
                  <select value={bet.status} onChange={(event) => updateStatus(bet.id, event.target.value as BetStatus)}>
                    <option value="open">Open</option>
                    <option value="won">Won</option>
                    <option value="lost">Lost</option>
                  </select>
                </td>
                <td className={betProfit(bet) >= 0 ? "positive" : "negative"}>{bet.status === "open" ? "-" : formatSignedNumber(betProfit(bet))}</td>
                <td>{formatDate(bet.createdAt)}</td>
                <td>
                  <button
                    className="icon-action square-action danger-action"
                    onClick={() => removeBet(bet.id)}
                    title="Remove bet"
                    aria-label={`Remove ${bet.horse} from bet journal`}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
            {bets.length === 0 && (
              <tr>
                <td colSpan={8}>No journal rows yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MethodologyPage() {
  return (
    <InfoPage
      eyebrow="Methodology"
      title="Model review before race decisions"
      intro="The current model uses historical result rows, pre-race features, chronological holdout evaluation, and market baselines."
      items={[
        ["Inputs", "Race-card attributes, market odds, ratings, field size, and entity form summaries."],
        ["Validation", "The latest chronological races are held out from training for model-quality checks."],
        ["Outputs", "Rank, win probability, model odds, value edge, and evaluation metrics stay visible together."]
      ]}
    />
  );
}

function ResponsibleUsePage() {
  const [safeguards, setSafeguards] = useState<ProductSafeguards>(DEFAULT_SAFEGUARDS);

  useEffect(() => {
    apiGet<ProductSafeguards>("/safeguards")
      .then(setSafeguards)
      .catch(() => setSafeguards(DEFAULT_SAFEGUARDS));
  }, []);

  const links = Object.entries(safeguards.links).filter((entry): entry is [string, string] => Boolean(entry[1]));

  return (
    <section className="content-band">
      <div className="section-heading">
        <div className="eyebrow">Responsible Use</div>
        <h1>Decision support, not certainty</h1>
        <p>{safeguards.responsibleUseNotice}</p>
      </div>
      <div className="info-grid safeguard-grid">
        <article>
          <h2>Model limits</h2>
          <ul className="notice-list">
            {safeguards.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
        <article>
          <h2>Data licensing</h2>
          <p>{safeguards.dataLicensingNotice}</p>
        </article>
        <article>
          <h2>Privacy</h2>
          <p>{safeguards.privacyNotice}</p>
        </article>
        <article>
          <h2>Terms</h2>
          <p>{safeguards.termsNotice}</p>
        </article>
        <article>
          <h2>Bankroll</h2>
          <p>Use stake sizes that stay inside a pre-defined budget and pause when data quality or holdout metrics degrade.</p>
        </article>
        <article>
          <h2>Policy links</h2>
          {links.length === 0 ? (
            <p>Policy URLs must be configured before production launch.</p>
          ) : (
            <div className="link-list">
              {links.map(([key, url]) => (
                <a href={url} key={key} target="_blank" rel="noreferrer">
                  <ExternalLink size={16} />
                  {formatPolicyLabel(key)}
                </a>
              ))}
            </div>
          )}
        </article>
      </div>
    </section>
  );
}

function formatPolicyLabel(value: string) {
  return value
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (match) => match.toUpperCase())
    .trim();
}

function ContactPage() {
  return (
    <InfoPage
      eyebrow="Contact"
      title="Support and operations"
      intro="The platform now exposes request IDs in API responses and headers so support work can start from a concrete trace."
      items={[
        ["API docs", "Open /docs on the local backend for the generated FastAPI reference."],
        ["Issue reports", "Include the API request ID, browser page, filter state, and race date."],
        ["Operations", "Provider ingestion, model checks, and frontend release work stay on the development branch before main."]
      ]}
    />
  );
}

function InfoPage({
  eyebrow,
  title,
  intro,
  items
}: {
  eyebrow: string;
  title: string;
  intro: string;
  items: [string, string][];
}) {
  return (
    <section className="content-band">
      <div className="section-heading">
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{intro}</p>
      </div>
      <div className="info-grid">
        {items.map(([label, detail]) => (
          <article key={label}>
            <h2>{label}</h2>
            <p>{detail}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
