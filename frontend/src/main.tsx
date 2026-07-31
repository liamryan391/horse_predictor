import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BookOpen,
  Database,
  Home,
  RefreshCw,
  ShieldCheck,
  Trophy
} from "lucide-react";
import "./styles.css";

type Page = "home" | "about" | "race-lab";

type DatabaseSummary = {
  environment: string;
  engine: string;
  driver: string;
  database: string | null;
};

type Summary = {
  database: DatabaseSummary;
  historicalRuns: number;
  currentRunners: number;
  lastRefresh: string | null;
};

type Prediction = {
  race_date: string | null;
  track: string | null;
  horse: string | null;
  jockey: string | null;
  trainer: string | null;
  odds: number | null;
  model_odds: number | null;
  win_probability: number | null;
  implied_probability: number | null;
  value_edge: number | null;
  suggested_rank: number | null;
};

type ModelStatus = {
  trainingRows: number;
  winnerRate: number;
  trainingStart: string | null;
  trainingEnd: string | null;
  featureCount: number;
  features: string[];
  historicalRows: number;
  evaluation: ModelEvaluation;
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
  jockey: TrendRow[];
  trainer: TrendRow[];
  owner: TrendRow[];
};

const navigation: { page: Page; label: string; icon: typeof Home }[] = [
  { page: "home", label: "Home", icon: Home },
  { page: "about", label: "About Us", icon: BookOpen },
  { page: "race-lab", label: "Race Lab", icon: BarChart3 }
];

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_PREFIX = "/api/v1";

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
  const [page, setPage] = useState<Page>("home");

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setPage("home")} aria-label="Go to home">
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
      </header>

      <main>
        {page === "home" && <HomePage onOpenTool={() => setPage("race-lab")} />}
        {page === "about" && <AboutPage />}
        {page === "race-lab" && <RaceLab />}
      </main>
    </div>
  );
}

function HomePage({ onOpenTool }: { onOpenTool: () => void }) {
  return (
    <section className="home-grid">
      <div className="hero-copy">
        <div className="eyebrow">MySQL-ready racing model</div>
        <h1>Professional race-card scoring for sharper pre-race decisions.</h1>
        <p>
          Horse Predictor turns API-fed racing data into ranked runners, model odds, value edges,
          and stable trend signals. The model retrains from the latest historical results in the
          database, so better data can support stronger model review.
        </p>
        <div className="hero-actions">
          <button className="primary-action" onClick={onOpenTool}>
            <BarChart3 size={18} />
            Open Race Lab
          </button>
          <span className="status-pill">
            <Database size={16} />
            API to SQL to model
          </span>
        </div>
      </div>
      <div className="signal-panel">
        <div className="panel-header">
          <Trophy size={22} />
          <span>Today&apos;s decision stack</span>
        </div>
        <div className="signal-list">
          <div>
            <strong>1. Ingest</strong>
            <span>Racecards and results update the SQL tables.</span>
          </div>
          <div>
            <strong>2. Learn</strong>
            <span>The model retrains from historical winners and losers.</span>
          </div>
          <div>
            <strong>3. Rank</strong>
            <span>Upcoming runners are scored by probability and value edge.</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function AboutPage() {
  return (
    <section className="content-band">
      <div className="section-heading">
        <div className="eyebrow">About Us</div>
        <h1>Built for disciplined racing analysis.</h1>
        <p>
          The product is designed for users who want structured evidence before making racing
          decisions. It is not a betting guarantee; it is an intelligence layer that rewards clean
          data, repeatable process, and sober model review.
        </p>
      </div>
      <div className="about-grid">
        <article>
          <ShieldCheck size={24} />
          <h2>Data first</h2>
          <p>API ingestion writes to SQL, giving the system a reliable history to learn from.</p>
        </article>
        <article>
          <Activity size={24} />
          <h2>Evidence review</h2>
          <p>Verified results create a deeper holdout record for model evaluation.</p>
        </article>
        <article>
          <Database size={24} />
          <h2>Production path</h2>
          <p>MySQL is the chosen database target for hosting, scheduled refreshes, and scaling.</p>
        </article>
      </div>
    </section>
  );
}

function RaceLab() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [trends, setTrends] = useState<Trends | null>(null);
  const [track, setTrack] = useState("All tracks");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [summaryResult, predictionResult, modelResult, trendResult] = await Promise.all([
        apiGet<Summary>("/summary"),
        apiGet<{ predictions: Prediction[] }>("/predictions"),
        apiGet<ModelStatus>("/model"),
        apiGet<Trends>("/trends")
      ]);
      setSummary(summaryResult);
      setPredictions(predictionResult.predictions);
      setModel(modelResult);
      setTrends(trendResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load racing data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadData();
  }, []);

  const tracks = useMemo(() => {
    const unique = new Set(predictions.map((item) => item.track).filter(Boolean) as string[]);
    return ["All tracks", ...Array.from(unique).sort()];
  }, [predictions]);

  const filtered = track === "All tracks" ? predictions : predictions.filter((item) => item.track === track);

  return (
    <section className="tool-page">
      <div className="tool-header">
        <div>
          <div className="eyebrow">Race Lab</div>
          <h1>Live model rankings</h1>
          <p>Review predicted win probability, market-implied probability, and value edge.</p>
        </div>
        <button className="icon-action" onClick={() => void loadData()} disabled={loading} title="Refresh data">
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="metric-row">
        <Metric label="Historical runs" value={summary?.historicalRuns.toLocaleString() ?? "-"} />
        <Metric label="Current runners" value={summary?.currentRunners.toLocaleString() ?? "-"} />
        <Metric label="Training rows" value={model?.trainingRows.toLocaleString() ?? "-"} />
        <Metric label="Winner rate" value={formatPercent(model?.winnerRate)} />
        <Metric label="Top-pick holdout" value={formatPercent(model?.evaluation.metrics.top_pick_win_rate)} />
        <Metric label="Holdout Brier" value={formatNumber(model?.evaluation.metrics.runner_brier_score, 3)} />
      </div>

      <div className="toolbar">
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
        <span>Last refresh: {summary?.lastRefresh ?? "Waiting for ingestion"}</span>
      </div>

      {model?.evaluation && <EvaluationPanel evaluation={model.evaluation} />}

      <PredictionTable rows={filtered} loading={loading} />

      <div className="trend-grid">
        <TrendTable title="Jockey form" rows={trends?.jockey ?? []} labelKey="jockey" />
        <TrendTable title="Trainer form" rows={trends?.trainer ?? []} labelKey="trainer" />
        <TrendTable title="Owner form" rows={trends?.owner ?? []} labelKey="owner" />
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

function EvaluationPanel({ evaluation }: { evaluation: ModelEvaluation }) {
  const metrics = evaluation.metrics;

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
        <Metric label="Training races" value={formatInteger(evaluation.trainingRaces)} />
        <Metric label="Validation races" value={formatInteger(evaluation.validationRaces)} />
        <Metric label="Runner log loss" value={formatNumber(metrics.runner_log_loss, 3)} />
        <Metric label="Market log loss" value={formatNumber(metrics.market_log_loss, 3)} />
        <Metric label="Calibration gap" value={formatPercent(metrics.calibration_mae)} />
        <Metric label="Market top pick" value={formatPercent(metrics.market_top_pick_win_rate)} />
        <Metric label="Mean winner rank" value={formatNumber(metrics.mean_winner_rank, 2)} />
        <Metric label="Fixed-stake bets" value={formatInteger(metrics.fixed_stake_bets)} />
        <Metric label="Fixed-stake profit" value={formatSignedNumber(metrics.fixed_stake_profit)} />
        <Metric label="Fixed-stake ROI" value={formatPercent(metrics.fixed_stake_roi)} />
      </div>
    </section>
  );
}

function PredictionTable({ rows, loading }: { rows: Prediction[]; loading: boolean }) {
  return (
    <div className="table-wrap">
      <table>
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
          {loading && (
            <tr>
              <td colSpan={9}>Loading predictions...</td>
            </tr>
          )}
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
              <td colSpan={9}>No race-card rows found.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function TrendTable({ title, rows, labelKey }: { title: string; rows: TrendRow[]; labelKey: keyof TrendRow }) {
  return (
    <article className="trend-table">
      <h2>{title}</h2>
      {rows.slice(0, 6).map((row) => (
        <div className="trend-row" key={String(row[labelKey])}>
          <span>{String(row[labelKey] ?? "-")}</span>
          <strong>{formatPercent(row.win_rate)}</strong>
        </div>
      ))}
    </article>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
