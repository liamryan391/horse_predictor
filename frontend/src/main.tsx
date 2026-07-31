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

type Summary = {
  database: string;
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
  featureCount: number;
  features: string[];
  historicalRows: number;
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

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
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
          database, so better data directly improves the tool.
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
          <h2>Self-improving</h2>
          <p>As more verified results arrive, the model can retrain with a deeper evidence base.</p>
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
        apiGet<Summary>("/api/summary"),
        apiGet<{ predictions: Prediction[] }>("/api/predictions"),
        apiGet<ModelStatus>("/api/model"),
        apiGet<Trends>("/api/trends")
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
