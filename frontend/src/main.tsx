import { StrictMode, useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Archive,
  BarChart3,
  BookOpen,
  CalendarDays,
  ClipboardList,
  Database,
  ExternalLink,
  Flag,
  KeyRound,
  LockKeyhole,
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
  betPayloadFromDraft,
  createBet,
  parseStoredBets,
  summarizeBets,
  type Bet,
  type BetStatus
} from "./journal";
import {
  buildRaceGroups,
  raceGroupKey,
  raceKey,
  runnerSignals,
  sortRaceRunners,
  type RaceCentreGroup,
  type RaceCentreSort
} from "./raceCentre";
import "./styles.css";

type Page = "workspace" | "journal" | "admin" | "methodology" | "responsible-use" | "contact";
type WorkspaceView = "rankings" | "race-centre" | "race-card" | "evaluation" | "monitoring" | "trends";

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
  country: string | null;
  distance_bucket: string | null;
  going_category: string | null;
  race_type: string | null;
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

type RaceDayRace = {
  raceDate: string | null;
  track: string | null;
  distance: number | null;
  surface: string | null;
  offTime: string | null;
  raceStatus: string;
  statusLabel: string;
  minutesToPost: number | null;
  runners: number;
  topRunner: string | null;
  topWinProbability: number | null;
  topValueEdge: number | null;
  marketFavorite: string | null;
  averageOdds: number | null;
  provider: string | null;
  lastIngestedAt: string | null;
  dataAgeHours: number | null;
};

type RaceDay = {
  requestId: string;
  asOf: string;
  today: string;
  timezone: string;
  nextRace: RaceDayRace | null;
  races: RaceDayRace[];
  page: PageMeta;
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

type ModelSnapshotResponse = {
  requestId: string;
  model: ModelRegistryRow;
};

type PredictionRunResponse = {
  requestId: string;
  run: PredictionRun;
  entries: Prediction[];
  page: PageMeta;
};

type SeedSampleResponse = {
  requestId: string;
  seeded: Record<string, number>;
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

type DataQualityIssue = {
  severity: string;
  rowNumber: number | null;
  field: string;
  message: string;
};

type FieldCoverage = {
  field: string;
  total: number;
  nonMissing: number;
  coverage: number;
};

type DataQualityTable = {
  tableName: string;
  rows: number;
  issueCount: number;
  issues: DataQualityIssue[];
  coverage: FieldCoverage[];
};

type ProviderFreshness = {
  provider: string;
  tableName: string;
  status: string;
  rowCount: number;
  completedAt: string | null;
  message: string | null;
};

type DataQuality = {
  requestId: string;
  tables: DataQualityTable[];
  providerFreshness: ProviderFreshness[];
};

type MonitoringMetric = {
  name: string;
  label: string;
  value: number | string | null;
  unit: string | null;
  status: string;
  description: string | null;
};

type MonitoringAlert = {
  severity: string;
  category: string;
  code: string;
  message: string;
  value: number | string | null;
  threshold: number | string | null;
};

type DriftMetricRow = {
  field: string;
  kind: string;
  status: string;
  score: number | null;
  referenceCount: number;
  currentCount: number;
  referenceMean: number | null;
  currentMean: number | null;
  referenceMissingRate: number | null;
  currentMissingRate: number | null;
  referenceShare: number | null;
  currentShare: number | null;
  topReferenceCategory: string | null;
  topCurrentCategory: string | null;
  maxShareDelta: number | null;
  newCategories: string[];
  detail: string | null;
};

type DriftReport = {
  status: string;
  generatedAt: string;
  referenceRows: number;
  currentRows: number;
  thresholds: Record<string, number>;
  featureDrift: DriftMetricRow[];
  predictionDrift: DriftMetricRow[];
};

type ApiMetrics = {
  totalRequests: number;
  errorRequests: number;
  slowRequests: number;
  errorRate: number;
  averageLatencyMs: number;
  statusCounts: Record<string, number>;
  topPaths: { path: string; requests: number }[];
  recent: { method: string; path: string; statusCode: number; elapsedMs: number; recordedAt: string }[];
  lastErrorAt: string | null;
  lastSlowAt: string | null;
};

type Monitoring = {
  requestId: string;
  status: string;
  generatedAt: string;
  dataFreshness: DataFreshness;
  providerFreshness: ProviderFreshness[];
  apiMetrics: ApiMetrics;
  metrics: MonitoringMetric[];
  alerts: MonitoringAlert[];
  drift: DriftReport;
  model: Record<string, string | number | boolean | null>;
};

type BetJournalListResponse = {
  requestId: string;
  bets: Bet[];
  page: PageMeta;
};

type BetJournalEntryResponse = {
  requestId: string;
  bet: Bet;
};

type BetJournalDeleteResponse = {
  requestId: string;
  id: number;
  deleted: boolean;
};

type Readiness = {
  requestId: string;
  status: string;
  databaseReady: boolean;
  modelReady: boolean;
  counts: Record<string, number>;
  message: string | null;
};

type IngestionRow = {
  table_name: string | null;
  source: string | null;
  row_count: number | null;
  status: string | null;
  message: string | null;
  ingested_at: string | null;
};

type AdminSession = {
  requestId: string;
  actor: string;
  roles: string[];
  environment: string;
  adminAuthRequired: boolean;
  journalAuthRequired: boolean;
  adminTokenConfigured: boolean;
  journalTokenConfigured: boolean;
};

type AdminAuditEvent = {
  id: number;
  actor: string;
  roles: string[];
  action: string;
  resourceType: string;
  resourceId: string | null;
  requestId: string | null;
  status: string;
  detail: string | null;
  payload: Record<string, unknown> | null;
  createdAt: string | null;
};

type AdminGovernance = {
  requestId: string;
  session: AdminSession;
  readiness: Readiness;
  summary: Summary;
  monitoring: Monitoring;
  ingestion: IngestionRow[];
  models: ModelRegistryRow[];
  predictionRuns: PredictionRun[];
  auditEvents: AdminAuditEvent[];
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
  { page: "admin", label: "Admin", icon: LockKeyhole },
  { page: "methodology", label: "Methodology", icon: BookOpen },
  { page: "responsible-use", label: "Responsible Use", icon: ShieldCheck },
  { page: "contact", label: "Contact", icon: UserRound }
];

const workspaceViews: { view: WorkspaceView; label: string; icon: LucideIcon }[] = [
  { view: "rankings", label: "Rankings", icon: Trophy },
  { view: "race-centre", label: "Race Centre", icon: Flag },
  { view: "race-card", label: "Race Card", icon: ClipboardList },
  { view: "evaluation", label: "Evaluation", icon: ShieldCheck },
  { view: "monitoring", label: "Monitoring", icon: Activity },
  { view: "trends", label: "Trends", icon: BarChart3 }
];

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_PREFIX = "/api/v1";
const TRACK_FILTER_KEY = "horse-predictor-track-filter";
const THEME_KEY = "horse-predictor-theme";
const BETS_KEY = "horse-predictor-bets";
const ACCESS_TOKEN_KEY = "horse-predictor-access-token";
const ACCESS_ACTOR_KEY = "horse-predictor-access-actor";
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
    "The bet journal stores operator-local records in the configured server database; do not enter personal data until account auth, export/delete controls, and a published privacy policy are configured.",
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

function formatDateTime(value: string | null | undefined) {
  if (!value) return "-";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatFreshness(value: DataFreshness | null | undefined) {
  if (!value) return "-";
  if (value.status === "fresh" && value.ageHours !== null) return `Fresh (${formatNumber(value.ageHours, 1)}h)`;
  if (value.status === "stale" && value.ageHours !== null) return `Stale (${formatNumber(value.ageHours, 1)}h)`;
  return value.status;
}

function formatOffTime(value: string | null | undefined) {
  if (!value) return "-";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatMinutesToPost(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value >= 60) return `${formatNumber(value / 60, 1)}h`;
  if (value >= 0) return `${Math.round(value)}m`;
  if (value > -60) return `Off ${Math.abs(Math.round(value))}m ago`;
  return `${formatNumber(Math.abs(value) / 60, 1)}h ago`;
}

function formatPostWindow(offTime: string | null | undefined, minutesToPost: number | null | undefined) {
  const timeLabel = formatOffTime(offTime);
  const windowLabel = formatMinutesToPost(minutesToPost);
  if (timeLabel === "-" && windowLabel === "-") return "-";
  if (timeLabel === "-") return windowLabel;
  if (windowLabel === "-") return timeLabel;
  return `${timeLabel} / ${windowLabel}`;
}

function formatDataAge(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value < 1) return `${Math.max(1, Math.round(value * 60))}m`;
  return `${formatNumber(value, 1)}h`;
}

function statusTone(status: string | null | undefined) {
  if (!status) return "superseded";
  if (["ok", "fresh", "success", "approved"].includes(status)) return "approved";
  if (["warning", "stale", "candidate", "degraded", "unknown"].includes(status)) return "candidate";
  if (["critical", "blocked", "failed", "missing", "error"].includes(status)) return "critical";
  return "superseded";
}

function raceStatusTone(status: string | null | undefined) {
  if (status === "live") return "live";
  if (["next", "race-day", "upcoming"].includes(status ?? "")) return "candidate";
  if (["stale", "complete"].includes(status ?? "")) return "superseded";
  if (status === "unknown") return "warning";
  return "superseded";
}

function formatMetricValue(metric: MonitoringMetric) {
  if (typeof metric.value === "number") {
    if (metric.unit === "percent") return formatPercent(metric.value);
    if (metric.unit === "ms") return `${formatNumber(metric.value, 1)}ms`;
    if (metric.unit === "hours") return `${formatNumber(metric.value, 1)}h`;
    if (metric.unit === "count") return formatInteger(metric.value);
    return formatNumber(metric.value, 2);
  }
  return metric.value === null || metric.value === undefined ? "-" : String(metric.value);
}

function localStorageValue(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

type ApiAuth = {
  token?: string;
  actor?: string;
};

function storedAccessAuth(): ApiAuth {
  return {
    token: localStorageValue(ACCESS_TOKEN_KEY, ""),
    actor: localStorageValue(ACCESS_ACTOR_KEY, "")
  };
}

async function apiGet<T>(path: string, auth?: ApiAuth): Promise<T> {
  return apiRequest<T>(path, {}, auth);
}

async function apiPost<T>(path: string, body?: unknown, auth?: ApiAuth): Promise<T> {
  return apiRequest<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }, auth);
}

async function apiPatch<T>(path: string, body: unknown, auth?: ApiAuth): Promise<T> {
  return apiRequest<T>(path, { method: "PATCH", body: JSON.stringify(body) }, auth);
}

async function apiDelete<T>(path: string, auth?: ApiAuth): Promise<T> {
  return apiRequest<T>(path, { method: "DELETE" }, auth);
}

async function apiRequest<T>(path: string, init: RequestInit = {}, auth: ApiAuth = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const token = auth.token?.trim();
  const actor = auth.actor?.trim();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (actor) {
    headers.set("X-Admin-Actor", actor);
    headers.set("X-Journal-Actor", actor);
  }
  const response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, { ...init, headers });
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
        {page === "admin" && <AdminConsole />}
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
  const [raceDay, setRaceDay] = useState<RaceDay | null>(null);
  const [raceCard, setRaceCard] = useState<RaceRunner[]>([]);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [modelRegistry, setModelRegistry] = useState<ModelRegistryRow[]>([]);
  const [predictionRuns, setPredictionRuns] = useState<PredictionRun[]>([]);
  const [trends, setTrends] = useState<Trends | null>(null);
  const [quality, setQuality] = useState<DataQuality | null>(null);
  const [monitoring, setMonitoring] = useState<Monitoring | null>(null);
  const [track, setTrack] = useState(() => localStorageValue(TRACK_FILTER_KEY, "All tracks"));
  const [search, setSearch] = useState("");
  const [view, setView] = useState<WorkspaceView>("rankings");
  const [selectedRaceId, setSelectedRaceId] = useState<string | null>(null);
  const [raceSort, setRaceSort] = useState<RaceCentreSort>("rank");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [
        summaryResult,
        predictionResult,
        modelResult,
        modelRegistryResult,
        predictionRunsResult,
        trendResult,
        meetingResult,
        raceDayResult,
        raceCardResult,
        qualityResult,
        monitoringResult
      ] =
        await Promise.all([
          apiGet<Summary>("/summary"),
          apiGet<{ predictions: Prediction[]; page: PageMeta }>("/predictions?limit=500"),
          apiGet<ModelStatus>("/model"),
          apiGet<{ models: ModelRegistryRow[]; page: PageMeta }>("/model/registry?limit=10"),
          apiGet<{ runs: PredictionRun[]; page: PageMeta }>("/prediction-runs?limit=5"),
          apiGet<Trends>("/trends"),
          apiGet<{ meetings: Meeting[]; page: PageMeta }>("/meetings?limit=200"),
          apiGet<RaceDay>("/race-day?limit=500"),
          apiGet<{ raceCard: RaceRunner[]; page: PageMeta }>("/race-card?limit=500"),
          apiGet<DataQuality>("/data-quality"),
          apiGet<Monitoring>("/monitoring")
        ]);
      setSummary(summaryResult);
      setPredictions(predictionResult.predictions);
      setModel(modelResult);
      setModelRegistry(modelRegistryResult.models);
      setPredictionRuns(predictionRunsResult.runs);
      setTrends(trendResult);
      setMeetings(meetingResult.meetings);
      setRaceDay(raceDayResult);
      setRaceCard(raceCardResult.raceCard);
      setQuality(qualityResult);
      setMonitoring(monitoringResult);
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

  const raceGroups = useMemo(() => {
    const raceDayById = new Map(
      (raceDay?.races ?? []).map((race) => [raceGroupKey(race.raceDate, race.track, race.distance), race])
    );
    return buildRaceGroups(filteredPredictions).map((group) => {
      const operationalRace = raceDayById.get(group.id);
      return {
        ...group,
        offTime: operationalRace?.offTime ?? group.offTime,
        raceStatus: operationalRace?.raceStatus ?? group.raceStatus,
        statusLabel: operationalRace?.statusLabel ?? group.statusLabel,
        minutesToPost: operationalRace?.minutesToPost ?? group.minutesToPost,
        provider: operationalRace?.provider ?? group.provider,
        lastIngestedAt: operationalRace?.lastIngestedAt ?? group.lastIngestedAt,
        dataAgeHours: operationalRace?.dataAgeHours ?? group.dataAgeHours,
        marketFavorite: operationalRace?.marketFavorite ?? group.marketFavorite,
        topValueEdge: operationalRace?.topValueEdge ?? group.topValueEdge
      };
    });
  }, [filteredPredictions, raceDay]);

  useEffect(() => {
    if (raceGroups.length === 0) {
      setSelectedRaceId(null);
      return;
    }
    if (!selectedRaceId || !raceGroups.some((race) => race.id === selectedRaceId)) {
      setSelectedRaceId(raceGroups[0].id);
    }
  }, [raceGroups, selectedRaceId]);

  const topRunner = filteredPredictions[0];
  const secondRunner = filteredPredictions[1];
  const qualityIssueCount = quality?.tables.reduce((total, table) => total + table.issueCount, 0);
  const dataQuality = qualityIssueCount === undefined ? "Loading" : qualityIssueCount === 0 ? "Clear" : `${qualityIssueCount} issues`;
  const monitoringSummary = monitoring ? `${monitoring.status} / ${monitoring.alerts.length} alerts` : "Loading";

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
        <Metric label="Monitoring" value={monitoringSummary} />
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
          {view !== "race-centre" && <RunnerComparison first={topRunner} second={secondRunner} />}
          {view === "rankings" && <PredictionTable rows={filteredPredictions} loading={loading} />}
          {view === "race-centre" && (
            <>
              <RaceDayOverview raceDay={raceDay} groups={raceGroups} loading={loading} />
              <RaceCentrePanel
                groups={raceGroups}
                rows={filteredPredictions}
                selectedRaceId={selectedRaceId}
                onSelectRace={setSelectedRaceId}
                sort={raceSort}
                onSortChange={setRaceSort}
                loading={loading}
              />
            </>
          )}
          {view === "race-card" && <RaceCardTable rows={filteredRaceCard} loading={loading} />}
          {view === "evaluation" && model?.evaluation && (
            <>
              <EvaluationPanel evaluation={model.evaluation} model={model} />
              <ModelRegistryTable rows={modelRegistry} />
              <PredictionRunTable rows={predictionRuns} />
              <DataQualityPanel quality={quality} />
            </>
          )}
          {view === "monitoring" && <MonitoringPanel monitoring={monitoring} />}
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

const raceSortOptions: { value: RaceCentreSort; label: string }[] = [
  { value: "rank", label: "Model rank" },
  { value: "value", label: "Value edge" },
  { value: "market", label: "Market odds" },
  { value: "draw", label: "Draw" },
  { value: "speed", label: "Speed rating" }
];

function formatDistance(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${formatNumber(value, 0)}y`;
}

function RaceDayOverview({
  raceDay,
  groups,
  loading
}: {
  raceDay: RaceDay | null;
  groups: RaceCentreGroup[];
  loading: boolean;
}) {
  const nextRace =
    groups.find((race) => race.raceStatus === "live") ??
    groups.find((race) => race.raceStatus === "next") ??
    groups.find((race) => ["race-day", "upcoming"].includes(race.raceStatus ?? "")) ??
    null;
  const statusCounts = ["live", "next", "race-day", "upcoming", "complete", "stale"]
    .map((status) => ({
      status,
      count: groups.filter((race) => race.raceStatus === status).length
    }))
    .filter((item) => item.count > 0);

  if (loading) {
    return (
      <section className="race-day-panel">
        <div className="race-day-header skeleton" />
        <div className="race-day-grid">
          {[0, 1, 2, 3].map((item) => (
            <div className="race-day-card skeleton" key={item} />
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="race-day-panel">
      <div className="race-day-header">
        <div className="evaluation-header">
          <Flag size={20} />
          <div>
            <h2>Race-Day Board</h2>
            <span>{raceDay ? `As of ${formatDateTime(raceDay.asOf)} ${raceDay.timezone}` : "No race-day snapshot"}</span>
          </div>
        </div>
        <span className={`race-state ${raceStatusTone(nextRace?.raceStatus)}`}>
          {nextRace?.statusLabel ?? "No active card"}
        </span>
      </div>

      <div className="race-day-grid">
        <article className="race-day-card">
          <span>Next race</span>
          <strong>{nextRace ? `${nextRace.track ?? "-"} ${formatDistance(nextRace.distance)}` : "-"}</strong>
          <small>{nextRace ? `${formatDate(nextRace.raceDate)} / ${nextRace.runners} runners` : "No visible race"}</small>
        </article>
        <article className="race-day-card">
          <span>Post window</span>
          <strong>{formatPostWindow(nextRace?.offTime, nextRace?.minutesToPost)}</strong>
          <small>{nextRace?.surface ?? nextRace?.raceType?.replace("_", " ") ?? "-"}</small>
        </article>
        <article className="race-day-card">
          <span>Provider</span>
          <strong>{nextRace?.provider ?? "-"}</strong>
          <small>Data age {formatDataAge(nextRace?.dataAgeHours)}</small>
        </article>
        <article className="race-day-card">
          <span>Top runner</span>
          <strong>{nextRace?.topRunner ?? "-"}</strong>
          <small>
            {formatPercent(nextRace?.topWinProbability)} win / {formatPercent(nextRace?.topValueEdge)} edge
          </small>
        </article>
      </div>

      <div className="race-day-status-strip" aria-label="Race-day status counts">
        {statusCounts.length === 0 && <span className="race-state superseded">No status rows</span>}
        {statusCounts.map((item) => (
          <span className={`race-state ${raceStatusTone(item.status)}`} key={item.status}>
            {item.status.replace("-", " ")} {item.count}
          </span>
        ))}
      </div>
    </section>
  );
}

function RaceCentrePanel({
  groups,
  rows,
  selectedRaceId,
  onSelectRace,
  sort,
  onSortChange,
  loading
}: {
  groups: RaceCentreGroup[];
  rows: Prediction[];
  selectedRaceId: string | null;
  onSelectRace: (raceId: string) => void;
  sort: RaceCentreSort;
  onSortChange: (sort: RaceCentreSort) => void;
  loading: boolean;
}) {
  const selectedRace = groups.find((race) => race.id === selectedRaceId) ?? groups[0];
  const selectedRows = selectedRace ? sortRaceRunners(rows.filter((row) => raceKey(row) === selectedRace.id), sort) : [];
  const focusRows = selectedRows.slice(0, 2);

  if (loading) {
    return (
      <section className="race-centre-layout">
        <div className="race-list">
          {[0, 1, 2].map((item) => (
            <div className="race-list-item skeleton" key={item} />
          ))}
        </div>
        <div className="race-detail-panel">
          <div className="race-detail-header skeleton" />
          <div className="table-wrap">
            <table>
              <tbody>
                {[0, 1, 2].map((item) => (
                  <tr key={item}>
                    <td colSpan={10}>
                      <div className="table-skeleton">Loading race centre</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="race-centre-layout">
      <aside className="race-list" aria-label="Race list">
        <div className="panel-title">
          <Flag size={18} />
          <span>Race Centre</span>
        </div>
        {groups.length === 0 && <EmptyState title="No races found" detail="Change the filters or refresh race data." />}
        {groups.map((race) => (
          <button
            key={race.id}
            className={`race-list-item ${race.id === selectedRace?.id ? "active" : ""}`}
            onClick={() => onSelectRace(race.id)}
          >
            <div className="race-list-top">
              <strong>{race.track ?? "-"}</strong>
              <span className={`race-state ${raceStatusTone(race.raceStatus)}`}>{race.statusLabel ?? "Card"}</span>
            </div>
            <span>
              {formatDate(race.raceDate)} / {formatDistance(race.distance)} / {formatPostWindow(race.offTime, race.minutesToPost)}
            </span>
            <small>{race.topRunner ? `${race.topRunner} ${formatPercent(race.topWinProbability)}` : "-"}</small>
          </button>
        ))}
      </aside>

      <div className="race-detail-panel">
        {!selectedRace && <EmptyState title="No race selected" detail="Select a race from the list." />}
        {selectedRace && (
          <>
            <div className="race-detail-header">
              <div className="evaluation-header">
                <Flag size={20} />
                <div>
                  <div className="race-title-row">
                    <h2>{selectedRace.track ?? "Race"}</h2>
                    <span className={`race-state ${raceStatusTone(selectedRace.raceStatus)}`}>
                      {selectedRace.statusLabel ?? "Card"}
                    </span>
                  </div>
                  <span>
                    {formatDate(selectedRace.raceDate)} / {formatDistance(selectedRace.distance)} / {selectedRace.surface ?? "-"} / {formatPostWindow(selectedRace.offTime, selectedRace.minutesToPost)}
                  </span>
                </div>
              </div>
              <label className="compact-control">
                Sort runners
                <select value={sort} onChange={(event) => onSortChange(event.target.value as RaceCentreSort)}>
                  {raceSortOptions.map((option) => (
                    <option value={option.value} key={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="race-summary-grid">
              <RaceSummaryMetric label="Runners" value={formatInteger(selectedRace.runners)} />
              <RaceSummaryMetric label="Avg odds" value={formatNumber(selectedRace.averageOdds)} />
              <RaceSummaryMetric label="Market fav" value={selectedRace.marketFavorite ?? "-"} />
              <RaceSummaryMetric label="Provider" value={selectedRace.provider ?? "-"} />
              <RaceSummaryMetric label="Data age" value={formatDataAge(selectedRace.dataAgeHours)} />
              <RaceSummaryMetric label="Last import" value={formatDateTime(selectedRace.lastIngestedAt)} />
              <RaceSummaryMetric label="Race type" value={selectedRace.raceType?.replace("_", " ") ?? "-"} />
              <RaceSummaryMetric label="Going" value={selectedRace.goingCategory?.replace("_", " ") ?? selectedRace.weather ?? "-"} />
              <RaceSummaryMetric label="Distance" value={selectedRace.distanceBucket ?? "-"} />
              <RaceSummaryMetric label="Country" value={selectedRace.country ?? "-"} />
            </div>

            <div className="race-focus-grid">
              {focusRows.map((runner) => (
                <RunnerFocusCard runner={runner} key={`${runner.race_date}-${runner.track}-${runner.horse}`} />
              ))}
              {focusRows.length === 0 && <EmptyState title="No runners found" detail="Change the filters or refresh race data." />}
            </div>

            <div className="table-wrap race-centre-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Runner</th>
                    <th>Jockey</th>
                    <th>Trainer</th>
                    <th>Market</th>
                    <th>Model</th>
                    <th>Win</th>
                    <th>Value</th>
                    <th>Draw</th>
                    <th>Signals</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRows.map((row) => {
                    const signals = runnerSignals(row);
                    return (
                      <tr key={`${row.race_date}-${row.track}-${row.horse}`}>
                        <td>{row.suggested_rank ?? "-"}</td>
                        <td>
                          <strong>{row.horse ?? "-"}</strong>
                          <span className="table-subtext">{row.owner ?? "-"}</span>
                        </td>
                        <td>{row.jockey ?? "-"}</td>
                        <td>{row.trainer ?? "-"}</td>
                        <td>{formatNumber(row.odds)}</td>
                        <td>{formatNumber(row.model_odds)}</td>
                        <td>{formatPercent(row.win_probability)}</td>
                        <td className={(row.value_edge ?? 0) >= 0 ? "positive" : "negative"}>{formatPercent(row.value_edge)}</td>
                        <td>{formatNumber(row.draw, 0)}</td>
                        <td>
                          <span className="table-subtext">{signals.slice(0, 2).join(" / ") || "-"}</span>
                        </td>
                      </tr>
                    );
                  })}
                  {selectedRows.length === 0 && (
                    <tr>
                      <td colSpan={10}>No runners found for this race.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function RaceSummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="race-summary-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RunnerFocusCard({ runner }: { runner: Prediction }) {
  const signals = runnerSignals(runner);

  return (
    <article className="runner-focus-card">
      <span>Rank {runner.suggested_rank ?? "-"}</span>
      <strong>{runner.horse ?? "-"}</strong>
      <dl>
        <div>
          <dt>Win probability</dt>
          <dd>{formatPercent(runner.win_probability)}</dd>
        </div>
        <div>
          <dt>Value edge</dt>
          <dd className={(runner.value_edge ?? 0) >= 0 ? "positive" : "negative"}>{formatPercent(runner.value_edge)}</dd>
        </div>
        <div>
          <dt>Market odds</dt>
          <dd>{formatNumber(runner.odds)}</dd>
        </div>
      </dl>
      <div className="signal-list">
        {signals.map((signal) => (
          <span className="signal-chip" key={signal}>
            {signal}
          </span>
        ))}
        {signals.length === 0 && <span className="signal-chip">No standout signal</span>}
      </div>
    </article>
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

function coverageValue(rows: FieldCoverage[], field: string) {
  return rows.find((row) => row.field === field)?.coverage;
}

function DataQualityPanel({ quality }: { quality: DataQuality | null }) {
  const tables = quality?.tables ?? [];
  const freshness = quality?.providerFreshness ?? [];

  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <Activity size={20} />
        <div>
          <h2>Data Quality</h2>
          <span>Provider freshness and enrichment coverage for model-serving inputs.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Table</th>
              <th>Rows</th>
              <th>Issues</th>
              <th>Country</th>
              <th>Distance bucket</th>
              <th>Going</th>
              <th>Race type</th>
            </tr>
          </thead>
          <tbody>
            {tables.map((row) => (
              <tr key={row.tableName}>
                <td>
                  <strong>{row.tableName}</strong>
                </td>
                <td>{formatInteger(row.rows)}</td>
                <td>{formatInteger(row.issueCount)}</td>
                <td>{formatPercent(coverageValue(row.coverage, "country"))}</td>
                <td>{formatPercent(coverageValue(row.coverage, "distance_bucket"))}</td>
                <td>{formatPercent(coverageValue(row.coverage, "going_category"))}</td>
                <td>{formatPercent(coverageValue(row.coverage, "race_type"))}</td>
              </tr>
            ))}
            {tables.length === 0 && (
              <tr>
                <td colSpan={7}>No data-quality snapshot available.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="table-wrap registry-table-wrap compact-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Provider</th>
              <th>Table</th>
              <th>Status</th>
              <th>Rows</th>
              <th>Completed</th>
            </tr>
          </thead>
          <tbody>
            {freshness.map((row) => (
              <tr key={`${row.provider}-${row.tableName}`}>
                <td>{row.provider}</td>
                <td>{row.tableName}</td>
                <td>
                  <span className={`status-pill ${row.status === "success" ? "approved" : "superseded"}`}>
                    {row.status}
                  </span>
                </td>
                <td>{formatInteger(row.rowCount)}</td>
                <td>{formatDate(row.completedAt)}</td>
              </tr>
            ))}
            {freshness.length === 0 && (
              <tr>
                <td colSpan={5}>No provider ingestion runs recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function driftSortValue(row: DriftMetricRow) {
  const severityRank: Record<string, number> = { critical: 0, blocked: 1, warning: 2, ok: 3 };
  return severityRank[row.status] ?? 4;
}

function driftReferenceValue(row: DriftMetricRow) {
  if (row.kind === "categorical") {
    return `${row.topReferenceCategory ?? "-"} (${formatPercent(row.referenceShare)})`;
  }
  return formatNumber(row.referenceMean, 3);
}

function driftCurrentValue(row: DriftMetricRow) {
  if (row.kind === "categorical") {
    return `${row.topCurrentCategory ?? "-"} (${formatPercent(row.currentShare)})`;
  }
  return formatNumber(row.currentMean, 3);
}

function DriftTable({ title, rows }: { title: string; rows: DriftMetricRow[] }) {
  const sortedRows = [...rows].sort((a, b) => driftSortValue(a) - driftSortValue(b) || (b.score ?? 0) - (a.score ?? 0));

  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <Database size={20} />
        <div>
          <h2>{title}</h2>
          <span>{formatInteger(rows.length)} checks against the historical baseline.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap drift-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Field</th>
              <th>Status</th>
              <th>Score</th>
              <th>Reference</th>
              <th>Current</th>
              <th>Rows</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={`${title}-${row.kind}-${row.field}`}>
                <td>
                  <strong>{row.field}</strong>
                  <span className="table-subtext">{row.kind}</span>
                </td>
                <td>
                  <span className={`status-pill ${statusTone(row.status)}`}>{row.status}</span>
                </td>
                <td>{formatPercent(row.score)}</td>
                <td>{driftReferenceValue(row)}</td>
                <td>{driftCurrentValue(row)}</td>
                <td>
                  {formatInteger(row.referenceCount)} / {formatInteger(row.currentCount)}
                </td>
                <td>
                  <span className="table-subtext drift-detail">{row.detail ?? "-"}</span>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7}>No drift checks are available.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MonitoringPanel({ monitoring }: { monitoring: Monitoring | null }) {
  if (!monitoring) {
    return (
      <section className="registry-panel">
        <EmptyState title="Monitoring is loading" detail="Refresh the workspace if the snapshot does not appear." />
      </section>
    );
  }

  const statusCounts = Object.entries(monitoring.apiMetrics.statusCounts);

  return (
    <section className="monitoring-panel">
      <div className="evaluation-header">
        <Activity size={20} />
        <div>
          <h2>Monitoring</h2>
          <span>
            Generated {formatDateTime(monitoring.generatedAt)} / serving {String(monitoring.model.servingMode ?? "-")} / tracing {String(monitoring.model.tracingStatus ?? "-")}
          </span>
        </div>
        <span className={`status-pill ${statusTone(monitoring.status)}`}>{monitoring.status}</span>
      </div>

      <div className="evaluation-grid monitoring-metrics">
        {monitoring.metrics.map((metric) => (
          <div className="metric-tile monitoring-metric" key={metric.name}>
            <span>{metric.label}</span>
            <strong>{formatMetricValue(metric)}</strong>
            <small>{metric.description ?? metric.unit ?? "-"}</small>
            <span className={`status-pill ${statusTone(metric.status)}`}>{metric.status}</span>
          </div>
        ))}
      </div>

      <section className="registry-panel">
        <div className="evaluation-header">
          <ShieldCheck size={20} />
          <div>
            <h2>Operator Alerts</h2>
            <span>{formatInteger(monitoring.alerts.length)} active signals.</span>
          </div>
        </div>
        <div className="table-wrap registry-table-wrap alert-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Severity</th>
                <th>Category</th>
                <th>Signal</th>
                <th>Value</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {monitoring.alerts.map((alert) => (
                <tr key={`${alert.category}-${alert.code}`}>
                  <td>
                    <span className={`status-pill ${statusTone(alert.severity)}`}>{alert.severity}</span>
                  </td>
                  <td>{alert.category}</td>
                  <td>
                    <strong>{alert.message}</strong>
                    <span className="table-subtext">{alert.code}</span>
                  </td>
                  <td>{typeof alert.value === "number" ? formatNumber(alert.value, 3) : alert.value ?? "-"}</td>
                  <td>{typeof alert.threshold === "number" ? formatNumber(alert.threshold, 3) : alert.threshold ?? "-"}</td>
                </tr>
              ))}
              {monitoring.alerts.length === 0 && (
                <tr>
                  <td colSpan={5}>No monitoring alerts are active.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <DriftTable title="Feature Drift" rows={monitoring.drift.featureDrift} />
      <DriftTable title="Prediction Drift" rows={monitoring.drift.predictionDrift} />

      <section className="registry-panel">
        <div className="evaluation-header">
          <Database size={20} />
          <div>
            <h2>API Traffic</h2>
            <span>
              {formatInteger(monitoring.apiMetrics.totalRequests)} requests / {formatPercent(monitoring.apiMetrics.errorRate)} error rate / {formatNumber(monitoring.apiMetrics.averageLatencyMs, 1)}ms average.
            </span>
          </div>
        </div>
        <div className="monitoring-api-grid">
          <div className="table-wrap compact-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Requests</th>
                </tr>
              </thead>
              <tbody>
                {statusCounts.map(([status, requests]) => (
                  <tr key={status}>
                    <td>{status}</td>
                    <td>{formatInteger(requests)}</td>
                  </tr>
                ))}
                {statusCounts.length === 0 && (
                  <tr>
                    <td colSpan={2}>No API requests recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="table-wrap compact-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Path</th>
                  <th>Requests</th>
                </tr>
              </thead>
              <tbody>
                {monitoring.apiMetrics.topPaths.map((row) => (
                  <tr key={row.path}>
                    <td>{row.path}</td>
                    <td>{formatInteger(row.requests)}</td>
                  </tr>
                ))}
                {monitoring.apiMetrics.topPaths.length === 0 && (
                  <tr>
                    <td colSpan={2}>No paths recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
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
  const [bets, setBets] = useState<Bet[]>(() => parseStoredBets(localStorageValue(BETS_KEY, "[]")));
  const [draft, setDraft] = useState({
    horse: "",
    track: "",
    raceDate: "",
    stake: "10",
    odds: "3.00",
    closingOdds: "",
    status: "open" as BetStatus,
    notes: ""
  });
  const [journalMode, setJournalMode] = useState<"server" | "local">("local");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadJournal();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(BETS_KEY, JSON.stringify(bets));
    } catch {
      // The server is authoritative once available; local storage is only a convenience cache.
    }
  }, [bets]);

  const journal = useMemo(() => summarizeBets(bets), [bets]);

  async function loadJournal() {
    setLoading(true);
    setError(null);
    try {
      const response = await apiGet<BetJournalListResponse>("/bet-journal?limit=200", storedAccessAuth());
      setBets(response.bets);
      setJournalMode("server");
    } catch (err) {
      setJournalMode("local");
      setBets((current) => (current.length > 0 ? current : parseStoredBets(localStorageValue(BETS_KEY, "[]"))));
      setError(err instanceof Error ? err.message : "Unable to load server bet journal.");
    } finally {
      setLoading(false);
    }
  }

  function addLocalBet() {
    const fallbackBet = createBet(draft, crypto.randomUUID(), new Date().toISOString());
    if (!fallbackBet) {
      return false;
    }
    setBets((current) => [fallbackBet, ...current]);
    return true;
  }

  async function submitBet(event: FormEvent) {
    event.preventDefault();
    const payload = betPayloadFromDraft(draft);
    if (!payload) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      if (journalMode === "server") {
        const response = await apiPost<BetJournalEntryResponse>("/bet-journal", payload, storedAccessAuth());
        setBets((current) => [response.bet, ...current.filter((bet) => bet.id !== response.bet.id)]);
      } else if (!addLocalBet()) {
        return;
      }
      setDraft({ horse: "", track: "", raceDate: "", stake: "10", odds: "3.00", closingOdds: "", status: "open", notes: "" });
    } catch (err) {
      if (addLocalBet()) {
        setJournalMode("local");
        setDraft({ horse: "", track: "", raceDate: "", stake: "10", odds: "3.00", closingOdds: "", status: "open", notes: "" });
      }
      setError(err instanceof Error ? `Saved locally while the server journal is unavailable: ${err.message}` : "Saved locally while the server journal is unavailable.");
    } finally {
      setSaving(false);
    }
  }

  async function updateStatus(id: Bet["id"], status: BetStatus) {
    if (journalMode === "server" && typeof id === "number") {
      setError(null);
      try {
        const response = await apiPatch<BetJournalEntryResponse>(`/bet-journal/${id}`, { status }, storedAccessAuth());
        setBets((current) => current.map((bet) => (bet.id === id ? response.bet : bet)));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to update bet status.");
      }
      return;
    }
    setBets((current) => current.map((bet) => (bet.id === id ? { ...bet, status, updatedAt: new Date().toISOString() } : bet)));
  }

  async function removeBet(id: Bet["id"]) {
    if (journalMode === "server" && typeof id === "number") {
      setError(null);
      try {
        await apiDelete<BetJournalDeleteResponse>(`/bet-journal/${id}`, storedAccessAuth());
        setBets((current) => current.filter((bet) => bet.id !== id));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to remove bet.");
      }
      return;
    }
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
        <button className="icon-action" onClick={() => void loadJournal()} disabled={loading} title="Refresh journal">
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <strong>{journalMode === "server" ? "Journal error" : "Local fallback"}</strong>
          <span>{error}</span>
          <button className="icon-action" onClick={() => void loadJournal()}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      )}

      <div className="metric-row">
        <Metric label="Recorded bets" value={formatInteger(journal.count)} />
        <Metric label="Open bets" value={formatInteger(journal.open)} />
        <Metric label="Total staked" value={formatNumber(journal.staked)} />
        <Metric label="Settled profit" value={formatSignedNumber(journal.profit)} />
        <Metric label="Settled ROI" value={formatPercent(journal.roi)} />
        <Metric label="Storage" value={journalMode === "server" ? "Server" : "Local"} />
      </div>

      <form className="journal-form" onSubmit={(event) => void submitBet(event)}>
        <label>
          Horse
          <input value={draft.horse} onChange={(event) => setDraft({ ...draft, horse: event.target.value })} />
        </label>
        <label>
          Track
          <input value={draft.track} onChange={(event) => setDraft({ ...draft, track: event.target.value })} />
        </label>
        <label>
          Race date
          <input type="date" value={draft.raceDate} onChange={(event) => setDraft({ ...draft, raceDate: event.target.value })} />
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
          Closing odds
          <input type="number" min="1.01" step="0.01" value={draft.closingOdds} onChange={(event) => setDraft({ ...draft, closingOdds: event.target.value })} />
        </label>
        <label>
          Status
          <select value={draft.status} onChange={(event) => setDraft({ ...draft, status: event.target.value as BetStatus })}>
            <option value="open">Open</option>
            <option value="won">Won</option>
            <option value="lost">Lost</option>
            <option value="void">Void</option>
          </select>
        </label>
        <label className="notes-field">
          Notes
          <input value={draft.notes} onChange={(event) => setDraft({ ...draft, notes: event.target.value })} />
        </label>
        <button className="primary-action" type="submit" disabled={saving}>
          <WalletCards size={18} />
          {saving ? "Saving" : "Add bet"}
        </button>
      </form>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Horse</th>
              <th>Track</th>
              <th>Race date</th>
              <th>Stake</th>
              <th>Odds</th>
              <th>Close</th>
              <th>Status</th>
              <th>Profit</th>
              <th>Date</th>
              <th>Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={11}>
                  <div className="table-skeleton">Loading journal rows.</div>
                </td>
              </tr>
            )}
            {!loading && bets.map((bet) => (
              <tr key={bet.id}>
                <td>
                  <strong>{bet.horse}</strong>
                </td>
                <td>{bet.track || "-"}</td>
                <td>{formatDate(bet.raceDate)}</td>
                <td>{formatNumber(bet.stake)}</td>
                <td>{formatNumber(bet.odds)}</td>
                <td>{formatNumber(bet.closingOdds)}</td>
                <td>
                  <select value={bet.status} onChange={(event) => void updateStatus(bet.id, event.target.value as BetStatus)}>
                    <option value="open">Open</option>
                    <option value="won">Won</option>
                    <option value="lost">Lost</option>
                    <option value="void">Void</option>
                  </select>
                </td>
                <td className={betProfit(bet) >= 0 ? "positive" : "negative"}>{bet.status === "open" ? "-" : formatSignedNumber(betProfit(bet))}</td>
                <td>{formatDate(bet.createdAt)}</td>
                <td>{bet.notes || "-"}</td>
                <td>
                  <button
                    className="icon-action square-action danger-action"
                    onClick={() => void removeBet(bet.id)}
                    title="Remove bet"
                    aria-label={`Remove ${bet.horse} from bet journal`}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
            {!loading && bets.length === 0 && (
              <tr>
                <td colSpan={11}>No journal rows yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function AdminConsole() {
  const [token, setToken] = useState(() => localStorageValue(ACCESS_TOKEN_KEY, ""));
  const [actor, setActor] = useState(() => localStorageValue(ACCESS_ACTOR_KEY, ""));
  const [session, setSession] = useState<AdminSession | null>(null);
  const [governance, setGovernance] = useState<AdminGovernance | null>(null);
  const [auditEvents, setAuditEvents] = useState<AdminAuditEvent[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const auth = useMemo(() => ({ token, actor }), [token, actor]);

  async function loadAdmin(nextAuth = auth) {
    setLoading(true);
    setError(null);
    try {
      const [sessionResult, governanceResult, auditResult] = await Promise.all([
        apiGet<AdminSession>("/admin/session", nextAuth),
        apiGet<AdminGovernance>("/admin/governance", nextAuth),
        apiGet<{ requestId: string; events: AdminAuditEvent[]; page: PageMeta }>("/admin/audit-log?limit=25", nextAuth)
      ]);
      setSession(sessionResult);
      setGovernance(governanceResult);
      setAuditEvents(auditResult.events);
      setSelectedModelId((current) => current || String(governanceResult.models[0]?.id ?? ""));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load admin governance data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadAdmin();
  }, []);

  function saveAccess(event: FormEvent) {
    event.preventDefault();
    localStorage.setItem(ACCESS_TOKEN_KEY, token.trim());
    localStorage.setItem(ACCESS_ACTOR_KEY, actor.trim());
    setNotice("Access details saved for this browser.");
    void loadAdmin({ token: token.trim(), actor: actor.trim() });
  }

  function clearAccess() {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(ACCESS_ACTOR_KEY);
    setToken("");
    setActor("");
    setNotice("Access details cleared.");
    void loadAdmin({ token: "", actor: "" });
  }

  async function runAdminAction(label: string, confirmText: string, action: () => Promise<unknown>) {
    if (!window.confirm(confirmText)) {
      return;
    }
    setBusyAction(label);
    setError(null);
    setNotice(null);
    try {
      const result = await action();
      const modelId = (result as { model?: { id?: number } }).model?.id;
      if (modelId) {
        setSelectedModelId(String(modelId));
      }
      setNotice(`${label} completed.`);
      await loadAdmin(auth);
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} failed.`);
    } finally {
      setBusyAction(null);
    }
  }

  const models = governance?.models ?? [];
  const latestModel = models[0];
  const selectedModel = models.find((model) => String(model.id) === selectedModelId);
  const canUseSelectedModel = Boolean(selectedModelId);
  const latestAudit = auditEvents[0];

  return (
    <section className="journal-page admin-page">
      <div className="workspace-header">
        <div>
          <div className="eyebrow">Admin Console</div>
          <h1>Governance desk</h1>
          <p>Controlled model operations, readiness checks, prediction snapshots, and audit history.</p>
        </div>
        <button className="icon-action" onClick={() => void loadAdmin()} disabled={loading} title="Refresh admin console">
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <strong>Admin error</strong>
          <span>{error}</span>
          <button className="icon-action" onClick={() => void loadAdmin()}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      )}

      {notice && (
        <div className="success-banner">
          <strong>Admin update</strong>
          <span>{notice}</span>
        </div>
      )}

      <form className="admin-token-panel" onSubmit={saveAccess}>
        <div className="panel-title">
          <KeyRound size={18} />
          <span>Access</span>
        </div>
        <label>
          Bearer token
          <input
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            autoComplete="off"
          />
        </label>
        <label>
          Actor
          <input value={actor} onChange={(event) => setActor(event.target.value)} placeholder="operator name" />
        </label>
        <div className="admin-token-actions">
          <button className="primary-action" type="submit">
            <KeyRound size={18} />
            Save
          </button>
          <button className="icon-action" type="button" onClick={clearAccess}>
            <Trash2 size={18} />
            Clear
          </button>
        </div>
      </form>

      <div className="metric-row">
        <Metric label="Environment" value={session?.environment ?? "-"} />
        <Metric label="Actor" value={session?.actor ?? "-"} />
        <Metric label="Roles" value={session?.roles.join(", ") ?? "-"} />
        <Metric label="Readiness" value={governance?.readiness.status ?? "-"} />
        <Metric label="Monitoring" value={governance ? `${governance.monitoring.status} / ${governance.monitoring.alerts.length} alerts` : "-"} />
        <Metric label="Models" value={formatInteger(models.length)} />
        <Metric label="Prediction runs" value={formatInteger(governance?.predictionRuns.length)} />
        <Metric label="Latest audit" value={latestAudit ? latestAudit.action : "-"} />
      </div>

      <section className="admin-grid">
        <div className="admin-action-panel">
          <div className="evaluation-header">
            <ShieldCheck size={20} />
            <div>
              <h2>Governed Actions</h2>
              <span>{selectedModel ? `Selected model #${selectedModel.id} / ${selectedModel.status}` : "Select or capture a model snapshot."}</span>
            </div>
          </div>

          <div className="admin-action-controls">
            <label>
              Model version
              <select value={selectedModelId} onChange={(event) => setSelectedModelId(event.target.value)}>
                <option value="">No model selected</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    #{model.id} / {model.status} / {formatDate(model.createdAt)}
                  </option>
                ))}
              </select>
            </label>
            <div className="admin-selected-model">
              <span className={`status-pill ${statusTone(selectedModel?.status)}`}>{selectedModel?.status ?? "none"}</span>
              <small>
                Artifact {selectedModel?.artifactReady ? "ready" : "missing"} / latest #{latestModel?.id ?? "-"}
              </small>
            </div>
          </div>

          <div className="admin-action-row">
            <button
              className="icon-action"
              type="button"
              disabled={Boolean(busyAction)}
              onClick={() =>
                void runAdminAction(
                  "Capture model snapshot",
                  "Capture a new model evaluation snapshot and artifact?",
                  () => apiPost<ModelSnapshotResponse>("/admin/model/evaluation", undefined, auth)
                )
              }
            >
              <Activity size={18} />
              Snapshot
            </button>
            <button
              className="icon-action"
              type="button"
              disabled={Boolean(busyAction) || !canUseSelectedModel}
              onClick={() =>
                void runAdminAction(
                  "Approve model",
                  `Approve model version #${selectedModelId} for artifact-backed serving?`,
                  () => apiPost<ModelSnapshotResponse>(`/admin/model/${selectedModelId}/approve`, undefined, auth)
                )
              }
            >
              <ShieldCheck size={18} />
              Approve
            </button>
            <button
              className="icon-action danger-action"
              type="button"
              disabled={Boolean(busyAction) || !canUseSelectedModel}
              onClick={() =>
                void runAdminAction(
                  "Supersede model",
                  `Supersede model version #${selectedModelId}?`,
                  () => apiPost<ModelSnapshotResponse>(`/admin/model/${selectedModelId}/supersede`, undefined, auth)
                )
              }
            >
              <Archive size={18} />
              Supersede
            </button>
            <button
              className="icon-action"
              type="button"
              disabled={Boolean(busyAction)}
              onClick={() =>
                void runAdminAction(
                  "Capture prediction run",
                  "Record a governed prediction snapshot using the approved model?",
                  () => apiPost<PredictionRunResponse>("/admin/prediction-runs?require_approved_model=true", undefined, auth)
                )
              }
            >
              <ClipboardList size={18} />
              Prediction Run
            </button>
            <button
              className="icon-action"
              type="button"
              disabled={Boolean(busyAction)}
              onClick={() =>
                void runAdminAction(
                  "Seed sample data",
                  "Seed the sample historical and current race tables?",
                  () => apiPost<SeedSampleResponse>("/admin/seed-sample", undefined, auth)
                )
              }
            >
              <Database size={18} />
              Seed
            </button>
          </div>
          {busyAction && <span className="table-subtext">{busyAction} is running.</span>}
        </div>

        <div className="admin-action-panel">
          <div className="evaluation-header">
            <Activity size={20} />
            <div>
              <h2>Readiness Snapshot</h2>
              <span>
                Data {governance?.readiness.databaseReady ? "ready" : "blocked"} / model {governance?.readiness.modelReady ? "ready" : "blocked"}
              </span>
            </div>
          </div>
          <div className="admin-readiness-grid">
            <Metric label="Historical rows" value={formatInteger(governance?.summary.historicalRuns)} />
            <Metric label="Current runners" value={formatInteger(governance?.summary.currentRunners)} />
            <Metric label="Freshness" value={formatFreshness(governance?.summary.dataFreshness)} />
            <Metric label="API error rate" value={formatPercent(governance?.monitoring.apiMetrics.errorRate)} />
          </div>
        </div>
      </section>

      <ModelRegistryTable rows={models} />
      <PredictionRunTable rows={governance?.predictionRuns ?? []} />
      <AdminIngestionTable rows={governance?.ingestion ?? []} loading={loading} />
      <AdminAuditTable rows={auditEvents.length ? auditEvents : governance?.auditEvents ?? []} loading={loading} />
    </section>
  );
}

function AdminIngestionTable({ rows, loading }: { rows: IngestionRow[]; loading: boolean }) {
  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <Database size={20} />
        <div>
          <h2>Ingestion Governance</h2>
          <span>Latest provider runs visible to operators before model actions.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap compact-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Table</th>
              <th>Source</th>
              <th>Status</th>
              <th>Rows</th>
              <th>Ingested</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={6}>Loading ingestion rows.</td>
              </tr>
            )}
            {!loading && rows.map((row, index) => (
              <tr key={`${row.table_name}-${row.source}-${row.ingested_at}-${index}`}>
                <td>{row.table_name ?? "-"}</td>
                <td>{row.source ?? "-"}</td>
                <td>
                  <span className={`status-pill ${statusTone(row.status)}`}>{row.status ?? "-"}</span>
                </td>
                <td>{formatInteger(row.row_count)}</td>
                <td>{formatDateTime(row.ingested_at)}</td>
                <td>
                  <span className="table-subtext">{row.message ?? "-"}</span>
                </td>
              </tr>
            ))}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={6}>No ingestion rows recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function AdminAuditTable({ rows, loading }: { rows: AdminAuditEvent[]; loading: boolean }) {
  return (
    <section className="registry-panel">
      <div className="evaluation-header">
        <LockKeyhole size={20} />
        <div>
          <h2>Audit History</h2>
          <span>Recorded governance actions with actor, role, resource, and request id.</span>
        </div>
      </div>
      <div className="table-wrap registry-table-wrap audit-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Action</th>
              <th>Actor</th>
              <th>Resource</th>
              <th>Status</th>
              <th>Request</th>
              <th>Created</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={7}>Loading audit history.</td>
              </tr>
            )}
            {!loading && rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <strong>{row.action}</strong>
                  <span className="table-subtext">{row.roles.join(", ") || "-"}</span>
                </td>
                <td>{row.actor}</td>
                <td>
                  {row.resourceType}
                  <span className="table-subtext">{row.resourceId ?? "-"}</span>
                </td>
                <td>
                  <span className={`status-pill ${statusTone(row.status)}`}>{row.status}</span>
                </td>
                <td>{row.requestId ?? "-"}</td>
                <td>{formatDateTime(row.createdAt)}</td>
                <td>
                  <span className="table-subtext audit-detail">{row.detail ?? formatAuditPayload(row.payload)}</span>
                </td>
              </tr>
            ))}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={7}>No audit events recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatAuditPayload(payload: Record<string, unknown> | null) {
  if (!payload) return "-";
  const text = JSON.stringify(payload);
  return text.length > 120 ? `${text.slice(0, 120)}...` : text;
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
