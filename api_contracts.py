from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    requestId: str
    statusCode: int
    detail: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class DatabaseSummary(BaseModel):
    environment: str
    engine: str
    driver: str
    database: str | None


class TableCounts(BaseModel):
    historical: int
    current: int


class PageMeta(BaseModel):
    limit: int
    offset: int
    returned: int
    total: int


class HealthResponse(BaseModel):
    status: str
    requestId: str
    database: DatabaseSummary
    counts: TableCounts


class ReadinessResponse(BaseModel):
    status: str
    requestId: str
    databaseReady: bool
    modelReady: bool
    counts: TableCounts
    message: str | None = None


class DataFreshness(BaseModel):
    status: str
    lastRefresh: str | None = None
    ageHours: float | None = None
    maxAgeHours: float


class SummaryResponse(BaseModel):
    requestId: str
    database: DatabaseSummary
    historicalRuns: int
    currentRunners: int
    lastRefresh: str | None
    dataFreshness: DataFreshness


class ProductSafeguardsResponse(BaseModel):
    requestId: str
    responsibleUseNotice: str
    limitations: List[str]
    dataLicensingNotice: str
    privacyNotice: str
    termsNotice: str
    links: Dict[str, str | None]


class RaceRunner(BaseModel):
    id: int | None = None
    race_date: str | None = None
    track: str | None = None
    distance: float | None = None
    surface: str | None = None
    horse: str | None = None
    jockey: str | None = None
    owner: str | None = None
    trainer: str | None = None
    odds: float | None = None
    finishing_position: float | None = None
    horse_age: float | None = None
    horse_weight: float | None = None
    draw: float | None = None
    speed_rating: float | None = None
    class_rating: float | None = None
    days_since_last_run: float | None = None
    past_bets_count: float | None = None
    past_bets_profit: float | None = None
    weather: str | None = None
    source: str | None = None
    ingested_at: str | None = None


class PredictionRow(RaceRunner):
    country: str | None = None
    course_latitude: float | None = None
    course_longitude: float | None = None
    distance_bucket: str | None = None
    going_category: str | None = None
    race_type: str | None = None
    race_month: float | None = None
    race_day_of_week: float | None = None
    implied_probability: float | None = None
    field_size: float | None = None
    odds_rank: float | None = None
    relative_speed_rating: float | None = None
    relative_class_rating: float | None = None
    win_probability: float | None = None
    model_odds: float | None = None
    value_edge: float | None = None
    suggested_rank: float | None = None


class RaceCardResponse(BaseModel):
    requestId: str
    raceCard: List[RaceRunner]
    page: PageMeta


class PredictionsResponse(BaseModel):
    requestId: str
    predictions: List[PredictionRow]
    page: PageMeta


class PredictionRunEntry(PredictionRow):
    id: int


class PredictionRunRow(BaseModel):
    id: int
    modelVersionId: int | None = None
    raceId: int | None = None
    runAt: str | None = None
    source: str
    notes: str | None = None
    runnerCount: int
    topRunner: str | None = None
    topWinProbability: float | None = None
    topValueEdge: float | None = None


class PredictionRunsResponse(BaseModel):
    requestId: str
    runs: List[PredictionRunRow]
    page: PageMeta


class PredictionRunResponse(BaseModel):
    requestId: str
    run: PredictionRunRow
    entries: List[PredictionRunEntry]
    page: PageMeta


BetJournalStatus = Literal["open", "won", "lost", "void"]


class BetJournalEntry(BaseModel):
    id: int
    accountKey: str
    raceEntryId: int | None = None
    predictionRunId: int | None = None
    predictionRunEntryId: int | None = None
    modelVersionId: int | None = None
    horse: str
    track: str | None = None
    raceDate: str | None = None
    betType: str = "win"
    stake: float
    odds: float
    closingOdds: float | None = None
    status: BetJournalStatus
    createdAt: str | None = None
    updatedAt: str | None = None
    settledAt: str | None = None
    profitLoss: float | None = None
    notes: str | None = None


class BetJournalCreateRequest(BaseModel):
    horse: str = Field(min_length=1, max_length=160)
    track: str | None = Field(default=None, max_length=120)
    raceDate: str | None = None
    betType: str = Field(default="win", min_length=1, max_length=80)
    stake: float = Field(gt=0)
    odds: float = Field(gt=1)
    closingOdds: float | None = Field(default=None, gt=1)
    status: BetJournalStatus = "open"
    notes: str | None = Field(default=None, max_length=2000)
    raceEntryId: int | None = None
    predictionRunId: int | None = None
    predictionRunEntryId: int | None = None
    modelVersionId: int | None = None


class BetJournalUpdateRequest(BaseModel):
    horse: str | None = Field(default=None, min_length=1, max_length=160)
    track: str | None = Field(default=None, max_length=120)
    raceDate: str | None = None
    betType: str | None = Field(default=None, min_length=1, max_length=80)
    stake: float | None = Field(default=None, gt=0)
    odds: float | None = Field(default=None, gt=1)
    closingOdds: float | None = Field(default=None, gt=1)
    status: BetJournalStatus | None = None
    notes: str | None = Field(default=None, max_length=2000)
    raceEntryId: int | None = None
    predictionRunId: int | None = None
    predictionRunEntryId: int | None = None
    modelVersionId: int | None = None


class BetJournalListResponse(BaseModel):
    requestId: str
    bets: List[BetJournalEntry]
    page: PageMeta


class BetJournalEntryResponse(BaseModel):
    requestId: str
    bet: BetJournalEntry


class BetJournalDeleteResponse(BaseModel):
    requestId: str
    id: int
    deleted: bool


class MeetingRow(BaseModel):
    race_date: str | None = None
    track: str | None = None
    races: int
    runners: int
    first_distance: float | None = None
    last_distance: float | None = None


class MeetingsResponse(BaseModel):
    requestId: str
    meetings: List[MeetingRow]
    page: PageMeta


class RaceSummaryRow(BaseModel):
    race_date: str | None = None
    track: str | None = None
    distance: float | None = None
    surface: str | None = None
    runners: int
    market_favorite: str | None = None
    average_odds: float | None = None


class RacesResponse(BaseModel):
    requestId: str
    races: List[RaceSummaryRow]
    page: PageMeta


class RaceDayRace(BaseModel):
    raceDate: str | None = None
    track: str | None = None
    distance: float | None = None
    surface: str | None = None
    offTime: str | None = None
    raceStatus: str
    statusLabel: str
    minutesToPost: float | None = None
    runners: int
    topRunner: str | None = None
    topWinProbability: float | None = None
    topValueEdge: float | None = None
    marketFavorite: str | None = None
    averageOdds: float | None = None
    provider: str | None = None
    lastIngestedAt: str | None = None
    dataAgeHours: float | None = None


class RaceDayResponse(BaseModel):
    requestId: str
    asOf: str
    today: str
    timezone: str
    nextRace: RaceDayRace | None = None
    races: List[RaceDayRace]
    page: PageMeta


class TrendRow(BaseModel):
    jockey: str | None = None
    trainer: str | None = None
    owner: str | None = None
    runs: int
    wins: int
    avg_odds: float | None = None
    win_rate: float | None = None


class TrendsResponse(BaseModel):
    requestId: str
    jockey: List[TrendRow]
    trainer: List[TrendRow]
    owner: List[TrendRow]


class DataQualityIssueRow(BaseModel):
    severity: str
    rowNumber: int | None = None
    field: str
    message: str


class FieldCoverageRow(BaseModel):
    field: str
    total: int
    nonMissing: int
    coverage: float


class DataQualityTable(BaseModel):
    tableName: str
    rows: int
    issueCount: int
    issues: List[DataQualityIssueRow]
    coverage: List[FieldCoverageRow]


class ProviderFreshnessRow(BaseModel):
    provider: str
    tableName: str
    status: str
    rowCount: int
    completedAt: str | None = None
    message: str | None = None


class DataQualityResponse(BaseModel):
    requestId: str
    tables: List[DataQualityTable]
    providerFreshness: List[ProviderFreshnessRow]


class MonitoringMetric(BaseModel):
    name: str
    label: str
    value: float | int | str | None = None
    unit: str | None = None
    status: str
    description: str | None = None


class MonitoringAlert(BaseModel):
    severity: str
    category: str
    code: str
    message: str
    value: float | int | str | None = None
    threshold: float | int | str | None = None


class ApiMetricsResponse(BaseModel):
    totalRequests: int
    errorRequests: int
    slowRequests: int
    errorRate: float
    averageLatencyMs: float
    statusCounts: Dict[str, int]
    topPaths: List[Dict[str, Any]]
    recent: List[Dict[str, Any]]
    lastErrorAt: str | None = None
    lastSlowAt: str | None = None


class DriftMetricRow(BaseModel):
    field: str
    kind: str
    status: str
    score: float | None = None
    referenceCount: int
    currentCount: int
    referenceMean: float | None = None
    currentMean: float | None = None
    referenceMissingRate: float | None = None
    currentMissingRate: float | None = None
    referenceShare: float | None = None
    currentShare: float | None = None
    topReferenceCategory: str | None = None
    topCurrentCategory: str | None = None
    maxShareDelta: float | None = None
    newCategories: List[str] = Field(default_factory=list)
    detail: str | None = None


class DriftReport(BaseModel):
    status: str
    generatedAt: str
    referenceRows: int
    currentRows: int
    thresholds: Dict[str, float]
    featureDrift: List[DriftMetricRow]
    predictionDrift: List[DriftMetricRow]


class MonitoringResponse(BaseModel):
    requestId: str
    status: str
    generatedAt: str
    dataFreshness: DataFreshness
    providerFreshness: List[ProviderFreshnessRow]
    apiMetrics: ApiMetricsResponse
    metrics: List[MonitoringMetric]
    alerts: List[MonitoringAlert]
    drift: DriftReport
    model: Dict[str, Any]


class ModelEvaluation(BaseModel):
    status: str
    message: str | None = None
    trainingRows: int
    validationRows: int
    trainingRaces: int
    validationRaces: int
    evaluationStart: str | None = None
    evaluationEnd: str | None = None
    metrics: Dict[str, float | int | None]
    leakageFeatures: List[str]


class ModelStatusResponse(BaseModel):
    requestId: str
    modelVersionId: int | None = None
    servingMode: str = "in_memory"
    artifactUri: str | None = None
    trainingRows: int
    winnerRate: float
    trainingStart: str | None = None
    trainingEnd: str | None = None
    featureCount: int
    features: List[str]
    historicalRows: int
    evaluation: ModelEvaluation


class ModelEvaluationResponse(BaseModel):
    requestId: str
    evaluation: ModelEvaluation


class ModelRegistryRow(BaseModel):
    id: int
    name: str
    algorithm: str
    status: str
    featureCount: int
    trainingStart: str | None = None
    trainingEnd: str | None = None
    artifactUri: str | None = None
    artifactSha256: str | None = None
    featureSchemaHash: str | None = None
    codeCommitSha: str | None = None
    artifactReady: bool = False
    createdAt: str | None = None
    updatedAt: str | None = None
    metrics: Dict[str, float | int | None]


class ModelRegistryResponse(BaseModel):
    requestId: str
    models: List[ModelRegistryRow]
    page: PageMeta


class ModelSnapshotResponse(BaseModel):
    requestId: str
    model: ModelRegistryRow


class IngestionRow(BaseModel):
    table_name: str | None = None
    source: str | None = None
    row_count: int | None = None
    status: str | None = None
    message: str | None = None
    ingested_at: str | None = None


class IngestionStatusResponse(BaseModel):
    requestId: str
    ingestion: List[IngestionRow]
    page: PageMeta


class BrokerRawPayloadRow(BaseModel):
    provider: str
    resource: str
    path: str
    payloadSha256: str
    fetchedAt: str | None = None
    endpoint: str | None = None
    sourceUrl: str | None = None
    rowCount: int
    licenseReference: str | None = None


class BrokerStatusResponse(BaseModel):
    requestId: str
    cacheDir: str
    rawPayloadCount: int
    providers: Dict[str, int]
    resources: Dict[str, int]
    latestFetchedAt: str | None = None
    ai: Dict[str, Any]


class BrokerRawPayloadsResponse(BaseModel):
    requestId: str
    payloads: List[BrokerRawPayloadRow]
    page: PageMeta


class BrokerPayloadShapeResponse(BaseModel):
    requestId: str
    metadata: Dict[str, Any]
    shape: Dict[str, Any]
    ai: Dict[str, Any] | None = None


class NormalizedTableCount(BaseModel):
    tableName: str
    rows: int


class NormalizedStatusResponse(BaseModel):
    requestId: str
    tables: List[NormalizedTableCount]


class NormalizedRaceEntry(BaseModel):
    raceEntryId: int
    raceId: int
    meetingId: int | None = None
    provider: str
    providerEntryId: str | None = None
    providerRaceId: str | None = None
    providerCourseId: str | None = None
    providerHorseId: str | None = None
    track: str | None = None
    country: str | None = None
    raceDate: str | None = None
    offTime: str | None = None
    distance: float | None = None
    surface: str | None = None
    going: str | None = None
    horse: str | None = None
    jockey: str | None = None
    trainer: str | None = None
    owner: str | None = None
    draw: float | None = None
    horseAge: float | None = None
    horseWeight: float | None = None
    speedRating: float | None = None
    classRating: float | None = None
    daysSinceLastRun: float | None = None
    odds: float | None = None
    oddsCapturedAt: str | None = None
    finishingPosition: float | None = None
    resultStatus: str | None = None


class NormalizedRaceEntriesResponse(BaseModel):
    requestId: str
    entries: List[NormalizedRaceEntry]
    page: PageMeta


class AccountSessionResponse(BaseModel):
    requestId: str
    actor: str
    accountKey: str
    accountId: int | None = None
    roles: List[str]
    authMode: str
    environment: str
    accountAuthEnabled: bool
    adminAuthRequired: bool
    journalAuthRequired: bool
    adminTokenConfigured: bool
    journalTokenConfigured: bool


class AdminSessionResponse(AccountSessionResponse):
    pass


class OperatorAccount(BaseModel):
    id: int
    accountKey: str
    displayName: str
    email: str | None = None
    roles: List[str]
    status: str
    tokenConfigured: bool
    privacyAcknowledgedAt: str | None = None
    lastAuthenticatedAt: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None


class OperatorAccountUpsertRequest(BaseModel):
    accountKey: str = Field(min_length=1, max_length=120)
    displayName: str = Field(min_length=1, max_length=160)
    email: str | None = Field(default=None, max_length=254)
    roles: List[str] = Field(default_factory=lambda: ["viewer"])
    token: str | None = Field(default=None, min_length=16, max_length=512)
    status: str = "active"
    privacyAcknowledged: bool = False


class OperatorAccountResponse(BaseModel):
    requestId: str
    account: OperatorAccount


class OperatorAccountListResponse(BaseModel):
    requestId: str
    accounts: List[OperatorAccount]
    page: PageMeta


class AdminAuditEvent(BaseModel):
    id: int
    actor: str
    roles: List[str]
    action: str
    resourceType: str
    resourceId: str | None = None
    requestId: str | None = None
    status: str
    detail: str | None = None
    payload: Dict[str, Any] | None = None
    createdAt: str | None = None


class AdminAuditResponse(BaseModel):
    requestId: str
    events: List[AdminAuditEvent]
    page: PageMeta


class AdminGovernanceResponse(BaseModel):
    requestId: str
    session: AdminSessionResponse
    readiness: ReadinessResponse
    summary: SummaryResponse
    monitoring: MonitoringResponse
    ingestion: List[IngestionRow]
    models: List[ModelRegistryRow]
    predictionRuns: List[PredictionRunRow]
    auditEvents: List[AdminAuditEvent]


class EntityProfileResponse(BaseModel):
    requestId: str
    entityType: str
    name: str
    runs: int
    wins: int
    winRate: float | None = None
    averageOdds: float | None = None
    latestRaceDate: str | None = None
    recentRuns: List[RaceRunner] = Field(default_factory=list)


class SeedSampleResponse(BaseModel):
    requestId: str
    seeded: Dict[str, int]
