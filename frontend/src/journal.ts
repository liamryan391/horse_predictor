export type BetStatus = "open" | "won" | "lost" | "void";

export type Bet = {
  id: string | number;
  accountKey?: string;
  createdAt: string;
  updatedAt?: string | null;
  settledAt?: string | null;
  horse: string;
  track: string | null;
  raceDate?: string | null;
  stake: number;
  odds: number;
  closingOdds?: number | null;
  status: BetStatus;
  profitLoss?: number | null;
  notes?: string | null;
};

export type BetDraft = {
  horse: string;
  track: string;
  raceDate: string;
  stake: string;
  odds: string;
  closingOdds: string;
  status: BetStatus;
  notes: string;
};

export type BetPayload = {
  horse: string;
  track: string | null;
  raceDate: string | null;
  stake: number;
  odds: number;
  closingOdds?: number | null;
  status: BetStatus;
  notes: string | null;
};

export type JournalSummary = {
  count: number;
  open: number;
  staked: number;
  profit: number;
  roi: number | null;
};

export function betProfit(bet: Bet) {
  if (typeof bet.profitLoss === "number" && bet.status !== "open") return bet.profitLoss;
  if (bet.status === "open") return 0;
  if (bet.status === "void") return 0;
  return bet.status === "won" ? bet.stake * (bet.odds - 1) : -bet.stake;
}

export function summarizeBets(bets: Bet[]): JournalSummary {
  const settled = bets.filter((bet) => bet.status !== "open");
  const staked = bets.reduce((total, bet) => total + bet.stake, 0);
  const settledStake = settled.reduce((total, bet) => total + bet.stake, 0);
  const profit = settled.reduce((total, bet) => total + betProfit(bet), 0);

  return {
    count: bets.length,
    open: bets.filter((bet) => bet.status === "open").length,
    staked,
    profit,
    roi: settledStake > 0 ? profit / settledStake : null
  };
}

export function createBet(draft: BetDraft, id: string, createdAt: string): Bet | null {
  const payload = betPayloadFromDraft(draft);
  if (!payload) {
    return null;
  }

  return {
    id,
    createdAt,
    updatedAt: createdAt,
    horse: payload.horse,
    track: payload.track,
    raceDate: payload.raceDate,
    stake: payload.stake,
    odds: payload.odds,
    closingOdds: payload.closingOdds ?? null,
    status: payload.status,
    notes: payload.notes
  };
}

export function betPayloadFromDraft(draft: BetDraft): BetPayload | null {
  const horse = draft.horse.trim();
  const track = draft.track.trim();
  const raceDate = draft.raceDate.trim();
  const notes = draft.notes.trim();
  const stake = Number(draft.stake);
  const odds = Number(draft.odds);
  const closingOdds = draft.closingOdds.trim() ? Number(draft.closingOdds) : null;
  if (
    !horse ||
    !Number.isFinite(stake) ||
    !Number.isFinite(odds) ||
    stake <= 0 ||
    odds <= 1 ||
    (closingOdds !== null && (!Number.isFinite(closingOdds) || closingOdds <= 1))
  ) {
    return null;
  }

  return {
    horse,
    track: track || null,
    raceDate: raceDate || null,
    stake,
    odds,
    closingOdds,
    status: draft.status,
    notes: notes || null
  };
}

export function parseStoredBets(raw: string | null): Bet[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isBet);
  } catch {
    return [];
  }
}

function isBet(value: unknown): value is Bet {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<Bet>;
  return (
    (typeof candidate.id === "string" || typeof candidate.id === "number") &&
    typeof candidate.createdAt === "string" &&
    typeof candidate.horse === "string" &&
    (candidate.track === null || typeof candidate.track === "string") &&
    typeof candidate.stake === "number" &&
    Number.isFinite(candidate.stake) &&
    typeof candidate.odds === "number" &&
    Number.isFinite(candidate.odds) &&
    (candidate.status === "open" || candidate.status === "won" || candidate.status === "lost" || candidate.status === "void")
  );
}
