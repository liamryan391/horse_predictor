export type BetStatus = "open" | "won" | "lost";

export type Bet = {
  id: string;
  createdAt: string;
  horse: string;
  track: string;
  stake: number;
  odds: number;
  status: BetStatus;
};

export type BetDraft = {
  horse: string;
  track: string;
  stake: string;
  odds: string;
  status: BetStatus;
};

export type JournalSummary = {
  count: number;
  open: number;
  staked: number;
  profit: number;
  roi: number | null;
};

export function betProfit(bet: Bet) {
  if (bet.status === "open") return 0;
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
  const stake = Number(draft.stake);
  const odds = Number(draft.odds);
  if (!draft.horse.trim() || !Number.isFinite(stake) || !Number.isFinite(odds) || stake <= 0 || odds <= 1) {
    return null;
  }

  return {
    id,
    createdAt,
    horse: draft.horse.trim(),
    track: draft.track.trim(),
    stake,
    odds,
    status: draft.status
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
    typeof candidate.id === "string" &&
    typeof candidate.createdAt === "string" &&
    typeof candidate.horse === "string" &&
    typeof candidate.track === "string" &&
    typeof candidate.stake === "number" &&
    Number.isFinite(candidate.stake) &&
    typeof candidate.odds === "number" &&
    Number.isFinite(candidate.odds) &&
    (candidate.status === "open" || candidate.status === "won" || candidate.status === "lost")
  );
}
