export type RaceCentreSort = "rank" | "value" | "market" | "draw" | "speed";

export type RaceCentreRunner = {
  race_date: string | null;
  track: string | null;
  distance: number | null;
  surface: string | null;
  horse: string | null;
  jockey: string | null;
  trainer: string | null;
  owner: string | null;
  odds: number | null;
  draw: number | null;
  speed_rating: number | null;
  class_rating: number | null;
  weather: string | null;
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

export type RaceCentreGroup = {
  id: string;
  raceDate: string | null;
  track: string | null;
  distance: number | null;
  surface: string | null;
  country: string | null;
  raceType: string | null;
  distanceBucket: string | null;
  goingCategory: string | null;
  weather: string | null;
  runners: number;
  topRunner: string | null;
  topWinProbability: number | null;
  averageOdds: number | null;
};

export function raceKey(row: RaceCentreRunner): string {
  return [row.race_date ?? "unknown-date", row.track ?? "unknown-track", row.distance ?? "unknown-distance"].join("|");
}

export function buildRaceGroups(rows: RaceCentreRunner[]): RaceCentreGroup[] {
  const grouped = new Map<string, RaceCentreRunner[]>();
  rows.forEach((row) => {
    const key = raceKey(row);
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  });

  return Array.from(grouped.entries())
    .map(([id, runners]) => {
      const first = runners[0];
      const sorted = sortRaceRunners(runners, "rank");
      const odds = runners.map((runner) => runner.odds).filter((value): value is number => typeof value === "number");
      return {
        id,
        raceDate: first.race_date,
        track: first.track,
        distance: first.distance,
        surface: first.surface,
        country: first.country,
        raceType: first.race_type,
        distanceBucket: first.distance_bucket,
        goingCategory: first.going_category,
        weather: first.weather,
        runners: runners.length,
        topRunner: sorted[0]?.horse ?? null,
        topWinProbability: sorted[0]?.win_probability ?? null,
        averageOdds: odds.length ? odds.reduce((total, value) => total + value, 0) / odds.length : null
      };
    })
    .sort((first, second) => {
      const dateCompare = String(first.raceDate ?? "").localeCompare(String(second.raceDate ?? ""));
      if (dateCompare !== 0) return dateCompare;
      const trackCompare = String(first.track ?? "").localeCompare(String(second.track ?? ""));
      if (trackCompare !== 0) return trackCompare;
      return (first.distance ?? Number.MAX_SAFE_INTEGER) - (second.distance ?? Number.MAX_SAFE_INTEGER);
    });
}

export function sortRaceRunners<T extends RaceCentreRunner>(rows: T[], sort: RaceCentreSort): T[] {
  const sorted = [...rows];
  const highLast = Number.POSITIVE_INFINITY;
  const lowLast = Number.NEGATIVE_INFINITY;

  return sorted.sort((first, second) => {
    if (sort === "value") {
      return (second.value_edge ?? lowLast) - (first.value_edge ?? lowLast);
    }
    if (sort === "market") {
      return (first.odds ?? highLast) - (second.odds ?? highLast);
    }
    if (sort === "draw") {
      return (first.draw ?? highLast) - (second.draw ?? highLast);
    }
    if (sort === "speed") {
      return (second.speed_rating ?? lowLast) - (first.speed_rating ?? lowLast);
    }
    const rankCompare = (first.suggested_rank ?? highLast) - (second.suggested_rank ?? highLast);
    if (rankCompare !== 0) return rankCompare;
    return (second.win_probability ?? lowLast) - (first.win_probability ?? lowLast);
  });
}

export function runnerSignals(row: RaceCentreRunner): string[] {
  const signals: string[] = [];
  if (row.suggested_rank === 1) signals.push("Top model rank");
  if ((row.value_edge ?? 0) >= 0.02) signals.push("Positive value edge");
  if ((row.value_edge ?? 0) <= -0.02) signals.push("Market shorter than model");
  if (row.odds_rank === 1) signals.push("Market favourite");
  if ((row.relative_speed_rating ?? 0) >= 4) signals.push("Speed above field");
  if ((row.relative_class_rating ?? 0) >= 4) signals.push("Class above field");
  if (row.draw !== null && row.draw !== undefined) signals.push(`Draw ${Math.round(row.draw)}`);
  if (row.going_category) signals.push(`Going ${row.going_category.replace("_", " ")}`);
  if (row.race_type) signals.push(row.race_type.replace("_", " "));
  return signals.slice(0, 5);
}
