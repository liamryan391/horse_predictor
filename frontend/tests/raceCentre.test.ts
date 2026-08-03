import test from "node:test";
import assert from "node:assert/strict";

import { buildRaceGroups, raceGroupKey, runnerSignals, sortRaceRunners, type RaceCentreRunner } from "../src/raceCentre.ts";

test("buildRaceGroups creates race summaries with top ranked runner", () => {
  const groups = buildRaceGroups([
    makeRunner("Golden Arrow", 2, 0.22, 5.2),
    makeRunner("Silver Line", 1, 0.31, 3.4),
    makeRunner("Newbury Star", 1, 0.28, 4.8, { track: "Newbury", distance: 1760 })
  ]);

  assert.equal(groups.length, 2);
  assert.equal(groups[0].track, "Newbury");
  assert.equal(groups[1].track, "York");
  assert.equal(groups[1].runners, 2);
  assert.equal(groups[1].topRunner, "Silver Line");
  assert.equal(groups[1].averageOdds, 4.3);
  assert.equal(groups[1].raceStatus, null);
  assert.equal(groups[1].provider, null);
  assert.equal(raceGroupKey(groups[1].raceDate, groups[1].track, groups[1].distance), groups[1].id);
});

test("sortRaceRunners supports race-centre sort modes", () => {
  const rows = [
    makeRunner("Golden Arrow", 2, 0.22, 5.2, { value_edge: 0.04, draw: 8, speed_rating: 74 }),
    makeRunner("Silver Line", 1, 0.31, 3.4, { value_edge: -0.01, draw: 2, speed_rating: 70 }),
    makeRunner("North Gate", 3, 0.18, 8.5, { value_edge: 0.08, draw: 5, speed_rating: 82 })
  ];

  assert.equal(sortRaceRunners(rows, "rank")[0].horse, "Silver Line");
  assert.equal(sortRaceRunners(rows, "value")[0].horse, "North Gate");
  assert.equal(sortRaceRunners(rows, "market")[0].horse, "Silver Line");
  assert.equal(sortRaceRunners(rows, "draw")[0].horse, "Silver Line");
  assert.equal(sortRaceRunners(rows, "speed")[0].horse, "North Gate");
});

test("runnerSignals keeps race-centre explanations bounded and transparent", () => {
  const signals = runnerSignals(
    makeRunner("Golden Arrow", 1, 0.34, 4.2, {
      value_edge: 0.05,
      odds_rank: 1,
      relative_speed_rating: 5,
      draw: 3,
      going_category: "all_weather"
    })
  );

  assert.deepEqual(signals, [
    "Top model rank",
    "Positive value edge",
    "Market favourite",
    "Speed above field",
    "Draw 3"
  ]);
});

function makeRunner(
  horse: string,
  suggestedRank: number,
  winProbability: number,
  odds: number,
  overrides: Partial<RaceCentreRunner> = {}
): RaceCentreRunner {
  return {
    race_date: "2026-05-05",
    track: "York",
    distance: 1400,
    surface: "Dirt",
    horse,
    jockey: "A. Lee",
    trainer: "P. Miller",
    owner: "Green Acres",
    odds,
    draw: 4,
    speed_rating: 76,
    class_rating: 80,
    weather: "Sunny",
    country: "GB",
    distance_bucket: "sprint",
    going_category: "all_weather",
    race_type: "flat_aw",
    model_odds: 1 / winProbability,
    win_probability: winProbability,
    implied_probability: 1 / odds,
    value_edge: winProbability - 1 / odds,
    suggested_rank: suggestedRank,
    field_size: 3,
    odds_rank: suggestedRank,
    relative_speed_rating: 0,
    relative_class_rating: 0,
    ...overrides
  };
}
