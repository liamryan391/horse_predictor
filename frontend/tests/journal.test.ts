import test from "node:test";
import assert from "node:assert/strict";

import { betPayloadFromDraft, betProfit, createBet, parseStoredBets, summarizeBets, type Bet } from "../src/journal.ts";

test("betProfit keeps open bets neutral and settles won/lost bets", () => {
  assert.equal(betProfit(makeBet("open", 10, 4)), 0);
  assert.equal(betProfit(makeBet("won", 10, 4)), 30);
  assert.equal(betProfit(makeBet("lost", 10, 4)), -10);
  assert.equal(betProfit(makeBet("void", 10, 4)), 0);
});

test("summarizeBets separates open positions from settled ROI", () => {
  const summary = summarizeBets([makeBet("open", 10, 4), makeBet("won", 10, 3), makeBet("lost", 5, 6)]);

  assert.deepEqual(summary, {
    count: 3,
    open: 1,
    staked: 25,
    profit: 15,
    roi: 1
  });
});

test("createBet trims valid drafts and rejects unsafe stake or odds values", () => {
  assert.deepEqual(
    createBet(
      {
        horse: "  Golden Arrow ",
        track: " York ",
        raceDate: "2026-08-03",
        stake: "10",
        odds: "3.4",
        closingOdds: "",
        status: "open",
        notes: "  watched market late "
      },
      "b1",
      "now"
    ),
    {
      id: "b1",
      createdAt: "now",
      updatedAt: "now",
      horse: "Golden Arrow",
      track: "York",
      raceDate: "2026-08-03",
      stake: 10,
      odds: 3.4,
      closingOdds: null,
      status: "open",
      notes: "watched market late"
    }
  );
  assert.equal(makeDraft({ stake: "0" }) && createBet(makeDraft({ stake: "0" }), "b1", "now"), null);
  assert.equal(makeDraft({ odds: "1" }) && createBet(makeDraft({ odds: "1" }), "b1", "now"), null);
});

test("betPayloadFromDraft creates the server request payload", () => {
  assert.deepEqual(betPayloadFromDraft(makeDraft({ closingOdds: "3.1", notes: "  settled late " })), {
    horse: "Golden Arrow",
    track: "York",
    raceDate: null,
    stake: 10,
    odds: 3.4,
    closingOdds: 3.1,
    status: "open",
    notes: "settled late"
  });
  assert.equal(betPayloadFromDraft(makeDraft({ closingOdds: "1" })), null);
});

test("parseStoredBets returns only valid journal rows", () => {
  const valid = makeBet("won", 10, 3);
  const raw = JSON.stringify([valid, { id: "bad", horse: "No stake" }]);

  assert.deepEqual(parseStoredBets(raw), [valid]);
  assert.deepEqual(parseStoredBets("not-json"), []);
});

function makeBet(status: Bet["status"], stake: number, odds: number): Bet {
  return {
    id: `${status}-${stake}-${odds}`,
    createdAt: "2026-07-31T00:00:00Z",
    horse: "Golden Arrow",
    track: "York",
    stake,
    odds,
    status
  };
}

function makeDraft(overrides = {}) {
  return {
    horse: "Golden Arrow",
    track: "York",
    raceDate: "",
    stake: "10",
    odds: "3.4",
    closingOdds: "",
    status: "open" as const,
    notes: "",
    ...overrides
  };
}
