# Professional Frontend

Phase 6 turns the React surface into a fuller racing workspace backed by the versioned `/api/v1` API.

## Navigation

The app now opens directly into the workspace and includes:

- Workspace
- Bet Journal
- Methodology
- Responsible Use
- Contact

## Workspace

The workspace combines:

- saved track filter
- runner search
- meetings strip
- race centre with selected-race detail
- race-card browser
- prediction rankings
- runner comparison
- model evaluation panel
- model registry with artifact readiness and recent prediction-run tables
- data-quality panel with enrichment coverage and provider freshness
- trend tables
- dark and light themes
- loading skeletons and retryable API errors

The frontend calls `/api/v1` endpoints by default.

## Race Centre

Phase 14 adds a race-by-race operating view inside the workspace. It groups prediction rows by race, keeps the selected race visible, and supports runner sorting by model rank, value edge, market odds, draw, and speed rating.

The selected race shows:

- race context: date, course, distance, race type, going, country, and average odds
- in-race top runner comparison cards
- runner-level rank, market odds, model odds, win probability, value edge, draw, jockey, trainer, and owner
- bounded transparent signal labels such as model rank, value edge, market favourite, speed context, draw, going, and race type

The signal labels are deterministic UI explanations from available model inputs; SHAP or true model-contribution explanations remain future work.

## Bet Journal

The bet journal stores entries in browser local storage. It tracks:

- open and settled positions
- total stake
- settled profit
- settled ROI

Journal metrics are intentionally separate from model-evaluation metrics.

## Responsible Use

The UI avoids language that guarantees profit. Betting outcomes are treated as uncertain, and model-quality signals remain visible beside prediction rankings.
