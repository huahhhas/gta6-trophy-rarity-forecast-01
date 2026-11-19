[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/huahhhas/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/huahhhas/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA 6 Trophy Rarity Forecast

Forecasts how rare each of the 55 official GTA 6 trophies will be — the
share of tracked players who will have earned each trophy by a target
time. For every trophy it outputs five quantiles (0.05 / 0.25 / 0.50 /
0.75 / 0.95) in percent.

Built for the [gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on [make.fun](https://make.fun).

## How it works

Exophase only refreshes its trophy pages every so often, so the observed
history is a step function: a trophy's displayed earn rate stays flat,
then jumps when the page refreshes. The model:

1. **Compresses history into levels.** Consecutive reads showing the same
   rate form one distinct refresh level, dated at its first read and
   weighted by how many reads it lasted.
2. **Fits a trend.** Within a recent window (24 h, widened to 48 h, then
   96 h, then everything, until at least three levels are available), a
   weighted least-squares line is fitted to the logit of the level rates
   against log time since the GTA 6 launch (2026-11-19T00:00Z).
3. **Shrinks the slope.** Each trophy's slope is pulled toward the median
   slope across all trophies (empirical Bayes, weight 0.77 on the trophy's
   own slope), so an extreme slope on one trophy is treated as noise.
4. **Caps the move.** The forecast median cannot move more than 3 logit
   units away from the last observed rate.
5. **Spreads the forecast.** The spread combines a random-walk term that
   grows with the horizon (0.45 per square root of a day) and the fit
   residual (floored at 0.15 logit, capped at 3.0).
6. **Falls back to persistence** when a trophy has fewer than three levels
   in any window: the median is the last observed rate, the spread is the
   random-walk term alone.

Quantiles use the standard z-scores (-1.645, -0.674, 0, 0.674, 1.645),
rounded half-up to four decimals, non-decreasing, clamped to [0, 100].

## Usage

```sh
python3 main.py
```

The program reads cases from `MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `<id>.out` JSON file per `<id>.in` case to
`MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3 standard library only,
no dependencies, fully offline, deterministic — two runs on the same
input produce byte-identical output.

## Verification

- Practice cases `p00` / `p01` (single-level histories): output matches
  the reference exactly (fallback persistence).
- Final-form synthetic world (121 reads, 4.5-day horizon): pinball
  (logit scale) 0.277 vs the starter curve baseline 0.324 and the
  persistence baseline 1.410 — about 15% better than the curve.

## Referral

Try make.fun with referral code `4bs5ey91`.
