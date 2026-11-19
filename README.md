# GTA VI Trophy Rarity Forecast — Week One

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/huahhhas/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/huahhhas/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun: given a case of Exophase trophy reads from launch night, it predicts, for each of the game's 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate at 2026-11-26T00:00Z, scored by pinball loss on the logit scale.

## How it works

Exophase only republishes a trophy page when it refreshes it, so the history is a step function: long runs of identical reads, then a jump. A read that repeats the previous rate carries no new information about the truth, only about how long the displayed level lasted. So `main.py`:

1. Reduces each trophy's history to its distinct refresh levels, each weighted by how many reads displayed it (its duration).
2. Fits a duration-weighted least-squares line of logit(rate) against ln(1 + hours since launch, 2026-11-19T00:00Z) over the most recent 24h, widening the window (48h, 96h, then all history) only until at least 3 levels are available, and reads the line at the target.
3. Shrinks each slope toward the cross-trophy median slope (empirical Bayes; a 3-point fit is never trusted absolutely, residual floored at 0.15 logit) and caps the median's move from the last read at 3.0 logit.
4. Sets the spread to a random walk in the horizon (0.45 logit per sqrt day) combined in quadrature with the fit residual, floored at the random-walk term and capped at 3.0 logit.
5. Falls back to persistence (last non-null rate) with the random-walk spread for a trophy with fewer than 3 levels even in its full history.

On synthetic launch-night worlds the model beats the house curve baseline in every regime tested (base, fast/slow refresh, fast/slow progress): mean pinball loss ≈ 0.41 vs ≈ 0.54 for the curve baseline over 60 worlds, and ≈ 0.28 vs ≈ 0.39 on the base regime.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13 standard library only, offline, deterministic (two runs write byte-identical output), well under the 8 s/case limit. Reads every `<id>.in` in the input directory and writes `<id>.out` — `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}` — to the output directory.

## Bounty

Built for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01). Referral code: `4bs5ey91`.

## License

MIT — see [LICENSE](LICENSE).
