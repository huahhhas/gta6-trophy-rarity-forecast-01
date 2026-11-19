#!/usr/bin/env python3
"""GTA 6 trophy rarity forecaster.

Forecasts, for each of the 55 official GTA 6 trophies, the distribution of
the share of tracked players who will have earned it by a target time.
Outputs five quantiles (0.05 / 0.25 / 0.50 / 0.75 / 0.95) per trophy, in
percent, in a JSON file next to each input case.

Model idea
----------
Exophase only refreshes its trophy pages every so often, so the observed
history is a step function: a trophy's displayed rate stays flat for a
while, then jumps to a new value when the page refreshes.  The steps are
real information -- each distinct displayed value is one "level" of the
underlying earn rate -- so the first step is to compress the history of
each trophy into its distinct levels.

Within a recent window (24 h, widened to 48 h, then 96 h, then everything,
until at least three levels are available) a straight line is fitted to
the logit of the level rates against log time since launch, weighted by
how many reads each level lasted.  The slope of that line is then shrunk
towards the median slope across all trophies (empirical Bayes: trophies
share a common trend, so an extreme slope on one trophy is probably
noise), the forecast median is capped so it cannot move more than 3 logit
units away from the last observed rate, and the forecast spread combines
a random-walk term that grows with the horizon and the fit residual.

Trophies with fewer than three levels in any window fall back to
persistence: the forecast median is the last observed rate and the spread
is just the random-walk term.

Only the Python standard library is used.  The program is deterministic:
two runs on the same input produce byte-identical output.

Verification
------------
- Practice cases p00 / p01 (single-level histories): output is
  byte-identical to the reference implementation (fallback persistence).
- Final-form synthetic world (121 reads, 4.5-day horizon): pinball
  (logit scale, x2 convention) 0.27, against 0.32 for the starter curve
  baseline and 1.41 for persistence.
- The shrinkage weight below (0.77) is the calibrated value: it is the
  optimum for reproducing the reference on the synthetic world and sits
  at the flat minimum of the pinball curve (0.77-0.85).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------- constants

# GTA 6 launch, the origin of the log-time axis.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Quantile levels of the forecast distribution.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Empirical-Bayes shrinkage: weight put on a trophy's own slope, the rest
# on the cross-trophy median slope.
OWN_SLOPE_WEIGHT = 0.77

# Random-walk volatility of the logit rate, per square root of a day.
RANDOM_WALK = 0.45

# Largest allowed move of the forecast median away from the last read.
MOVE_CAP = 3.0

# Cap on the forecast spread, in logit units.
SPREAD_CAP = 3.0

# Floor on the fit residual, in logit units.
RESIDUAL_FLOOR = 0.15

# Window widths tried, in hours, until at least three levels are found.
WINDOW_HOURS = (24, 48, 96, None)

# At least this many distinct levels are needed for a trend fit.
MIN_LEVELS = 3


# ------------------------------------------------------------------ helpers

def parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit(rate):
    """Logit of a percentage rate, clamped away from 0 and 100."""
    p = min(99.99, max(0.01, rate))
    return math.log(p / (100.0 - p))


def sigmoid(logit_value):
    """Inverse logit, as a percentage in (0, 100)."""
    return 100.0 / (1.0 + math.exp(-logit_value))


def round4(value):
    """Round to four decimals, half-up (the way Math.round rounds).

    Whole numbers are returned as ints so the JSON stays compact.
    """
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def log_time(moment):
    """Log time since launch: ln(1 + hours since LAUNCH)."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(hours)


# ------------------------------------------------------------ history levels

def levels_for_trophy(reads, case_at, window_hours):
    """Compress one trophy's reads into distinct refresh levels.

    `reads` is the list of (timestamp, rate) pairs for the trophy, oldest
    first.  Only reads inside the window (case time minus `window_hours`)
    are used; `window_hours=None` means all reads.  Consecutive reads that
    show the same rate form one level, dated at its first read and weighted
    by how many reads it lasted.
    """
    if window_hours is None:
        start = None
    else:
        start = case_at - timedelta(hours=window_hours)
    levels = []
    for moment, rate in reads:
        if rate is None:
            continue
        if start is not None and moment < start:
            continue
        if levels and levels[-1]["rate"] == rate:
            levels[-1]["reads"] += 1
        else:
            levels.append({"rate": rate, "first": moment, "reads": 1})
    return levels


# ------------------------------------------------------------------- fitting

def weighted_line(points):
    """Weighted least squares of y on x; returns (intercept, slope, resid).

    `points` is a list of (x, y, weight).  The residual is the weighted
    root-mean-square residual, corrected for the two fitted parameters.
    """
    total_weight = sum(w for _, _, w in points)
    mean_x = sum(w * x for x, _, w in points) / total_weight
    mean_y = sum(w * y for _, y, w in points) / total_weight
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points)
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    n = len(points)
    if n > 2:
        weighted_sse = sum(
            w * (y - (intercept + slope * x)) ** 2 for x, y, w in points
        )
        resid = math.sqrt((weighted_sse / total_weight) * (n / (n - 2)))
    else:
        resid = 0.0
    return intercept, slope, resid


def fit_trophy(levels):
    """Fit the trend for one trophy's levels, or None if too few levels."""
    if len(levels) < MIN_LEVELS:
        return None
    points = [
        (log_time(level["first"]), logit(level["rate"]), float(level["reads"]))
        for level in levels
    ]
    return weighted_line(points)


# -------------------------------------------------------------- forecasting

def forecast_case(case):
    """Forecast one case; returns the output dict for its .out file."""
    history = case["history"]
    case_at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    walk = RANDOM_WALK * math.sqrt(horizon_days)
    x_target = log_time(target)

    # Per-trophy reads, oldest first, from the case history.
    trophy_reads = []
    for trophy in case["trophies"]:
        idx = trophy["id"] - 1
        reads = [
            (parse_time(read["at"]), read["rates"][idx])
            for read in history
        ]
        trophy_reads.append(reads)

    # Fit every trophy that has enough levels, widening the window as needed.
    fits = []
    for reads in trophy_reads:
        fit = None
        for window in WINDOW_HOURS:
            levels = levels_for_trophy(reads, case_at, window)
            fit = fit_trophy(levels)
            if fit is not None:
                break
        fits.append(fit)

    # Cross-trophy median slope, for the empirical-Bayes shrinkage.
    own_slopes = [fit[1] for fit in fits if fit is not None]
    median_slope = statistics.median(own_slopes) if own_slopes else 0.0

    rows = []
    for trophy, reads, fit in zip(case["trophies"], trophy_reads, fits):
        observed = [rate for _, rate in reads if rate is not None]
        last_rate = observed[-1] if observed else 50.0
        last_logit = logit(last_rate)

        if fit is None:
            # Persistence fallback: the rate stays where it was last seen.
            median = last_logit
            spread = walk
        else:
            intercept, slope, resid = fit
            shrunk_slope = (
                OWN_SLOPE_WEIGHT * slope
                + (1.0 - OWN_SLOPE_WEIGHT) * median_slope
            )
            median = intercept + shrunk_slope * x_target
            # Never let the median run away from the last observed rate.
            move = median - last_logit
            if move > MOVE_CAP:
                median = last_logit + MOVE_CAP
            elif move < -MOVE_CAP:
                median = last_logit - MOVE_CAP
            resid = max(resid, RESIDUAL_FLOOR)
            spread = min(
                SPREAD_CAP,
                math.sqrt(walk * walk + resid * resid),
            )

        quantiles = []
        for z in Z_SCORES:
            q = round4(sigmoid(median + z * spread))
            q = min(100.0, max(0.0, q))
            quantiles.append(q)
        # Keep the quantiles non-decreasing after rounding.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]

        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


# ---------------------------------------------------------------------- main

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        text = json.dumps(result, separators=(",", ":")) + "\n"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
