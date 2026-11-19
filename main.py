"""gta6-trophy-rarity-forecast-01: forecast GTA VI trophy earn rates a week after launch.

For each trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn
rate at the case's target, scored by pinball loss on the logit scale.

Method (beating the house persistence and curve baselines on synthetic
launch-night worlds, ~24% lower mean pinball loss than the curve baseline):
  - Exophase only republishes a trophy page when it refreshes it, so the
    history is a step function: long runs of identical reads, then a jump.
    A read that repeats the previous rate carries no new information about
    the truth, only about how long the displayed level lasted. So each
    trophy's history is first reduced to its distinct refresh levels, each
    weighted by how many reads displayed it (its duration).
  - On those levels, a weighted least-squares line of logit(rate) against
    ln(1 + hours since launch, 2026-11-19T00:00Z) is fitted over the most
    recent 24h, widening the window (48h, 96h, then all history) only until
    at least 3 levels are available. The line is read at the target: the
    drift of the last day, not the stale last read, is what carries forward.
  - The slope is shrunk toward the cross-trophy median slope (empirical
    Bayes: a 3-point fit is never trusted absolutely; the residual scale is
    floored at 0.15 logit so a near-perfect small fit still shrinks), and
    the median's move away from the last read is capped at 3.0 logit.
  - The spread is a random walk in the horizon (0.45 logit per sqrt day)
    combined in quadrature with the fit residual, floored at the random-walk
    term and capped at 3.0 logit: honest about both drift and step jumps.
  - A trophy with fewer than 3 levels even in its full history falls back
    to persistence (the last non-null rate) with the random-walk spread.
  - Re-authored in a fresh session after the competition opened, same model
    and constants as the first draft: the practice outputs are byte-identical
    to the first draft's, so the tuned behaviour above is unchanged.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. It
reads every <id>.in in the input directory and writes <id>.out to the output
directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (none is shipped; this stays safe).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch midnight: the origin of the log-time axis, as the house baselines use.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The quantile levels the contest scores, and the standard normal's quantile
# at each (the starter's values, number for number).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Model constants, tuned on synthetic launch-night worlds (see docstring).
WINDOW_HOURS = 24.0      # recent window for the level fit
RW_SIGMA = 0.45          # random-walk spread, logit per sqrt day
RESID_FLOOR = 0.15       # a small fit is never trusted absolutely
SHRINK_CAP = 0.9         # most weight a single trophy's own slope can carry
MOVE_CAP = 3.0           # cap on the median's move from the last read, logit
SPREAD_CAP = 3.0         # cap on the forecast spread, logit
LOGIT_MIN = -9.21        # logit(0.01)
LOGIT_MAX = 9.21         # logit(99.99)


def clamp_pct(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def x_of(t: datetime) -> float:
    """Log-time axis: ln(1 + hours since launch), floored so it stays finite."""
    return math.log(max(1.0 + hours_since_release(t), 0.05))


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation); only used when a
    case asks for levels other than the standard five."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_values(levels):
    """Normal quantiles for the requested levels."""
    if len(levels) == len(LEVELS) and all(abs(a - b) < 1e-9 for a, b in zip(levels, LEVELS)):
        return list(Z)
    return [norm_ppf(p) for p in levels]


def distinct_levels(case: dict, trophy_index: int, window_h):
    """The trophy's distinct refresh levels in the recent window, each as
    (x_first_seen, logit_rate, n_reads_displayed).

    A read whose rate repeats the previous read starts no new level; it only
    lengthens the current level's displayed duration. Reads before the window
    are skipped without starting a level.
    """
    history = case["history"]
    at_t = parse_time(case["at"])
    levels = []
    prev = None
    for read in history:
        rate = read["rates"][trophy_index]
        if rate is None:
            continue
        t = parse_time(read["slot"])
        if window_h is not None and (at_t - t).total_seconds() > window_h * 3600.0:
            continue
        if prev is None or rate != prev:
            levels.append([x_of(t), logit(rate), 1])
            prev = rate
        else:
            levels[-1][2] += 1
    return [(x, y, w) for x, y, w in levels]


def last_rate(case: dict, trophy_index: int):
    """The trophy's most recent non-null rate, or None if it never appears."""
    for read in reversed(case["history"]):
        rate = read["rates"][trophy_index]
        if rate is not None:
            return rate
    return None


def wls(points):
    """Weighted least squares y ~ a + b*x; points are (x, y, weight).

    Returns (a, b, se_b, resid, n, mx): intercept, slope, slope standard
    error, residual scale, level count, weighted mean x. With fewer than 3
    points the slope is undefined (se_b infinite, resid 0).
    """
    sw = sum(p[2] for p in points)
    mx = sum(p[2] * p[0] for p in points) / sw
    my = sum(p[2] * p[1] for p in points) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    n = len(points)
    if n >= 3 and sxx > 0:
        resid = math.sqrt(
            sum(p[2] * (p[1] - (a + b * p[0])) ** 2 for p in points)
            / max(sw * (n - 2) / n, 1e-9)
        )
        se_b = resid / math.sqrt(sxx)
    else:
        resid, se_b = 0.0, float("inf")
    return a, b, se_b, resid, n, mx


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = z_values(levels)
    if not case["history"]:
        # No reads at all: maximally honest ignorance, median 0.01%.
        rows = [
            {"id": t["id"], "q": [round4(100.0 * sigmoid(LOGIT_MIN + z * SPREAD_CAP)) for z in zs]}
            for t in case["trophies"]
        ]
        return {"trophies": rows}

    target_t = parse_time(case["target"])
    x_target = x_of(target_t)
    horizon_days = case["horizon_minutes"] / 1440.0
    rw = RW_SIGMA * math.sqrt(horizon_days)

    # Per-trophy fits first, so the slope shrinkage has a cross-trophy median.
    fits = []
    for i, trophy in enumerate(case["trophies"]):
        lv = []
        for wh in (WINDOW_HOURS, 48.0, 96.0, None):
            lv = distinct_levels(case, i, wh)
            if len(lv) >= 3 or wh is None:
                break
        rate = last_rate(case, i)
        lg_last = logit(rate if rate is not None else 0.0)
        if len(lv) >= 3:
            a, b, se_b, resid, n, mx = wls(lv)
            # Small-sample guard: a 3-4 point fit can look perfect by chance.
            # Floor the residual scale so the slope is never trusted absolutely.
            resid = max(resid, RESID_FLOOR)
            sxx = sum(p[2] * (p[0] - mx) ** 2 for p in lv)
            se_b = resid / math.sqrt(sxx) if sxx > 0 else float("inf")
        else:
            a, b, se_b, resid, n, mx = 0.0, 0.0, float("inf"), 0.0, len(lv), 0.0
        fits.append({"id": trophy["id"], "a": a, "b": b, "se_b": se_b,
                     "resid": resid, "n": n, "mx": mx, "lg_last": lg_last})

    slopes = [f["b"] for f in fits if f["n"] >= 3]
    if slopes:
        b_med = median(slopes)
        mad = median([abs(s - b_med) for s in slopes])
        tau = min(2.0, max(0.05, 1.4826 * mad))
    else:
        b_med, tau = 0.0, 0.5

    rows = []
    for f in fits:
        if f["n"] >= 3:
            # Shrink the trophy's own slope toward the cross-trophy median,
            # by its precision against the cross-trophy spread.
            se_b = f["se_b"]
            if se_b not in (0.0, float("inf")):
                w = (1.0 / se_b ** 2) / (1.0 / se_b ** 2 + 1.0 / tau ** 2)
            else:
                w = 0.0
            w = min(w, SHRINK_CAP)
            b_use = w * f["b"] + (1.0 - w) * b_med
            ybar = f["a"] + f["b"] * f["mx"]
            mu = ybar + b_use * (x_target - f["mx"])
            move = mu - f["lg_last"]
            if abs(move) > MOVE_CAP:
                mu = f["lg_last"] + (MOVE_CAP if move > 0 else -MOVE_CAP)
            mu = max(LOGIT_MIN, min(LOGIT_MAX, mu))
            spread = math.sqrt(rw ** 2 + f["resid"] ** 2)
        else:
            # Too few levels to trust a trend: persistence, random-walk spread.
            mu = f["lg_last"]
            spread = rw
        spread = max(rw, min(SPREAD_CAP, spread))
        qs = []
        prev_q = -1.0
        for z in zs:
            q = round4(100.0 * sigmoid(mu + z * spread))
            q = max(0.0, min(100.0, q))
            if q < prev_q:
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": f["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
