#!/usr/bin/env python3
"""
Synthetic EURUSD L2 generator (intraday-aware)

Adds:
- Sessionality (Asia / London / NY / overlap): vol/spread/volume multipliers by time-of-day
- News shocks (Poisson arrivals) that create jumps + temporary vol/spread/volume bursts with decay
- Tick-volume proxy per snapshot (useful since spot FX has no centralized volume)

Output per row (snapshot):
  ts_ns, mid, spread, tick_vol,
  bid_p1..bid_pN, bid_s1..bid_sN,
  ask_p1..ask_pN, ask_s1..ask_sN

Recommended output: Parquet (zstd) if pyarrow is installed.
"""

import argparse
import math
import os
import time
from datetime import datetime, timezone

import numpy as np

# Optional parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_ARROW = True
except Exception:
    _HAS_ARROW = False


# ------------------------- time helpers -------------------------

def parse_start_time_to_ns(s: str) -> int:
    """
    Accepts:
      - "now" (default)
      - ISO8601 like "2026-01-05T08:00:00Z" or without Z (assumed UTC)
    """
    if s.lower() == "now":
        return int(datetime.now(timezone.utc).timestamp() * 1e9)

    if s.endswith("Z"):
        s = s[:-1]
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1e9)


def ts_ns_to_utc_hour(ts_ns: int) -> float:
    """Return fractional UTC hour [0,24)."""
    sec = ts_ns / 1e9
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


# ------------------------- L2 book core -------------------------

def init_book(mid: float, depth: int, tick: float, base_spread_ticks: int, rng: np.random.Generator):
    spread_ticks = max(1, base_spread_ticks)
    best_bid = mid - (spread_ticks * tick) / 2.0
    best_ask = mid + (spread_ticks * tick) / 2.0

    bid_prices = best_bid - tick * np.arange(depth)
    ask_prices = best_ask + tick * np.arange(depth)

    # Sizes: heavier near top, lognormal with decay
    decay = np.exp(-0.18 * np.arange(depth))
    bid_sizes = (rng.lognormal(mean=0.2, sigma=0.65, size=depth) * 1e6 * decay).astype(np.float64)
    ask_sizes = (rng.lognormal(mean=0.2, sigma=0.65, size=depth) * 1e6 * decay).astype(np.float64)

    return bid_prices, bid_sizes, ask_prices, ask_sizes, spread_ticks


def refresh_book_around_mid(mid: float, depth: int, tick: float, spread_ticks: int,
                           bid_sizes: np.ndarray, ask_sizes: np.ndarray,
                           rng: np.random.Generator):
    best_bid = mid - (spread_ticks * tick) / 2.0
    best_ask = mid + (spread_ticks * tick) / 2.0
    bid_prices = best_bid - tick * np.arange(depth)
    ask_prices = best_ask + tick * np.arange(depth)

    # mild multiplicative drift
    bid_sizes = np.maximum(1.0, bid_sizes * rng.lognormal(mean=0.0, sigma=0.05, size=depth))
    ask_sizes = np.maximum(1.0, ask_sizes * rng.lognormal(mean=0.0, sigma=0.05, size=depth))

    return bid_prices, bid_sizes, ask_prices, ask_sizes


def micro_events(bid_sizes: np.ndarray, ask_sizes: np.ndarray, rng: np.random.Generator,
                 event_rate: float):
    """
    Queue dynamics: random cancels/adds. event_rate scales how many events happen.
    """
    depth = len(bid_sizes)
    # Poisson number of events per snapshot
    k = int(rng.poisson(lam=max(0.0, event_rate)))
    for _ in range(k):
        side = rng.choice(["bid", "ask"])
        lvl = int(rng.integers(0, depth))
        if rng.random() < 0.58:
            # cancel fraction
            frac = float(rng.uniform(0.03, 0.45))
            if side == "bid":
                bid_sizes[lvl] = max(1.0, bid_sizes[lvl] * (1.0 - frac))
            else:
                ask_sizes[lvl] = max(1.0, ask_sizes[lvl] * (1.0 - frac))
        else:
            # add
            add = float(rng.lognormal(mean=0.0, sigma=0.7) * 2.5e5)
            if side == "bid":
                bid_sizes[lvl] += add
            else:
                ask_sizes[lvl] += add
    return bid_sizes, ask_sizes


# ------------------------- intraday regime -------------------------

def session_multipliers(utc_hour: float):
    """
    Rough EURUSD intraday liquidity/vol profile in UTC.
    Returns (vol_mult, spread_mult, volu_mult).

    Heuristic:
      - Asia: 00-06  (quiet)
      - London: 07-12 (active)
      - Overlap: 12-16 (most active)
      - NY: 16-21 (active)
      - Late: 21-24 (quiet)
    """
    h = utc_hour
    if 0 <= h < 6:
        return 0.8, 1.2, 0.7
    if 6 <= h < 7:
        return 0.9, 1.1, 0.85
    if 7 <= h < 12:
        return 1.2, 0.95, 1.25
    if 12 <= h < 16:
        return 1.45, 0.9, 1.55
    if 16 <= h < 21:
        return 1.15, 1.0, 1.2
    return 0.85, 1.15, 0.75


def regime_switch(vol: float, rng: np.random.Generator):
    """
    Baseline volatility regime drift (without session multiplier).
    vol is log-return std per step.
    """
    u = rng.random()
    if u < 0.82:
        target = 0.000018
    elif u < 0.97:
        target = 0.000055
    else:
        target = 0.000160
    return 0.975 * vol + 0.025 * target


def update_mid(mid: float, vol_step: float, rng: np.random.Generator):
    return mid * math.exp(rng.normal(0.0, vol_step))


def update_spread_ticks(spread_ticks: int, vol_step: float, spread_mult: float, rng: np.random.Generator):
    """
    Spread depends on volatility AND session liquidity.
    """
    if vol_step < 0.00003:
        target = 1
    elif vol_step < 0.000085:
        target = 2
    else:
        target = 3 + int(rng.integers(0, 3))  # 3-5

    # apply session spread multiplier (wider in illiquid hours)
    target = int(max(1, round(target * spread_mult)))

    new = int(round(0.85 * spread_ticks + 0.15 * target))
    new += int(rng.choice([-1, 0, 0, 0, 1]))
    return max(1, min(new, 10))


# ------------------------- news shocks -------------------------

class ShockState:
    """
    News shock process:
    - Poisson arrivals with intensity (lambda per hour)
    - On arrival: jump in log-mid + temporary volatility/spread/volume burst
    - Decays exponentially over 'shock_half_life_steps'
    """
    def __init__(self, rng: np.random.Generator, dt_ms: int,
                 lambda_per_hour: float,
                 jump_sigma: float,
                 shock_vol_boost: float,
                 shock_spread_mult: float,
                 shock_volume_mult: float,
                 shock_half_life_steps: int):
        self.rng = rng
        self.dt_ms = dt_ms
        self.lambda_per_hour = lambda_per_hour
        self.jump_sigma = jump_sigma
        self.shock_vol_boost = shock_vol_boost
        self.shock_spread_mult = shock_spread_mult
        self.shock_volume_mult = shock_volume_mult
        self.shock_half_life_steps = max(1, shock_half_life_steps)

        self.level = 0.0  # shock intensity level [0..inf)

        # arrival probability per step
        dt_hours = (dt_ms / 1000.0) / 3600.0
        self.p_arrival = 1.0 - math.exp(-lambda_per_hour * dt_hours)

        # decay factor per step from half-life
        self.decay = 2.0 ** (-1.0 / self.shock_half_life_steps)

    def step(self):
        # decay
        self.level *= self.decay

        # arrival?
        if self.rng.random() < self.p_arrival:
            # additive intensity bump
            self.level += float(self.rng.uniform(0.7, 1.4))
            return True
        return False

    def apply_jump_to_mid(self, mid: float) -> float:
        # jump in log space proportional to level
        j = self.rng.normal(0.0, self.jump_sigma) * max(0.5, self.level)
        return mid * math.exp(j)

    def multipliers(self):
        # Convert shock level to multipliers (bounded)
        L = min(self.level, 4.0)
        vol_add = self.shock_vol_boost * L
        spread_mult = 1.0 + (self.shock_spread_mult - 1.0) * (L / 4.0)
        volu_mult = 1.0 + (self.shock_volume_mult - 1.0) * (L / 4.0)
        return vol_add, spread_mult, volu_mult


# ------------------------- output writers -------------------------

def rows_to_batch_arrays(rows, depth: int):
    n = len(rows)
    ts = np.empty(n, dtype=np.int64)
    mid = np.empty(n, dtype=np.float64)
    spr = np.empty(n, dtype=np.float64)
    tvol = np.empty(n, dtype=np.float64)

    bid_p = np.empty((n, depth), dtype=np.float64)
    bid_s = np.empty((n, depth), dtype=np.float64)
    ask_p = np.empty((n, depth), dtype=np.float64)
    ask_s = np.empty((n, depth), dtype=np.float64)

    for i, r in enumerate(rows):
        ts[i] = r[0]
        mid[i] = r[1]
        spr[i] = r[2]
        tvol[i] = r[3]
        bid_p[i, :] = r[4]
        bid_s[i, :] = r[5]
        ask_p[i, :] = r[6]
        ask_s[i, :] = r[7]

    arrays = {
        "ts_ns": ts,
        "mid": mid,
        "spread": spr,
        "tick_vol": tvol,
    }
    for j in range(depth):
        arrays[f"bid_p{j+1}"] = bid_p[:, j]
    for j in range(depth):
        arrays[f"bid_s{j+1}"] = bid_s[:, j]
    for j in range(depth):
        arrays[f"ask_p{j+1}"] = ask_p[:, j]
    for j in range(depth):
        arrays[f"ask_s{j+1}"] = ask_s[:, j]
    return arrays


def write_parquet_stream(out_path: str, depth: int, generator, total_rows: int, batch_rows: int):
    schema = pa.schema([
        ("ts_ns", pa.int64()),
        ("mid", pa.float64()),
        ("spread", pa.float64()),
        ("tick_vol", pa.float64()),
        *[(f"bid_p{i}", pa.float64()) for i in range(1, depth + 1)],
        *[(f"bid_s{i}", pa.float64()) for i in range(1, depth + 1)],
        *[(f"ask_p{i}", pa.float64()) for i in range(1, depth + 1)],
        *[(f"ask_s{i}", pa.float64()) for i in range(1, depth + 1)],
    ])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
        written = 0
        buf = []
        for row in generator:
            buf.append(row)
            if len(buf) >= batch_rows:
                arrays = rows_to_batch_arrays(buf, depth)
                table = pa.Table.from_pydict(arrays, schema=schema)
                writer.write_table(table)
                written += len(buf)
                buf.clear()
                if written % (batch_rows * 10) == 0:
                    print(f"  wrote {written:,}/{total_rows:,} rows ...")
        if buf:
            arrays = rows_to_batch_arrays(buf, depth)
            table = pa.Table.from_pydict(arrays, schema=schema)
            writer.write_table(table)
            written += len(buf)

    return written


def write_csv_stream(out_path: str, depth: int, generator, total_rows: int, batch_rows: int):
    import csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cols = ["ts_ns", "mid", "spread", "tick_vol"]
    cols += [f"bid_p{i}" for i in range(1, depth + 1)]
    cols += [f"bid_s{i}" for i in range(1, depth + 1)]
    cols += [f"ask_p{i}" for i in range(1, depth + 1)]
    cols += [f"ask_s{i}" for i in range(1, depth + 1)]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)

        written = 0
        buf = []
        for row in generator:
            buf.append(row)
            if len(buf) >= batch_rows:
                for r in buf:
                    ts_ns, mid, spread, tick_vol, bid_p, bid_s, ask_p, ask_s = r
                    line = [ts_ns, mid, spread, tick_vol,
                            *bid_p.tolist(), *bid_s.tolist(),
                            *ask_p.tolist(), *ask_s.tolist()]
                    w.writerow(line)
                written += len(buf)
                buf.clear()
                if written % (batch_rows * 10) == 0:
                    print(f"  wrote {written:,}/{total_rows:,} rows ...")

        if buf:
            for r in buf:
                ts_ns, mid, spread, tick_vol, bid_p, bid_s, ask_p, ask_s = r
                line = [ts_ns, mid, spread, tick_vol,
                        *bid_p.tolist(), *bid_s.tolist(),
                        *ask_p.tolist(), *ask_s.tolist()]
                w.writerow(line)
            written += len(buf)

    return written


# ------------------------- generator -------------------------

def generate_l2_snapshots(
    rows: int,
    depth: int,
    tick: float,
    dt_ms: int,
    seed: int,
    start_mid: float,
    start_time_ns: int,
    base_spread_ticks: int,
    base_tickvol: float,
    micro_event_rate: float,
    news_lambda_per_hour: float,
    news_jump_sigma: float,
    shock_vol_boost: float,
    shock_spread_mult: float,
    shock_volume_mult: float,
    shock_half_life_steps: int,
):
    rng = np.random.default_rng(seed)

    mid = float(start_mid)
    vol = 0.00005  # baseline log-return std per step
    spread_ticks = max(1, base_spread_ticks)

    bid_p, bid_s, ask_p, ask_s, spread_ticks = init_book(
        mid=mid, depth=depth, tick=tick, base_spread_ticks=spread_ticks, rng=rng
    )

    ts = int(start_time_ns)
    dt_ns = int(dt_ms * 1e6)

    shock = ShockState(
        rng=rng,
        dt_ms=dt_ms,
        lambda_per_hour=news_lambda_per_hour,
        jump_sigma=news_jump_sigma,
        shock_vol_boost=shock_vol_boost,
        shock_spread_mult=shock_spread_mult,
        shock_volume_mult=shock_volume_mult,
        shock_half_life_steps=shock_half_life_steps
    )

    for _ in range(rows):
        utc_h = ts_ns_to_utc_hour(ts)
        sess_vol_mult, sess_spread_mult, sess_volume_mult = session_multipliers(utc_h)

        # baseline vol regime drift
        vol = regime_switch(vol, rng)

        # shock update
        arrived = shock.step()
        if arrived:
            # jump on arrival
            mid = shock.apply_jump_to_mid(mid)

        vol_add, shock_sp_mult, shock_v_mult = shock.multipliers()

        # final per-step volatility
        vol_step = max(1e-8, vol * sess_vol_mult + vol_add)

        # mid update (diffusion)
        mid = update_mid(mid, vol_step, rng)

        # spread update
        spread_ticks = update_spread_ticks(spread_ticks, vol_step, sess_spread_mult * shock_sp_mult, rng)

        # re-anchor book around mid
        bid_p, bid_s, ask_p, ask_s = refresh_book_around_mid(
            mid=mid, depth=depth, tick=tick, spread_ticks=spread_ticks,
            bid_sizes=bid_s, ask_sizes=ask_s, rng=rng
        )

        # microstructure events intensity scales with activity (session + shock)
        activity = sess_volume_mult * shock_v_mult
        bid_s, ask_s = micro_events(bid_s, ask_s, rng, event_rate=micro_event_rate * activity)

        spread = float(ask_p[0] - bid_p[0])

        # tick-volume proxy: base scaled by activity + noise, loosely correlated with volatility
        tick_vol = float(max(
            1.0,
            base_tickvol * activity * (1.0 + 6.0 * vol_step / 0.00006) * rng.lognormal(mean=0.0, sigma=0.25)
        ))

        yield (ts, float(mid), spread, tick_vol, bid_p.copy(), bid_s.copy(), ask_p.copy(), ask_s.copy())
        ts += dt_ns


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="Number of L2 snapshots to generate")
    ap.add_argument("--out", type=str, default=None, help="Output file (.parquet recommended; else .csv)")
    ap.add_argument("--depth", type=int, default=10, help="Depth per side (default 10)")
    ap.add_argument("--dt_ms", type=int, default=100, help="Milliseconds between snapshots (default 100)")
    ap.add_argument("--tick", type=float, default=0.00001, help="Tick size (default 1e-5)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--start_mid", type=float, default=1.0950, help="Starting mid price")
    ap.add_argument("--start_time", type=str, default="now", help='Start time UTC: "now" or ISO "2026-01-05T08:00:00Z"')
    ap.add_argument("--batch_rows", type=int, default=50_000, help="Rows per write batch (default 50k)")

    # market behavior knobs
    ap.add_argument("--base_spread_ticks", type=int, default=2, help="Base spread in ticks (default 2)")
    ap.add_argument("--base_tickvol", type=float, default=120.0, help="Baseline tick-volume per snapshot (default 120)")
    ap.add_argument("--micro_event_rate", type=float, default=2.2, help="Avg micro events per snapshot at baseline (default 2.2)")

    # news shock knobs
    ap.add_argument("--news_lambda_per_hour", type=float, default=0.35, help="News shock intensity per hour (default 0.35)")
    ap.add_argument("--news_jump_sigma", type=float, default=0.00020, help="Jump sigma in log space (default 0.00020)")
    ap.add_argument("--shock_vol_boost", type=float, default=0.00006, help="Additive vol boost at shock level=1 (default 6e-5)")
    ap.add_argument("--shock_spread_mult", type=float, default=2.2, help="Max spread multiplier during shocks (default 2.2)")
    ap.add_argument("--shock_volume_mult", type=float, default=3.0, help="Max volume multiplier during shocks (default 3.0)")
    ap.add_argument("--shock_half_life_steps", type=int, default=120, help="Shock half-life in steps (default 120)")

    args = ap.parse_args()

    # Ask interactively if not provided
    if args.rows is None:
        while True:
            s = input("Quante righe (snapshots L2) vuoi generare? Es: 1000000  --> ").strip()
            try:
                args.rows = int(s)
                if args.rows <= 0:
                    raise ValueError
                break
            except Exception:
                print("Valore non valido. Inserisci un intero positivo, es. 1000000.")

    if args.out is None:
        ext = ".parquet" if _HAS_ARROW else ".csv"
        args.out = f"eurusd_l2_synth_{args.rows}_rows{ext}"

    start_ns = parse_start_time_to_ns(args.start_time)

    use_parquet = args.out.lower().endswith(".parquet") and _HAS_ARROW
    if args.out.lower().endswith(".parquet") and not _HAS_ARROW:
        print("pyarrow non disponibile: non posso scrivere parquet. Userò CSV.")
        args.out = args.out.rsplit(".", 1)[0] + ".csv"
        use_parquet = False

    print("Synthetic L2 EURUSD generator (intraday + news shocks)")
    print(f"  rows      : {args.rows:,}")
    print(f"  depth     : {args.depth}")
    print(f"  dt_ms     : {args.dt_ms}")
    print(f"  start_mid : {args.start_mid}")
    print(f"  start_time: {args.start_time} (UTC)")
    print(f"  out       : {args.out}")
    print(f"  format    : {'parquet(zstd)' if use_parquet else 'csv'}")

    gen = generate_l2_snapshots(
        rows=args.rows,
        depth=args.depth,
        tick=args.tick,
        dt_ms=args.dt_ms,
        seed=args.seed,
        start_mid=args.start_mid,
        start_time_ns=start_ns,
        base_spread_ticks=args.base_spread_ticks,
        base_tickvol=args.base_tickvol,
        micro_event_rate=args.micro_event_rate,
        news_lambda_per_hour=args.news_lambda_per_hour,
        news_jump_sigma=args.news_jump_sigma,
        shock_vol_boost=args.shock_vol_boost,
        shock_spread_mult=args.shock_spread_mult,
        shock_volume_mult=args.shock_volume_mult,
        shock_half_life_steps=args.shock_half_life_steps,
    )

    t0 = time.time()
    if use_parquet:
        written = write_parquet_stream(args.out, args.depth, gen, args.rows, args.batch_rows)
    else:
        written = write_csv_stream(args.out, args.depth, gen, args.rows, args.batch_rows)
    t1 = time.time()

    size_bytes = os.path.getsize(args.out)
    print("\nDone.")
    print(f"  wrote rows : {written:,}")
    print(f"  file size  : {size_bytes / (1024**2):,.1f} MB")
    print(f"  elapsed    : {t1 - t0:,.1f} s")
    print("\nTip: per arrivare a ~1GB, usa parquet e aumenta --rows (milioni di righe).")


if __name__ == "__main__":
    main()
