#!/usr/bin/env python3
import argparse
import os
import time

import pyarrow.parquet as pq
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Fast Parquet reader for synthetic L2 EURUSD dataset")
    ap.add_argument("path", help="Input .parquet path")
    ap.add_argument("--cols", default=None,
                    help='Comma-separated columns to load. Example: "ts_ns,mid,spread,tick_vol,bid_p1,ask_p1"')
    ap.add_argument("--head", type=int, default=5, help="Print first N rows (default 5)")
    ap.add_argument("--rows", type=int, default=None, help="Read only first N rows (default: all)")
    ap.add_argument("--batch", type=int, default=250_000, help="Batch size for streaming (default 250k)")
    ap.add_argument("--stats", action="store_true", help="Print quick stats")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"File not found: {args.path}")

    cols = None
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    pf = pq.ParquetFile(args.path)

    t0 = time.time()
    read_rows = 0
    dfs = []

    for batch in pf.iter_batches(batch_size=args.batch, columns=cols):
        df = batch.to_pandas(types_mapper=pd.ArrowDtype)
        dfs.append(df)
        read_rows += len(df)
        if args.rows is not None and read_rows >= args.rows:
            break

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if args.rows is not None and len(df) > args.rows:
        df = df.iloc[:args.rows].copy()

    t1 = time.time()

    print(f"Loaded rows: {len(df):,}")
    print(f"Elapsed: {t1 - t0:,.2f}s")

    if args.head and len(df) > 0:
        print("\nHEAD:")
        print(df.head(args.head).to_string(index=False))

    if args.stats and len(df) > 0:
        # quick sanity stats
        out = {}
        for c in ["mid", "spread", "tick_vol"]:
            if c in df.columns:
                s = df[c].astype("float64")
                out[c] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "p50": float(s.quantile(0.50)),
                    "p95": float(s.quantile(0.95)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
        if "bid_p1" in df.columns and "ask_p1" in df.columns:
            spr = (df["ask_p1"].astype("float64") - df["bid_p1"].astype("float64"))
            out["top_spread_check"] = {
                "mean": float(spr.mean()),
                "p95": float(spr.quantile(0.95)),
            }

        print("\nSTATS:")
        for k, v in out.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
