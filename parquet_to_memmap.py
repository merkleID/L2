#!/usr/bin/env python3
import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq


DEFAULT_COLS = [
    "ts_ns", "mid", "spread", "tick_vol",
    "bid_p1", "bid_s1", "ask_p1", "ask_s1"
]


def infer_total_rows(pf: pq.ParquetFile) -> int:
    md = pf.metadata
    return md.num_rows if md is not None else -1


def make_memmaps(out_dir: str, cols: List[str], n: int) -> Tuple[dict, dict]:
    """
    Create memmaps for each column.
    Returns (maps, dtypes)
    """
    os.makedirs(out_dir, exist_ok=True)

    dtypes = {}
    maps = {}

    for c in cols:
        if c == "ts_ns":
            dt = np.int64
        else:
            dt = np.float64
        dtypes[c] = dt
        path = os.path.join(out_dir, f"{c}.mmap")
        maps[c] = np.memmap(path, dtype=dt, mode="w+", shape=(n,))
    return maps, dtypes


def main():
    ap = argparse.ArgumentParser(description="Convert Parquet L2 dataset into NumPy memmaps per column")
    ap.add_argument("path", help="Input .parquet path")
    ap.add_argument("--out", default="l2_memmap", help="Output directory for .mmap files")
    ap.add_argument("--cols", default=",".join(DEFAULT_COLS),
                    help=f"Comma-separated columns. Default: {','.join(DEFAULT_COLS)}")
    ap.add_argument("--batch", type=int, default=500_000, help="Batch size (default 500k)")
    ap.add_argument("--limit", type=int, default=None, help="Convert only first N rows (default all)")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    if not os.path.exists(args.path):
        raise SystemExit(f"File not found: {args.path}")

    pf = pq.ParquetFile(args.path)
    total = infer_total_rows(pf)
    if total < 0:
        raise SystemExit("Could not infer total rows from Parquet metadata.")

    n = total if args.limit is None else min(total, args.limit)
    print(f"Parquet rows: {total:,}  -> converting: {n:,}")
    print(f"Columns: {cols}")
    print(f"Output dir: {args.out}")

    maps, dtypes = make_memmaps(args.out, cols, n)

    # stream batches and write sequentially
    t0 = time.time()
    pos = 0
    for batch in pf.iter_batches(batch_size=args.batch, columns=cols):
        b = batch.to_pydict()
        b_len = len(next(iter(b.values()))) if b else 0
        if b_len == 0:
            continue

        # clamp to limit
        if pos + b_len > n:
            b_len = n - pos
            if b_len <= 0:
                break

        for c in cols:
            arr = np.asarray(b[c])
            maps[c][pos:pos + b_len] = arr[:b_len]

        pos += b_len
        if pos % (args.batch * 5) == 0:
            print(f"  wrote {pos:,}/{n:,} rows ...")
        if pos >= n:
            break

    # flush
    for m in maps.values():
        m.flush()

    t1 = time.time()

    # write a tiny metadata file
    meta_path = os.path.join(args.out, "meta.npz")
    np.savez(meta_path, n=n, cols=np.array(cols), dtypes=np.array([str(dtypes[c]) for c in cols]))
    print("\nDone.")
    print(f"  rows written: {pos:,}")
    print(f"  elapsed: {t1 - t0:,.2f}s")
    print(f"  meta: {meta_path}")
    print("  memmaps:")
    for c in cols:
        print(f"    - {os.path.join(args.out, c + '.mmap')}")


if __name__ == "__main__":
    main()
