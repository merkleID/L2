import os
import numpy as np

DIR = "eurusd_mmap"

meta = np.load(os.path.join(DIR, "meta.npz"), allow_pickle=True)
n = int(meta["n"])
cols = list(meta["cols"])

def mm(col, dtype):
    return np.memmap(os.path.join(DIR, f"{col}.mmap"), dtype=dtype, mode="r", shape=(n,))

ts = mm("ts_ns", np.int64)
mid = mm("mid", np.float64)
spread = mm("spread", np.float64)
tick_vol = mm("tick_vol", np.float64)
bid1 = mm("bid_p1", np.float64)
ask1 = mm("ask_p1", np.float64)

# esempio: ritorni 15m se dt_ms=100 => 15m = 900s => 9000 step
step_15m = 9000
ret_15m = np.log(mid[step_15m:] / mid[:-step_15m])

print("n:", n)
print("ret_15m mean/std:", float(ret_15m.mean()), float(ret_15m.std()))
print("spread p95:", float(np.quantile(spread, 0.95)))
