#!/usr/bin/env python3
"""
Build an augmented full-year EPW from a source window (default: hours 5088-6551
= 1 Aug - 30 Sep) so a DRL agent sees many *different* but *plausible* weather
trajectories that resemble the training period.

Method (per output day):
  1. Block bootstrap: copy runs of 2-6 consecutive whole days from the source
     window (keeps day-to-day persistence: heat waves, cool spells, rainy spells).
  2. Perturb each day, with all variables staying physically coupled:
       - dry bulb: smooth AR(1) offset + random diurnal-amplitude scaling
       - dew point: follows dry bulb partly + own AR(1) offset, capped at dry bulb;
         relative humidity is recomputed from T and Tdp (Magnus)
       - wind speed: log-normal scaling per block; wind direction rotated per block
       - solar: DNI/DHI scaled by a cloudiness factor, GHI recomputed from
         GHI = DHI + DNI*cos(z) (cos(z) taken from the source hour), sky cover adjusted
       - horizontal IR scaled with T^4, illuminances scaled with their irradiances
  3. Smooth the jumps at block boundaries (T, Tdp) with a short linear cross-fade.

Usage:  python augment_epw.py in.epw out.epw [--seed 42]
"""
import argparse
import numpy as np
import pandas as pd

# EPW column indices (0-based) of the 35 data columns
C = dict(year=0, month=1, day=2, hour=3, minute=4, flags=5, T=6, Td=7, RH=8, P=9,
         ETR_h=10, ETR_n=11, HIR=12, GHI=13, DNI=14, DHI=15, Eg=16, En=17, Ed=18, Lz=19,
         wdir=20, wspd=21, sky=22, opaque=23, vis=24, ceil=25, pwo=26, pwc=27, pw=28,
         aod=29, snow=30, dsnow=31, albedo=32, rain=33, rainq=34)
FMT = {6: "{:.1f}", 7: "{:.1f}", 8: "{:.0f}", 9: "{:.0f}", 12: "{:.0f}", 13: "{:.0f}",
       14: "{:.0f}", 15: "{:.0f}", 16: "{:.0f}", 17: "{:.0f}", 18: "{:.0f}", 19: "{:.0f}",
       20: "{:.0f}", 21: "{:.1f}", 22: "{:.0f}", 23: "{:.0f}"}


def ar1(n, sigma, phi, rng):
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
    return x


def smooth_daily(vals):
    """daily values -> hourly curve, linear between day centres (no midnight jumps)."""
    n = len(vals)
    centres = np.arange(n) * 24 + 11.5
    return np.interp(np.arange(n * 24), centres, vals)


def rh_from_td(T, Td):
    a, b = 17.625, 243.04
    return np.clip(100 * np.exp(a * Td / (b + Td) - a * T / (b + T)), 1, 100)


def crossfade(x, bounds, width=6):
    for b in bounds:
        jump = x[b] - x[b - 1]
        for k in range(width):
            if b + k < len(x):
                x[b + k] -= jump * (1 - k / width)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--start", type=int, default=5088)
    ap.add_argument("--end", type=int, default=6551)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-min", type=int, default=2)
    ap.add_argument("--block-max", type=int, default=6)
    ap.add_argument("--t-sigma", type=float, default=2.0, help="K, daily temp offset std")
    ap.add_argument("--td-sigma", type=float, default=1.5, help="K, extra dew-point offset std")
    ap.add_argument("--amp", type=float, default=0.15, help="diurnal amplitude scale +-")
    ap.add_argument("--wind-sigma", type=float, default=0.25)
    ap.add_argument("--solar-sigma", type=float, default=0.12)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    with open(a.src, "r", encoding="latin-1", newline="") as f:
        lines = f.read().split("\n")
    header = [l.rstrip("\r") for l in lines[:8]]
    df = pd.read_csv(a.src, skiprows=8, header=None, encoding="latin-1")
    assert len(df) == 8760, len(df)

    src = df.iloc[a.start:a.end + 1].reset_index(drop=True)
    nsd = len(src) // 24
    src_days = [src.iloc[d * 24:(d + 1) * 24].reset_index(drop=True) for d in range(nsd)]

    # ---- 1. block bootstrap of source days -> sequence of 365 source-day indices
    seq, block_id = [], []
    b = 0
    while len(seq) < 365:
        L = int(rng.integers(a.block_min, a.block_max + 1))
        s = int(rng.integers(0, nsd - L + 1))
        for i in range(L):
            if len(seq) < 365:
                seq.append(s + i); block_id.append(b)
        b += 1
    seq, block_id = np.array(seq), np.array(block_id)
    out = pd.concat([src_days[i] for i in seq], ignore_index=True)
    bounds = [d * 24 for d in range(1, 365) if block_id[d] != block_id[d - 1]]
    N = len(out)

    # keep the target calendar / flags / snow info of the original file
    for c in (0, 1, 2, 3, 4, 5, 30, 31):
        out[c] = df[c].values
    out[30] = 0

    # ---- 2. perturbations
    T0 = out[C["T"]].to_numpy(float); Td0 = out[C["Td"]].to_numpy(float)
    day_mean = T0.reshape(365, 24).mean(1)
    dev = T0 - np.repeat(day_mean, 24)
    amp = np.repeat(rng.uniform(1 - a.amp, 1 + a.amp, 365), 24)
    t_off = smooth_daily(ar1(365, a.t_sigma, 0.7, rng))
    td_off = smooth_daily(ar1(365, a.td_sigma, 0.6, rng))
    T = np.repeat(day_mean, 24) + amp * dev + t_off
    Td = Td0 + 0.7 * t_off + td_off
    T = crossfade(T, bounds); Td = crossfade(Td, bounds)
    Td = np.minimum(Td, T - 0.1)
    out[C["T"]] = T; out[C["Td"]] = Td; out[C["RH"]] = rh_from_td(T, Td)
    out[C["P"]] = out[C["P"]].to_numpy(float) + smooth_daily(ar1(365, 150, 0.8, rng))
    out[C["HIR"]] = out[C["HIR"]].to_numpy(float) * ((T + 273.15) / (T0 + 273.15)) ** 4

    # wind: one factor / rotation per block
    nb = block_id.max() + 1
    wf = np.exp(rng.normal(0, a.wind_sigma, nb))[block_id].repeat(24)
    rot = rng.normal(0, 40, nb)[block_id].repeat(24)
    ws = out[C["wspd"]].to_numpy(float)
    out[C["wspd"]] = np.clip(ws * wf, 0, ws.max() * 1.3)
    wd = out[C["wdir"]].to_numpy(float)
    out[C["wdir"]] = np.where(ws > 0, (wd + rot) % 360, wd)

    # solar: daily cloudiness factor k (k<1 cloudier)
    k = np.clip(rng.normal(1.0, a.solar_sigma, 365), 0.6, 1.2)
    k = np.repeat(k, 24)
    ghi, dni, dhi = (out[C[x]].to_numpy(float) for x in ("GHI", "DNI", "DHI"))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosz = np.where(dni > 20, np.clip((ghi - dhi) / dni, 0, 1), 0)
    dni_n = np.clip(dni * k, 0, 1000)
    dhi_n = dhi * (1 + 0.5 * (1 - k))
    ghi_n = np.where(dni > 20, dhi_n + dni_n * cosz, ghi * k)
    ghi_n = np.minimum(ghi_n, np.maximum(out[C["ETR_h"]].to_numpy(float), ghi))
    for name, new, old in (("GHI", ghi_n, ghi), ("DNI", dni_n, dni), ("DHI", dhi_n, dhi)):
        out[C[name]] = new
        ratio = np.where(old > 0, new / np.where(old > 0, old, 1), 1)
        col = {"GHI": "Eg", "DNI": "En", "DHI": "Ed"}[name]
        out[C[col]] = out[C[col]].to_numpy(float) * ratio
        if name == "DHI":
            out[C["Lz"]] = out[C["Lz"]].to_numpy(float) * ratio
    for c in ("sky", "opaque"):
        out[C[c]] = np.clip(np.round(out[C[c]].to_numpy(float) + (1 - k) * 8), 0, 10)

    # ---- 3. write EPW
    header[6] = header[6].rstrip('"') + (f' | AUGMENTED from {a.start}-{a.end} (Aug-Sep) by '
                                         f'block bootstrap + perturbation, seed={a.seed}"')
    with open(a.dst, "w", encoding="latin-1", newline="") as f:
        f.write("\r\n".join(header) + "\r\n")
        for row in out.itertuples(index=False, name=None):
            cells = []
            for j, v in enumerate(row):
                if j in FMT:
                    cells.append(FMT[j].format(float(v)))
                elif j in (0, 1, 2, 3, 4, 30, 31):
                    cells.append(str(int(float(v))))
                else:
                    cells.append(str(v))
            f.write(",".join(cells) + "\r\n")
    print("wrote", a.dst, N, "rows")


if __name__ == "__main__":
    main()