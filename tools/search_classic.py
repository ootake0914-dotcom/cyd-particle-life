#!/usr/bin/env python3
"""hunar4321/particle-life の some_patterns.jpg 風の古典パターンを探索する。

ファーム(cyd-particle-life.ino)の物理へ厳密に合わせる:
  * ワールド 320x170、相互作用半径R=60（タイプ別半径も可、ただし<=60）
  * 壁バネ: 縁12px、ばね定数0.1、1回反射
  * 減衰: v = v*(1-viscosity) + force （力= g * inv(d) * d、gは[-1,1]、負=引力）
  * 粒子数900、タイプ数6

アーキタイプ（some_patterns.jpg の典型的パターン）:
  cells     : 核と膜を持つ丸い多色生物
  snake     : 循環追跡でできる動くチェーン/ワーム
  orbit     : 非対称引力で相互周回する回転塊
  membrane  : 核をタイプが取り囲む膜・殻構造

例:
  python tools/search_classic.py search --archetype cells --samples 36
  python tools/search_classic.py evaluate --matrix "...36 values..." --viscosity 0.7
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

WORLD_W = 320.0
WORLD_H = 170.0
WALL = 12.0
WALL_K = 0.1
NUM_TYPES = 6
TOTAL_PARTICLES = 900

COUNT_PROFILES = [
    (150, 150, 150, 150, 150, 150),
    (210, 210, 160, 160, 80, 80),
    (260, 260, 130, 130, 60, 60),
    (130, 130, 130, 130, 190, 190),
]


@dataclass
class Candidate:
    score: float
    rules: np.ndarray
    viscosity: float
    radius: float
    counts: tuple
    metrics: dict
    name: str = "candidate"


def init_particles(rng, counts):
    n = int(sum(counts))
    x = rng.uniform(0.0, WORLD_W, n).astype(np.float32)
    y = rng.uniform(0.0, WORLD_H, n).astype(np.float32)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    types = np.repeat(np.arange(NUM_TYPES), counts).astype(np.int8)
    return x, y, vx, vy, types


def advance(x, y, vx, vy, gij, damp, r2max):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    mask = d2 < r2max
    np.fill_diagonal(mask, False)

    inv = np.zeros_like(d2)
    np.divide(1.0, np.sqrt(np.maximum(d2, 1e-12)), out=inv, where=mask)
    force = gij * inv
    fx = (force * dx).sum(axis=1)
    fy = (force * dy).sum(axis=1)

    fx += np.where(x < WALL, (WALL - x) * WALL_K, 0.0)
    fx += np.where(x > WORLD_W - WALL, (WORLD_W - WALL - x) * WALL_K, 0.0)
    fy += np.where(y < WALL, (WALL - y) * WALL_K, 0.0)
    fy += np.where(y > WORLD_H - WALL, (WORLD_H - WALL - y) * WALL_K, 0.0)

    vx = vx * damp + fx
    vy = vy * damp + fy
    x = x + vx
    y = y + vy

    lo = x < 0.0
    hi = x >= WORLD_W
    x = np.where(lo, -x, x)
    x = np.where(hi, 2.0 * WORLD_W - x, x)
    vx = np.where(lo | hi, -vx, vx)

    lo = y < 0.0
    hi = y >= WORLD_H
    y = np.where(lo, -y, y)
    y = np.where(hi, 2.0 * WORLD_H - y, y)
    vy = np.where(lo | hi, -vy, vy)
    return x, y, vx, vy


def safe_state(x, y, vx, vy):
    return (
        np.isfinite(x).all() and np.isfinite(y).all()
        and np.isfinite(vx).all() and np.isfinite(vy).all()
        and (x >= 0.0).all() and (x < WORLD_W).all()
        and (y >= 0.0).all() and (y < WORLD_H).all()
        and (np.abs(vx) < 2.0 * WORLD_W).all()
        and (np.abs(vy) < 2.0 * WORLD_H).all()
    )


def cluster_labels(x, y, connect):
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    close = np.triu(dx * dx + dy * dy < connect * connect, 1)
    rows, cols = np.where(close)
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(rows, cols):
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[rb] = ra
    labels = np.array([root(i) for i in range(n)], dtype=np.int32)
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def measure(x, y, vx, vy, types, x0, y0, radius):
    n = len(x)
    r2 = radius * radius
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    near = d2 < r2
    np.fill_diagonal(near, False)

    neighbors = near.sum(axis=1).astype(np.float32)
    same = (near & (types[:, None] == types[None, :])).sum(axis=1)
    same_fraction = np.divide(
        same, neighbors, out=np.zeros(n, dtype=np.float32), where=neighbors > 0
    )
    diff_fraction = 1.0 - same_fraction

    connect = max(18.0, radius * 0.45)
    labels = cluster_labels(x, y, connect)
    counts = np.bincount(labels)
    min_blob = max(10, n // 60)
    big = np.where(counts >= min_blob)[0]
    num_blobs = len(big)
    clustered = np.isin(labels, big)
    frac_clustered = float(clustered.mean()) if num_blobs else 0.0
    largest_frac = float(counts.max()) / n if num_blobs else 1.0

    aspects, omegas, spins, shells = [], [], [], []
    for b in big:
        m = labels == b
        if m.sum() < min_blob:
            continue
        bx = x[m] - x[m].mean()
        by = y[m] - y[m].mean()
        cov = np.cov(np.stack([bx, by]))
        evals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-9))
        aspect = float(np.sqrt(evals[1] / evals[0]))
        aspects.append((aspect, int(m.sum())))

        rx, ryv = bx, by
        cross = rx * vy[m] - ryv * vx[m]
        rr = rx * rx + ryv * ryv
        denom = float(rr.sum())
        if denom > 1e-6 and m.sum() >= 8:
            omega = float(cross.sum()) / denom
            abs_cross = np.abs(cross).sum()
            omegas.append((omega, int(m.sum())))
            if abs_cross > 1e-6:
                spins.append((abs(float(cross.sum())) / float(abs_cross), int(m.sum())))

        rms = math.sqrt(denom / m.sum())
        t_radii = []
        for t in np.unique(types[m]):
            tm = m & (types == t)
            if tm.sum() < 6:
                continue
            tr = math.sqrt(float(((x[tm] - x[m].mean()) ** 2
                                  + (y[tm] - y[m].mean()) ** 2).mean()))
            t_radii.append((tr, int(tm.sum())))
        if len(t_radii) >= 2 and rms > 1e-6:
            t_radii.sort()
            span = (t_radii[-1][0] - t_radii[0][0]) / rms
            shells.append((span, int(m.sum())))

    def wmean(pairs):
        if not pairs:
            return 0.0
        w = sum(c for _, c in pairs)
        return sum(v * c for v, c in pairs) / w if w else 0.0

    aspect = wmean(aspects)
    abs_omega = wmean([(abs(o), c) for o, c in omegas])
    spin = wmean(spins)
    shell = wmean(shells)

    displacement = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    cluster_move = float(displacement[clustered].mean() / radius) if clustered.any() else 0.0

    speed = float(np.sqrt(vx * vx + vy * vy).mean())
    spread = float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)))

    return {
        "num_blobs": num_blobs,
        "largest_frac": largest_frac,
        "frac_clustered": frac_clustered,
        "mix": float(diff_fraction.mean()),
        "aspect": aspect,
        "abs_omega": abs_omega,
        "spin": spin,
        "shell": shell,
        "cluster_move": cluster_move,
        "mean_speed": speed,
        "spread": spread,
    }


def gauss(v, center, sigma):
    return math.exp(-((v - center) / sigma) ** 2)


def clip01(v):
    return float(np.clip(v, 0.0, 1.0))


def score_cells(m):
    colony = clip01(m["frac_clustered"] / 0.72) * clip01((0.55 - m["largest_frac"]) / 0.2)
    blobs = gauss(m["num_blobs"], 6.0, 4.0)
    roundness = gauss(m["aspect"], 1.6, 1.1)
    mixing = gauss(m["mix"], 0.5, 0.32)
    motion = gauss(m["cluster_move"], 0.12, 0.16)
    return float(0.24 * colony + 0.14 * blobs + 0.20 * roundness
                 + 0.22 * mixing + 0.20 * motion)


def score_snake(m):
    coherence_blob = gauss(m["num_blobs"], 6.0, 5.0)
    elong = clip01((m["aspect"] - 1.8) / 2.0)
    motion = clip01((m["cluster_move"] - 0.18) / 0.5)
    cohesion = clip01(m["frac_clustered"] / 0.65) * clip01((0.65 - m["largest_frac"]) / 0.25)
    return float(0.30 * elong + 0.30 * motion + 0.22 * cohesion + 0.18 * coherence_blob)


def score_orbit(m):
    spin_amt = clip01(m["abs_omega"] / 0.012)
    spin_cons = clip01((m["spin"] - 0.30) / 0.45)
    cohesion = clip01(m["frac_clustered"] / 0.62) * clip01((0.55 - m["largest_frac"]) / 0.25)
    blobs = gauss(m["num_blobs"], 4.0, 3.5)
    motion = gauss(m["cluster_move"], 0.25, 0.35)
    return float(0.30 * spin_amt + 0.28 * spin_cons + 0.22 * cohesion
                 + 0.10 * blobs + 0.10 * motion)


def score_membrane(m):
    shell = clip01((m["shell"] - 0.25) / 0.35)
    colony = clip01(m["frac_clustered"] / 0.70) * clip01((0.55 - m["largest_frac"]) / 0.2)
    roundness = gauss(m["aspect"], 1.7, 1.2)
    mixing = gauss(m["mix"], 0.45, 0.35)
    blobs = gauss(m["num_blobs"], 5.0, 4.0)
    return float(0.32 * shell + 0.24 * colony + 0.18 * roundness
                 + 0.14 * mixing + 0.12 * blobs)


def score_mitosis(m):
    """多数の小さな丸い細胞が分かれている状態を評価する。"""
    blobs = gauss(m["num_blobs"], 11.0, 4.5)
    small = clip01((0.22 - m["largest_frac"]) / 0.12)
    colony = clip01(m["frac_clustered"] / 0.75)
    roundness = gauss(m["aspect"], 1.5, 0.9)
    motion = gauss(m["cluster_move"], 0.10, 0.15)
    mixing = gauss(m["mix"], 0.5, 0.30)
    return float(0.24 * blobs + 0.20 * small + 0.18 * colony
                 + 0.14 * roundness + 0.12 * motion + 0.12 * mixing)


def score_vortex(m):
    """強く回転する塊を評価する。"""
    spin_amt = clip01(m["abs_omega"] / 0.02)
    spin_cons = clip01((m["spin"] - 0.45) / 0.40)
    cohesion = clip01(m["frac_clustered"] / 0.60) * clip01((0.60 - m["largest_frac"]) / 0.30)
    roundness = gauss(m["aspect"], 1.6, 1.0)
    motion = gauss(m["cluster_move"], 0.30, 0.40)
    return float(0.32 * spin_amt + 0.30 * spin_cons + 0.20 * cohesion
                 + 0.10 * roundness + 0.08 * motion)


def score_gliders(m):
    """高速で移動し続ける塊を評価する。"""
    motion = clip01((m["cluster_move"] - 0.25) / 0.60)
    speed = clip01((m["mean_speed"] - 8.0) / 18.0)
    cohesion = clip01(m["frac_clustered"] / 0.60) * clip01((0.62 - m["largest_frac"]) / 0.30)
    blobs = gauss(m["num_blobs"], 5.0, 4.0)
    return float(0.34 * motion + 0.22 * speed + 0.28 * cohesion + 0.16 * blobs)


def score_crystals(m):
    """ほぼ静止した整った構造を評価する。"""
    still = gauss(m["cluster_move"], 0.02, 0.06)
    slow = gauss(m["mean_speed"], 3.0, 4.0)
    colony = clip01(m["frac_clustered"] / 0.80)
    blobs = gauss(m["num_blobs"], 7.0, 5.0)
    roundness = gauss(m["aspect"], 1.4, 0.9)
    return float(0.30 * still + 0.20 * slow + 0.25 * colony
                 + 0.15 * blobs + 0.10 * roundness)


def score_amoeba(m):
    """大きな塊がゆっくり形を変える状態を評価する。"""
    big = gauss(m["largest_frac"], 0.55, 0.18)
    colony = clip01(m["frac_clustered"] / 0.85)
    slow = gauss(m["cluster_move"], 0.10, 0.12)
    shape = gauss(m["aspect"], 2.0, 1.0)
    shell = clip01(m["shell"] / 0.60)
    return float(0.30 * big + 0.22 * colony + 0.18 * slow
                 + 0.15 * shape + 0.15 * shell)


SCORERS = {
    "cells": score_cells,
    "snake": score_snake,
    "orbit": score_orbit,
    "membrane": score_membrane,
    "mitosis": score_mitosis,
    "vortex": score_vortex,
    "gliders": score_gliders,
    "crystals": score_crystals,
    "amoeba": score_amoeba,
}

VISCOSITY_RANGE = {
    "cells": (0.60, 0.75),
    "snake": (0.32, 0.50),
    "orbit": (0.42, 0.62),
    "membrane": (0.60, 0.75),
    "mitosis": (0.55, 0.75),
    "vortex": (0.40, 0.60),
    "gliders": (0.40, 0.58),
    "crystals": (0.65, 0.80),
    "amoeba": (0.55, 0.72),
}

RADIUS_CHOICES = {
    "cells": (60.0, 55.0, 50.0),
    "snake": (60.0, 50.0, 42.0, 36.0),
    "orbit": (60.0, 55.0, 50.0),
    "membrane": (60.0, 55.0, 50.0),
    "mitosis": (60.0, 55.0, 50.0),
    "vortex": (60.0, 55.0, 50.0),
    "gliders": (60.0, 50.0, 42.0),
    "crystals": (60.0, 50.0, 40.0),
    "amoeba": (60.0, 55.0),
}


def gen_cells(rng):
    g = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    for i in range(NUM_TYPES):
        g[i, i] = rng.uniform(-0.55, 0.05)
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            if i != j and g[i, j] == 0.0 and g[j, i] == 0.0:
                if rng.random() < 0.72:
                    g[i, j] = rng.uniform(-0.98, -0.15)
                    g[j, i] = rng.uniform(0.10, 0.98)
                else:
                    g[i, j] = rng.uniform(-0.6, 0.9)
                    g[j, i] = rng.uniform(-0.6, 0.9)
    return np.clip(g, -1.0, 1.0)


def gen_snake(rng):
    g = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    rev = rng.random() < 0.5
    for i in range(NUM_TYPES):
        g[i, i] = rng.uniform(-0.35, -0.08)
        nxt = (i - 1) % NUM_TYPES if rev else (i + 1) % NUM_TYPES
        g[i, nxt] = rng.uniform(-0.95, -0.30)
        g[nxt, i] = rng.uniform(0.15, 0.9)
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            if i != j and g[i, j] == 0.0:
                g[i, j] = rng.uniform(-0.35, 0.35)
    return np.clip(g, -1.0, 1.0)


def gen_orbit(rng):
    g = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    order = rng.permutation(NUM_TYPES)
    for i in range(NUM_TYPES):
        g[i, i] = rng.uniform(-0.40, 0.05)
    for k in range(0, NUM_TYPES - 1, 2):
        a, b = int(order[k]), int(order[k + 1])
        g[a, b] = rng.uniform(-0.95, -0.30)
        g[b, a] = rng.uniform(-0.25, 0.55)
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            if i != j and g[i, j] == 0.0:
                g[i, j] = rng.uniform(-0.45, 0.45)
    return np.clip(g, -1.0, 1.0)


def gen_membrane(rng):
    g = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    perm = rng.permutation(NUM_TYPES)
    n_shell = int(rng.integers(2, 4))
    shells = set(int(t) for t in perm[:n_shell])
    cores = set(int(t) for t in perm[n_shell:])
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            if i == j:
                g[i, i] = rng.uniform(-0.50, -0.08)
            elif i in shells and j in cores:
                g[i, j] = rng.uniform(0.05, 0.70)
            elif i in cores and j in shells:
                g[i, j] = rng.uniform(-0.85, -0.20)
            else:
                g[i, j] = rng.uniform(-0.45, 0.35)
    return np.clip(g, -1.0, 1.0)


def gen_crystals(rng):
    """対称で穏やかなルール。追跡項がないため静止した整った構造が生じる。"""
    g = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
    for i in range(NUM_TYPES):
        for j in range(i, NUM_TYPES):
            if i == j:
                v = rng.uniform(-0.32, -0.05)
            else:
                v = rng.uniform(-0.25, 0.38)
            g[i, j] = v
            g[j, i] = v
    return np.clip(g, -1.0, 1.0)


GENERATORS = {
    "cells": gen_cells,
    "snake": gen_snake,
    "orbit": gen_orbit,
    "membrane": gen_membrane,
    "mitosis": gen_cells,
    "vortex": gen_orbit,
    "gliders": gen_cells,
    "crystals": gen_crystals,
    "amoeba": gen_membrane,
}


def mutate(parent, rng, sigma, big_jump=False):
    g = parent.copy()
    scale = sigma * (2.4 if big_jump else 1.0)
    mask = rng.random(g.shape) < (0.45 if big_jump else 0.70)
    g[mask] += rng.normal(0.0, scale, size=int(mask.sum())).astype(np.float32)
    if rng.random() < 0.35:
        g[np.diag_indices(NUM_TYPES)] += rng.normal(0.0, scale * 0.7, NUM_TYPES)
    return np.clip(g, -1.0, 1.0).astype(np.float32)


def simulate(rules, viscosity, counts, radius, settle, track, seed):
    rng = np.random.default_rng(seed)
    x, y, vx, vy, types = init_particles(rng, counts)
    gij = rules[types[:, None], types[None, :]].astype(np.float32)
    damp = 1.0 - float(viscosity)
    r2max = radius * radius

    for _ in range(settle):
        x, y, vx, vy = advance(x, y, vx, vy, gij, damp, r2max)
        if not safe_state(x, y, vx, vy):
            return None
    x0, y0 = x.copy(), y.copy()
    for _ in range(track):
        x, y, vx, vy = advance(x, y, vx, vy, gij, damp, r2max)
        if not safe_state(x, y, vx, vy):
            return None
    metrics = measure(x, y, vx, vy, types, x0, y0, radius)
    return metrics, x, y, types


def scale_counts(profile, total=TOTAL_PARTICLES):
    s = sum(profile)
    counts = [int(round(c * total / s)) for c in profile]
    counts[-1] += total - sum(counts)
    return tuple(counts)


def evaluate(rules, viscosity, radius, counts, archetype, settle, track, seed):
    result = simulate(rules, viscosity, counts, radius, settle, track, seed)
    if result is None:
        return -1.0, {}
    metrics, *_ = result
    return SCORERS[archetype](metrics), metrics


def evaluate_robust(rules, viscosity, radius, counts, archetype, args, seeds):
    scores, merged = [], {}
    for seed in seeds:
        s, m = evaluate(rules, viscosity, radius, counts, archetype,
                        args.settle, args.track, seed)
        if s < 0.0:
            return -1.0, {}
        scores.append(s)
        for k, v in m.items():
            merged.setdefault(k, []).append(v)
    metrics = {k: float(np.mean(v)) for k, v in merged.items()}
    return float(np.mean(scores)), metrics


def print_candidate(rank, c):
    m = c.metrics
    print(
        f"#{rank} score={c.score:.4f} visc={c.viscosity:.2f} r={c.radius:.0f} "
        f"blobs={m.get('num_blobs', 0):4.1f} "
        f"cluster={m.get('frac_clustered', 0.0):.2f} "
        f"largest={m.get('largest_frac', 0.0):.2f} "
        f"mix={m.get('mix', 0.0):.2f} aspect={m.get('aspect', 0.0):.2f} "
        f"spin={m.get('spin', 0.0):.2f} w={m.get('abs_omega', 0.0):.5f} "
        f"shell={m.get('shell', 0.0):.2f} move={m.get('cluster_move', 0.0):.3f}R"
    )


def cpp_block(c, name):
    lines = [f"    {{ // {name} (score {c.score:.3f}, visc {c.viscosity:.2f}, "
             f"r{c.radius:.0f})"]
    for row in c.rules:
        lines.append("      { " + ", ".join(f"{v:+.3f}f" for v in row) + " },")
    lines.append("    },")
    return "\n".join(lines)


def counts_block(c):
    return "{" + ", ".join(str(v) for v in c.counts) + "}"


def save_preview(c, path, seed, settle=300, track=40):
    try:
        from PIL import Image
    except ImportError:
        return
    result = simulate(c.rules, c.viscosity, c.counts, c.radius, settle, track, seed)
    if result is None:
        return
    _, x, y, types = result
    palette = [(60, 255, 80), (255, 60, 60), (255, 220, 60),
               (60, 220, 255), (255, 60, 255), (90, 110, 255)]
    img = Image.new("RGB", (int(WORLD_W), int(WORLD_H)), (0, 0, 0))
    px = img.load()
    for xi, yi, ti in zip(x, y, types):
        px[int(np.clip(xi, 0, WORLD_W - 1)), int(np.clip(yi, 0, WORLD_H - 1))] = palette[int(ti)]
    img.save(path)
    print(f"preview -> {path}")


def search(args):
    archetype = args.archetype
    rng = np.random.default_rng(args.seed)
    generator = GENERATORS[archetype]
    visc_lo, visc_hi = VISCOSITY_RANGE[archetype]
    radii = RADIUS_CHOICES[archetype]
    seeds = tuple(args.eval_seed + i for i in range(args.robust_seeds))
    scale = args.particle_scale

    def make_counts():
        profile = COUNT_PROFILES[int(rng.integers(0, len(COUNT_PROFILES)))]
        return scale_counts(profile, int(TOTAL_PARTICLES * scale))

    ranked = []
    for k in range(args.samples):
        rules = generator(rng)
        viscosity = float(rng.uniform(visc_lo, visc_hi))
        radius = float(radii[int(rng.integers(0, len(radii)))])
        counts = make_counts()
        score, metrics = evaluate_robust(
            rules, viscosity, radius, counts, archetype, args, seeds)
        ranked.append(Candidate(score, rules, viscosity, radius, counts,
                                metrics, f"{archetype}-rand-{k}"))
        if (k + 1) % 6 == 0:
            print(f"  sampled {k + 1}/{args.samples}", flush=True)

    ranked.sort(key=lambda c: c.score, reverse=True)
    elites = ranked[:max(4, args.elites)]

    for generation in range(args.generations):
        sigma = 0.15 * (0.72 ** generation)
        population = list(elites)
        for k in range(args.children):
            parent = elites[k % len(elites)]
            big_jump = rng.random() < 0.12
            rules = mutate(parent.rules, rng, sigma, big_jump)
            viscosity = float(np.clip(parent.viscosity + rng.normal(0.0, sigma * 0.35),
                                      visc_lo, visc_hi))
            radius = parent.radius
            if rng.random() < 0.18:
                radius = float(radii[int(rng.integers(0, len(radii)))])
            counts = parent.counts
            if rng.random() < 0.15:
                counts = make_counts()
            score, metrics = evaluate_robust(
                rules, viscosity, radius, counts, archetype, args, seeds)
            population.append(Candidate(score, rules, viscosity, radius, counts,
                                        metrics, f"{archetype}-gen{generation + 1}-{k}"))
        population.sort(key=lambda c: c.score, reverse=True)
        elites = population[:max(4, args.elites)]
        print(f"generation {generation + 1}: best={elites[0].score:.4f}", flush=True)

    print(f"\nVALIDATING TOP {args.validate_top} at full particle count")
    validated = []
    for c in elites[:args.validate_top]:
        full_counts = scale_counts(c.counts)
        score, metrics = evaluate_robust(
            c.rules, c.viscosity, c.radius, full_counts, archetype, args,
            args.validation_seeds)
        validated.append(Candidate(score, c.rules, c.viscosity, c.radius,
                                   full_counts, metrics, c.name))
    validated.sort(key=lambda c: c.score, reverse=True)

    for rank, c in enumerate(validated, 1):
        print_candidate(rank, c)
        print(cpp_block(c, f"{archetype}{rank}"))
        print(f"counts = {counts_block(c)}")
        if args.preview_dir:
            save_preview(c, f"{args.preview_dir}_{archetype}_{rank}.png",
                         args.validation_seeds[0])
    print("SEARCH DONE")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search")
    s.add_argument("--archetype", required=True,
                   choices=["cells", "snake", "orbit", "membrane", "mitosis",
                            "vortex", "gliders", "crystals", "amoeba"])
    s.add_argument("--samples", type=int, default=40)
    s.add_argument("--generations", type=int, default=2)
    s.add_argument("--children", type=int, default=14)
    s.add_argument("--elites", type=int, default=6)
    s.add_argument("--settle", type=int, default=150)
    s.add_argument("--track", type=int, default=45)
    s.add_argument("--particle-scale", type=float, default=0.6)
    s.add_argument("--validate-top", type=int, default=3)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--eval-seed", type=int, default=11)
    s.add_argument("--robust-seeds", type=int, default=2)
    s.add_argument("--preview-dir", default="")
    s.set_defaults(validation_seeds=(0, 1, 2))

    e = sub.add_parser("evaluate")
    e.add_argument("--matrix", required=True)
    e.add_argument("--viscosity", type=float, default=0.7)
    e.add_argument("--radius", type=float, default=60.0)
    e.add_argument("--counts", default="150,150,150,150,150,150")
    e.add_argument("--settle", type=int, default=300)
    e.add_argument("--track", type=int, default=45)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--save", default="")

    p = sub.add_parser("preview")
    p.add_argument("--matrix", required=True)
    p.add_argument("--viscosity", type=float, default=0.7)
    p.add_argument("--radius", type=float, default=60.0)
    p.add_argument("--counts", default="150,150,150,150,150,150")
    p.add_argument("--settle", type=int, default=300)
    p.add_argument("--track", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "search":
        search(args)
        return

    values = [float(v) for v in args.matrix.replace(",", " ").split()]
    if len(values) != NUM_TYPES * NUM_TYPES:
        raise SystemExit(f"--matrix needs exactly {NUM_TYPES * NUM_TYPES} values")
    rules = np.array(values, dtype=np.float32).reshape(NUM_TYPES, NUM_TYPES)
    counts = tuple(int(v) for v in args.counts.split(","))
    if args.command == "evaluate":
        for archetype, scorer in SCORERS.items():
            score, m = evaluate(rules, args.viscosity, args.radius, counts,
                                archetype, args.settle, args.track, args.seed)
            print(f"--- {archetype}: score={score:.4f}")
            for k, v in m.items():
                print(f"  {k}={v:.4f}")
        return

    c = Candidate(0.0, rules, args.viscosity, args.radius, counts, {})
    save_preview(c, args.save, args.seed, args.settle, args.track)


if __name__ == "__main__":
    main()
