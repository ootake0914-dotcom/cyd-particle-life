#!/usr/bin/env python3
"""現行ファームモデル向けに、ステージごとの3色粒子数を探索する。"""
import numpy as np
import find_rules as fr


WORLD_W, WORLD_H = 320, 170
R = 60.0
N = 900
NUM_TYPES = 3
COARSE_CANDIDATES = 16
COARSE_STEPS = 70
VALIDATE_STEPS = 200
VALIDATION_SEEDS = (0, 1, 2)

STAGES = (
    "classic", "colony1", "colony2", "swim",
    "swarm", "garden", "islands", "py-garden", "py-virus",
)

RULES = np.array([
    [[-0.32, -0.17,  0.34], [-0.34, -0.10,  0.00], [-0.20,  0.00,  0.15]],
    [[-0.31, -0.98,  0.14], [-0.78, -0.29,  0.51], [-0.65,  0.88, -0.44]],
    [[-0.47,  0.56,  0.51], [ 0.71, -0.52,  0.50], [-0.68,  0.53, -0.26]],
    [[-0.44,  0.88, -0.66], [ 0.63, -0.38,  0.55], [ 0.69,  0.59, -0.29]],
    [[ 0.04, -0.43, -0.35], [ 0.33, -0.48, -0.92], [ 0.13,  0.90, -0.26]],
    [[-0.49, -0.31,  0.25], [ 0.33, -0.47, -0.93], [-0.24,  0.65, -0.28]],
    [[-0.31, -0.70,  0.24], [ 0.50, -0.37, -0.88], [-0.02,  0.48, -0.31]],
    [[-0.47,  0.77, -0.33], [-0.25, -0.30,  0.57], [ 0.38,  0.82, -0.34]],
    [[-0.447, 0.770,  0.184], [-0.353, 0.101, 0.485], [ 0.624, 0.199, 0.158]],
], dtype=np.float32)


fr.R = R
fr.R2 = R * R
fr.N = N
fr.PER_TYPE = N // NUM_TYPES
fr.NUM_TYPES = NUM_TYPES


def simulate_counts(rules, counts, steps, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, WORLD_W, N).astype(np.float32)
    y = rng.uniform(0, WORLD_H, N).astype(np.float32)
    vx = np.zeros(N, np.float32)
    vy = np.zeros(N, np.float32)
    types = np.repeat(np.arange(NUM_TYPES), counts).astype(np.int8)
    gij = rules[types[:, None], types[None, :]].astype(np.float32)
    for _ in range(steps):
        x, y, vx, vy = fr.step(x, y, vx, vy, gij)
    return fr.metrics(x, y, types)


def visual_score(metrics, stage):
    colony = fr.score(metrics)
    membrane = fr.membrane_score(metrics)
    # classicとcolony1は核・膜形状の候補なので膜スコアを重くする。
    if stage < 2:
        return 0.70 * membrane + 0.30 * colony
    return 0.30 * membrane + 0.70 * colony


def ratio_candidates(rng):
    result = {(300, 300, 300)}
    while len(result) < COARSE_CANDIDATES:
        parts = rng.multinomial(24, rng.dirichlet(np.ones(NUM_TYPES))) + 2
        result.add(tuple((parts * 30).tolist()))
    return sorted(result)


def validate(stage, counts):
    values = []
    blobs = []
    segs = []
    for seed in VALIDATION_SEEDS:
        metrics = simulate_counts(RULES[stage], counts, VALIDATE_STEPS, seed)
        values.append(visual_score(metrics, stage))
        blobs.append(metrics["num_blobs"])
        segs.append(metrics["segregation"])
    return float(np.mean(values)), blobs, float(np.mean(segs))


def main():
    rng = np.random.default_rng(9025)
    selected = []
    for stage, name in enumerate(STAGES):
        candidates = ratio_candidates(rng)
        coarse = []
        for counts in candidates:
            metrics = simulate_counts(RULES[stage], counts, COARSE_STEPS, 1000 + stage)
            coarse.append((visual_score(metrics, stage), counts))
        coarse.sort(reverse=True)
        best = max(
            (validate(stage, counts)[0], counts)
            for _, counts in coarse[:4]
        )
        score, counts = best
        full_score, blobs, segregation = validate(stage, counts)
        selected.append(counts)
        print(
            f"stage {stage} {name:8s}: counts={counts} "
            f"score={full_score:.4f} blobs={blobs} seg={segregation:.3f}",
            flush=True,
        )

    print("\nC++ counts:")
    for stage, counts in enumerate(selected):
        print(f"  {{ {counts[0]:3d}, {counts[1]:3d}, {counts[2]:3d} }}, // ステージ{stage} {STAGES[stage]}")


if __name__ == "__main__":
    main()
