#!/usr/bin/env python3
"""
実験: 粒子数と色数を変えて比較する。

使い方:
  python experiment.py            # 固定した設定一式を実行
  python experiment.py T P [seed] # Tタイプ、各P個の設定を1つ実行
"""
import sys
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("fr", "find_rules.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

CHARS = "grybcmwk"   # 緑 赤 黄 青 シアン マゼンタ 白
PAL = {
    0: (60, 255, 80), 1: (255, 60, 60), 2: (255, 220, 60), 3: (60, 130, 255),
    4: (60, 240, 240), 5: (255, 90, 255), 6: (220, 220, 220),
}


def setup(types, per_type):
    fr.NUM_TYPES = types
    fr.PER_TYPE = per_type
    fr.N = types * per_type
    fr.TYPE_CHAR = CHARS[:types]


def run(types, per_type, seed=0, steps=1500, matrix=None, label=""):
    setup(types, per_type)
    if matrix is None:
        g = fr.biased_random(np.random.default_rng(123 + types * 100))
    else:
        g = np.array(matrix, np.float32).reshape(types, types)
    x, y, vx, vy, t = fr.init_particles(np.random.default_rng(seed))
    gij = fr.make_gij(t, g)
    for _ in range(steps):
        x, y, vx, vy = fr.step(x, y, vx, vy, gij)
    print(f"===== {types} types x {per_type} = {fr.N} particles {label} =====")
    print(fr.ascii_render(x, y, t, 90, 40))
    print()

    # PNGプレビューを保存
    from PIL import Image
    img = Image.new("RGB", (fr.WORLD_W, fr.WORLD_H), (0, 0, 0))
    px = img.load()
    for i in range(fr.N):
        xi = int(np.clip(x[i], 0, fr.WORLD_W - 1))
        yi = int(np.clip(y[i], 0, fr.WORLD_H - 1))
        px[xi, yi] = PAL[int(t[i])]
    img.save(f"exp_t{types}_p{per_type}.png")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run(int(sys.argv[1]), int(sys.argv[2]),
            seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        classic = [-0.32, -0.17, 0.34, -0.34, -0.10, 0.00, -0.20, 0.00, 0.15]
        run(3, 300, matrix=classic, label="(classic, denser)")  # 900粒子
        run(4, 150, label="(4 colors)")                          # 600粒子
        run(5, 120, label="(5 colors)")                          # 600粒子
        run(4, 200, label="(4 colors, denser)")                  # 800粒子
