# CYD Particle Life

A high-performance Particle Life artificial life simulation optimized for the ESP32-2432S028R (Cheap Yellow Display / CYD) with a 2.8" 240x320 TFT display.

![CYD Particle Life Demo](demo.gif)

## Web Installer (One-Click Browser Flash)

Flash directly to your CYD board from your Web Browser without installing Arduino IDE or configuring build tools:
- **Web Installer Page:** [https://ootake0914-dotcom.github.io/cyd-particle-life/](https://ootake0914-dotcom.github.io/cyd-particle-life/)

Requires Google Chrome, Microsoft Edge, or any browser with Web Serial API support.

---

## Overview

Particle Life is an artificial life simulation where simple attraction and repulsion rules between different types of particles give rise to complex, emergent behaviors resembling biological organisms, cell division, flocking, and fluid dynamics.

This project ports, parallelizes, and re-engineers the Particle Life engine from scratch specifically for the dual-core ESP32 microcontroller, achieving **900 simultaneous particles running in real-time** directly on a low-cost embedded device.

---

## Features

- **900 Particles Real-Time Simulation:** Simulates attraction and repulsion forces across 6 distinct particle species (Green, Red, Yellow, Cyan, Magenta, Blue).
- **15 Curated Ecosystem Stages:** Tap the simulation viewport to cycle through 15 mathematically discovered stages:
  1. `cells`: Multicellular organisms with distinct nuclei and protective membranes.
  2. `chains`: Cyclic chasing chains and undulating worm-like strings.
  3. `orbits`: Asymmetric attraction clusters orbiting in chaotic harmony.
  4. `crawlers`: Crawling colonies traveling across the screen.
  5. `microbes`: Dense, active microbial ecosystems.
  6. `mitosis`: Small circular cells that grow and divide.
  7. `membranes`: Outer shells protecting dense core particles.
  8. `worms`: Fast undulating predator-prey strings.
  9. `vortex`: High-spin rotational vortex capturing nearby particles.
  10. `planets`: Planetary systems with revolving satellite particles.
  11. `gliders`: Autonomous gliding structures navigating across space.
  12. `swarm`: Dynamic flocking swarms with cohesive group velocity.
  13. `colonies`: Stable isolated multi-color colonial structures.
  14. `crystals`: Symmetrical and stationary crystal lattices.
  15. `amoeba`: Large slowly deforming amoeboid super-colonies.
- **Built-in Desk Clock (HUD):** Functions as an ambient desk clock with customizable hours, minutes, and zero-second reset buttons via on-screen touch controls.
- **Phosphor Trail & Glow Rendering:** Employs a 4-bit per-pixel decay buffer with single-cycle 32-bit SIMD-style LUT lookup to render soft particle glows and smooth fading trails.

---

## Performance and Engine Optimizations

1. **Dual-Core Work-Stealing Force Parallelism:**
   - **Core 0 (Master):** Builds spatial grid / cell lists, executes Core 0's half of the force calculation jobs, performs velocity/position integration, and handles double-buffered state snapshots.
   - **Core 1 (Worker & Render):** Simultaneously computes Core 1's half of force calculation jobs, builds RGB565 framebuffers, and triggers non-blocking asynchronous DMA transfers.
   - Dynamic job scheduling and core-bias auto-balancing maintain a near-zero synchronization wait (`wait ≈ 1 µs`) between both cores.

2. **Spatial Grid (Cell-List) & Scatter Pair Processing:**
   - Particles are partitioned into spatial grid cells. Neighbor searches are restricted to adjacent 3x3 cells.
   - Force evaluation uses a symmetric scatter approach (Newton's third law: $F_{ji} = -F_{ij}$), halving the number of pair evaluations with zero lock contention via core-private accumulation arrays.
   - Fast Manhattan and bounding box pruning reject out-of-range pairs before computing squared distances.

3. **Full Fixed-Point Arithmetic & Precomputed 32-bit LUTs:**
   - Inner-loop physics operate entirely in fixed-point math: Positions in Q4 (1/16 px), interaction rule matrix and distance rankings in Q8, and force/velocity accumulators in Q12.
   - Precomputes a 3D LUT `forceScale[type_i][distance_rank][type_j]` holding packed 16-bit bidirectional force scaling factors, eliminating floating-point divisions and square roots in the inner loop.

4. **Zero-Copy Asynchronous 80MHz SPI DMA Pipeline:**
   - Framebuffers are rendered directly into pre-swapped RGB565 memory and pushed via a custom `push565DMA()` method over 80MHz SPI DMA.
   - CPU simulation on both cores runs concurrently with the hardware DMA transfer (~11 ms), completely hiding SPI communication latency.
   - Double-buffered particle arrays and snapshot unions eliminate large `memcpy` operations.

---

## Hardware Requirements

- **Target Board:** ESP32-2432S028R (CYD / Cheap Yellow Display)
- **Supported Display Panels:** 2.8" TFT LCD (240x320 resolution, ST7789 / ILI9341)
- **Touch Panel:** XPT2046 Resistive Touch Controller (via dedicated HSPI)

---

## How to Build

See [BUILD.md](BUILD.md) for step-by-step instructions on setting up the Arduino IDE, installing required libraries (LovyanGFX, XPT2046_Touchscreen), and uploading firmware to the board.

---

## Rule Exploration Tools

The `tools/` directory contains Python scripts for offline rule parameter searching, behavioral metrics evaluation, and population ratio optimization:

- `search_classic.py`: Core genetic search engine with mathematical archetypes and multi-metric scoring.
- `longcheck.py`: Automated multi-seed long-term stability and behavioral verification tool.
- `optimize_ratios.py`: Particle count and species ratio optimization.
- `experiment.py`: Experimental physics and clustering tests.

---

## Credits and Attribution

This project is built upon open-source software and algorithms:

- **Original Algorithm & Concept:** [Particle Life](https://github.com/hunar4321/particle-life) by Hunar Ahmad ([@hunar4321](https://github.com/hunar4321)), licensed under the **MIT License**.
- **Display Graphics Library:** [LovyanGFX](https://github.com/lovyan03/LovyanGFX) by lovyan03 ([@lovyan03](https://github.com/lovyan03)), licensed under the **FreeBSD License (2-Clause BSD)**.
- **Touch Controller Library:** [XPT2046_Touchscreen](https://github.com/PaulStoffregen/XPT2046_Touchscreen) by Paul Stoffregen ([@PaulStoffregen](https://github.com/PaulStoffregen)).

For full third-party license texts, please refer to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
