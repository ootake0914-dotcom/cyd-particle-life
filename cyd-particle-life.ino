//  CYD Particle Life - 高速シミュレーションエンジン (ESP32-2432S028R / CYD)

#include <math.h>
#include <string.h>
#include <esp_random.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <esp_attr.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <XPT2046_Touchscreen.h>
#include "config.h"

// --- タッチピン（CYD, HSPI） ---
#define TOUCH_CS   33
#define TOUCH_IRQ  36
#define TOUCH_MOSI 32
#define TOUCH_MISO 39
#define TOUCH_CLK  25

// --- タッチ補正パラメータ ---
#define TOUCH_MIN_X 200
#define TOUCH_MAX_X 3700
#define TOUCH_MIN_Y 300
#define TOUCH_MAX_Y 3800

LGFX_CYD lcd;
SPIClass touchSpi(HSPI);
XPT2046_Touchscreen ts(TOUCH_CS, TOUCH_IRQ);

// --- 描画・ワールド領域 ---
#define WORLD_W   320
#define WORLD_H   170
#define ORIGIN_Y  ((240 - WORLD_H) / 2)  // 35
#define HUD_Y     (ORIGIN_Y + WORLD_H)   // 205
#define HUD_H     (240 - HUD_Y)          // 35

// --- シミュレーション定数 ---
#define MAX_TYPES        6
#define NUM_PARTICLES    900
#define R                60
#define FPB              4                       // 固定小数点（位置: Q4）
#define RQ4              (R << FPB)
#define R2Q8             ((R * R) << (2 * FPB))
#define TRAIL_LEVELS     16
#define TRAIL_MASK       (TRAIL_LEVELS - 1)
#define LUT_SIZE         (R * R + 2)
#define INV_LEVELS       72
#define TARGET_FPS       60   // vTaskDelayUntilでの一定周期ペーシング用

// 粒子データ構造体（キャッシュ効率化のため6バイトに圧縮）
struct Particle {
  int16_t x, y;    // Q4位置
  uint8_t type;    // 粒子タイプ
  uint8_t _pad;
};

// シミュレーション（Core 0）から描画（Core 1）へ渡す4バイトスナップショット
union Snap {
  uint32_t u;
  struct { int16_t px; uint8_t py; uint8_t type; };
};

// ダブルバッファ粒子・IDポインタ
Particle* partsCur = nullptr;
Particle* partsNext = nullptr;
uint16_t* idCur = nullptr;
uint16_t* idNext = nullptr;

// 物理パラメータ・力・速度配列
int32_t  velXq12[NUM_PARTICLES];
int32_t  velYq12[NUM_PARTICLES];
int32_t  forceXq12[2][NUM_PARTICLES];
int32_t  forceYq12[2][NUM_PARTICLES];
int16_t  ruleGq8[MAX_TYPES][MAX_TYPES];
int16_t  ruleGq8T[MAX_TYPES][MAX_TYPES];
int32_t  radius2Q8[MAX_TYPES];
int32_t  dampQ12;
uint8_t  typeColor[MAX_TYPES];

// LUTテーブル
uint8_t   invRank[LUT_SIZE];
uint16_t  invValue[INV_LEVELS];
int       invLevelsCount = 0;
uint32_t (*forceScale)[INV_LEVELS][MAX_TYPES] = nullptr;
uint32_t  trailStep[256];
int32_t   currentRadiusQ4;
int32_t   currentRadius2Q8;
uint8_t   gammaLut[256];

// 手動時計
int clockHour = 0, clockMin = 0, clockSec = 0;
unsigned long lastClockTick = 0;

// フレームバッファ・残光バッファ
uint16_t* frame565 = nullptr;
uint16_t  pal565[256];
uint8_t*  trail8 = nullptr;
uint16_t  trail565[256];
uint16_t  arenaBackground565;
uint16_t  arenaBorder565;

// スナップショット＆同期セマフォ
Snap snap[2][NUM_PARTICLES];
SemaphoreHandle_t semEmpty, semFull;
SemaphoreHandle_t semForceGo, semForceDone;

// 2Dセルリスト空間分割
#define CELL_SHIFT 6
#define CELL_SIZE (1 << CELL_SHIFT)
#define GRID_W ((WORLD_W + CELL_SIZE - 1) >> CELL_SHIFT)
#define GRID_H ((WORLD_H + CELL_SIZE - 1) >> CELL_SHIFT)
#define NUM_CELLS (GRID_W * GRID_H)

uint16_t cellCount[NUM_CELLS];
uint16_t cellStart[NUM_CELLS + 1];

// 並列力計算ジョブ定義
#define MAX_PAIRS_PER_JOB 1024
#define MAX_JOBS_PER_CORE 512
struct ForceJob {
  uint16_t aBegin;
  uint16_t aEnd;
  uint16_t bBegin;
  uint16_t bEnd;
  bool     sameCell;
};

ForceJob* jobs[2] = { nullptr, nullptr };
uint16_t numJobs[2];
uint32_t workUnits[2];
volatile bool forceJobOverflow = false;
volatile bool wantNextPreset = false;

// Core 0 / Core 1 の負荷平準化用 EMA処理時間計測変数
volatile uint32_t sortUsAvg = 0;
volatile uint32_t integrateUsAvg = 0;
volatile uint32_t buildUsAvg = 0;
volatile uint32_t pushUsAvg = 0;

// 逆平方根LUTの初期化
void initLUT() {
  invLevelsCount = 0;
  uint16_t lastVal = 0xFFFF;
  for (int k = 0; k < LUT_SIZE; k++) {
    uint16_t val = (k == 0) ? 256 : (uint16_t)(1.0f / sqrtf((float)k) * 256.0f + 0.5f);
    if (val != lastVal && invLevelsCount < INV_LEVELS) {
      invValue[invLevelsCount] = val;
      lastVal = val;
      invLevelsCount++;
    }
    invRank[k] = (uint8_t)(invLevelsCount - 1);
  }
}

// ガンマ補正LUTの初期化
void initGammaLUT() {
  for (int i = 0; i < 256; i++) {
    float t = i / 255.0f;
    float e = t * t * (3.0f - 2.0f * t);
    gammaLut[i] = (uint8_t)(e * 255.0f + 0.5f);
  }
}

// 手動時計更新
void tickClock() {
  unsigned long now = millis();
  while (now - lastClockTick >= 1000) {
    lastClockTick += 1000;
    if (++clockSec >= 60) {
      clockSec = 0;
      if (++clockMin >= 60) {
        clockMin = 0;
        clockHour = (clockHour + 1) % 24;
      }
    }
  }
}

#define NUM_PRESETS 15
#define DEFAULT_PRESET 0

// 6色ルール行列プリセット（15ステージ）
const float presets[NUM_PRESETS][MAX_TYPES][MAX_TYPES] = {
  { // 0: cells
    { -0.253f, -0.751f, -0.400f, -0.148f, -0.578f, -0.282f },
    {  0.381f, -0.213f, -0.344f,  0.152f,  0.069f,  0.608f },
    {  0.601f, -0.041f, -0.329f,  0.030f,  0.123f,  0.192f },
    {  0.180f, -0.630f, -0.695f, -0.117f, -0.791f,  0.122f },
    {  0.276f, -0.272f, -0.073f,  0.491f, -0.267f,  0.716f },
    { -0.239f, -0.240f, -1.000f,  0.166f, -0.746f, -0.056f },
  },
  { // 1: chains
    { -0.103f, -0.307f, -0.257f, -0.221f, -0.137f,  0.208f },
    {  0.609f, -0.170f, -0.509f, -0.037f,  0.300f, -0.155f },
    {  0.279f,  0.578f, -0.093f, -0.555f,  0.126f,  0.247f },
    { -0.068f, -0.348f,  0.402f, -0.212f, -0.938f,  0.201f },
    { -0.178f, -0.250f,  0.210f,  0.693f, -0.163f, -0.676f },
    { -0.731f,  0.130f, -0.201f, -0.163f,  0.789f, -0.146f },
  },
  { // 2: orbits
    { -0.109f, -0.064f,  0.144f, -0.480f, -0.496f, -0.223f },
    { -0.433f, -0.048f,  0.036f, -0.124f,  0.169f,  0.692f },
    { -0.212f,  0.315f, -0.202f, -0.559f, -0.199f,  0.398f },
    {  0.675f,  0.020f, -0.334f, -0.191f, -0.031f,  0.279f },
    { -0.095f, -0.189f,  0.273f, -0.224f, -0.275f,  0.198f },
    { -0.041f, -0.644f,  0.181f,  0.470f, -0.401f, -0.258f },
  },
  { // 3: crawlers
    { -0.121f, -0.093f,  0.969f, -0.411f, -0.708f, -0.710f },
    {  0.308f, -0.553f, -0.703f, -0.211f, -0.568f, -0.183f },
    { -0.263f,  0.200f,  0.243f, -1.000f, -0.746f, -0.509f },
    {  0.442f,  0.139f,  0.139f, -0.499f, -0.231f, -0.915f },
    {  0.104f,  0.071f,  0.165f,  0.395f, -0.485f, -0.526f },
    { -0.149f,  0.845f,  0.064f,  0.615f,  0.562f, -0.419f },
  },
  { // 4: microbes
    { -0.116f, -0.241f, -0.614f, -0.270f, -0.442f, -0.291f },
    {  0.008f, -0.490f,  0.451f, -0.413f, -0.496f, -0.297f },
    {  0.823f,  0.209f, -0.427f, -1.000f, -0.027f, -0.388f },
    {  0.538f,  0.310f, -0.151f, -0.132f, -0.372f, -0.248f },
    {  0.174f,  0.571f,  0.847f,  0.110f, -0.369f, -0.217f },
    { -0.281f,  0.862f,  0.328f,  0.571f,  0.168f, -0.344f },
  },
  { // 5: mitosis
    { -0.089f, -0.327f,  0.267f, -0.635f,  0.383f, -0.713f },
    {  0.743f, -0.263f, -0.943f,  0.319f, -0.970f, -0.340f },
    { -0.274f,  0.683f, -0.440f,  0.405f, -0.282f, -0.770f },
    {  0.835f,  0.838f, -0.379f, -0.327f, -0.807f, -0.842f },
    {  0.700f,  0.380f,  0.943f, -0.007f, -0.601f, -0.210f },
    {  0.634f,  0.009f,  0.913f,  0.529f,  0.603f, -0.161f },
  },
  { // 6: membranes
    { -0.485f,  0.113f,  0.391f,  0.461f, -0.278f,  0.645f },
    { -0.732f, -0.318f,  0.062f,  0.049f, -0.717f, -0.293f },
    { -0.683f,  0.003f, -0.148f, -0.311f, -0.533f, -0.273f },
    { -0.343f, -0.134f, -0.326f, -0.218f, -0.499f,  0.081f },
    { -0.276f,  0.222f,  0.259f,  0.599f, -0.378f,  0.498f },
    { -0.441f,  0.087f, -0.096f,  0.154f, -0.537f, -0.391f },
  },
  { // 7: worms
    { -0.102f, -0.573f,  0.012f,  0.102f, -0.264f,  0.775f },
    {  0.304f, -0.282f, -0.701f, -0.197f,  0.270f,  0.190f },
    { -0.056f,  0.513f, -0.317f, -0.406f,  0.134f, -0.333f },
    { -0.130f,  0.143f,  0.701f, -0.161f, -0.560f,  0.259f },
    { -0.173f, -0.291f,  0.328f,  0.891f, -0.292f, -0.480f },
    { -0.378f, -0.259f,  0.161f, -0.011f,  0.377f, -0.250f },
  },
  { // 8: vortex
    { -0.000f, -0.692f, -0.049f,  0.273f,  0.223f, -0.212f },
    {  0.145f,  0.042f, -0.032f, -0.412f, -0.117f, -0.631f },
    {  0.491f, -0.100f, -0.430f,  0.423f,  0.117f, -0.347f },
    { -0.428f, -0.294f, -0.424f, -0.063f, -0.149f, -0.164f },
    {  0.455f, -0.008f, -0.632f,  0.390f,  0.084f,  0.203f },
    { -0.041f, -0.065f,  0.430f, -0.287f,  0.172f, -0.076f },
  },
  { // 9: planets
    { -0.017f, -0.199f, -0.131f,  0.539f, -0.436f, -0.462f },
    {  0.132f, -0.301f, -0.120f, -0.569f, -0.100f, -0.881f },
    {  0.291f, -0.214f, -0.644f, -0.192f, -0.024f, -0.357f },
    {  0.284f,  0.140f, -0.446f, -1.000f,  0.020f, -0.053f },
    { -0.054f,  0.854f, -0.269f,  0.360f, -0.386f, -0.328f },
    {  0.549f,  0.471f,  0.381f,  0.391f,  0.394f, -0.095f },
  },
  { // 10: gliders
    { -0.129f, -0.326f, -0.532f,  0.377f, -0.821f, -0.551f },
    {  0.586f, -0.052f, -0.553f, -0.749f, -0.846f, -0.470f },
    {  0.338f,  0.113f, -0.454f, -0.649f, -0.807f, -0.526f },
    { -0.129f,  0.890f,  0.918f, -0.342f, -0.644f, -0.433f },
    {  0.817f,  0.330f,  0.248f,  0.133f, -0.273f,  0.427f },
    {  0.457f,  0.896f,  0.445f,  0.756f, -0.482f, -0.017f },
  },
  { // 11: swarm
    { -0.456f, -0.362f, -0.238f, -0.243f, -0.763f, -0.910f },
    {  0.110f, -0.407f,  0.870f,  0.423f, -0.695f, -0.405f },
    {  0.745f, -0.757f, -0.210f, -0.160f, -0.804f, -0.115f },
    { -0.150f,  0.330f,  0.535f, -0.153f, -0.939f, -0.584f },
    {  0.484f,  0.031f,  0.253f,  0.407f, -0.290f, -0.936f },
    {  0.560f,  0.762f,  0.311f,  0.368f,  0.153f, -0.381f },
  },
  { // 12: colonies
    { -0.611f, -0.618f, -0.398f, -0.788f,  0.652f, -0.809f },
    {  0.351f, -0.514f, -0.305f, -1.000f, -0.289f, -0.326f },
    {  0.247f,  0.501f, -0.037f, -1.000f, -0.444f, -0.663f },
    {  0.508f,  0.642f,  0.269f, -0.267f, -0.431f, -0.483f },
    {  0.037f,  0.937f,  0.254f,  0.476f, -0.500f, -0.374f },
    {  0.716f,  0.655f,  0.763f,  0.135f,  0.424f, -0.439f },
  },
  { // 13: crystals
    { -0.050f, -0.063f,  0.007f, -0.164f,  0.112f,  0.378f },
    { -0.063f, -0.131f,  0.125f, -0.003f,  0.327f,  0.063f },
    {  0.007f,  0.125f, -0.284f, -0.020f, -0.208f, -0.123f },
    { -0.164f, -0.003f, -0.020f, -0.315f,  0.036f,  0.150f },
    {  0.112f,  0.327f, -0.208f,  0.036f, -0.227f,  0.015f },
    {  0.378f,  0.063f, -0.123f,  0.150f,  0.015f, -0.061f },
  },
  { // 14: amoeba
    { -0.106f,  0.271f, -0.417f,  0.469f,  0.308f, -0.090f },
    { -0.435f, -0.098f, -0.436f,  0.058f, -0.237f, -0.606f },
    { -0.216f,  0.533f, -0.113f,  0.124f,  0.488f,  0.209f },
    { -0.416f,  0.167f, -0.286f, -0.486f,  0.168f, -0.283f },
    { -0.268f, -0.448f, -0.544f,  0.039f, -0.493f, -0.272f },
    { -0.405f,  0.112f, -0.072f,  0.056f,  0.497f, -0.083f },
  },
};

const float presetViscosity[NUM_PRESETS] = {
  0.64f, 0.49f, 0.53f, 0.65f, 0.75f, 0.70f, 0.67f, 0.37f,
  0.57f, 0.50f, 0.32f, 0.75f, 0.75f, 0.76f, 0.63f
};

const uint16_t presetCounts[NUM_PRESETS][MAX_TYPES] = {
  { 260, 260, 130, 130,  60,  60 }, // 0: cells
  { 210, 210, 160, 160,  80,  80 }, // 1: chains
  { 260, 260, 130, 130,  60,  60 }, // 2: orbits
  { 210, 210, 160, 160,  80,  80 }, // 3: crawlers
  { 150, 150, 150, 150, 150, 150 }, // 4: microbes
  { 260, 260, 130, 130,  60,  60 }, // 5: mitosis
  { 260, 260, 130, 130,  60,  60 }, // 6: membranes
  { 260, 260, 130, 130,  60,  60 }, // 7: worms
  { 150, 150, 150, 150, 150, 150 }, // 8: vortex
  { 260, 260, 130, 130,  60,  60 }, // 9: planets
  { 130, 130, 130, 130, 190, 190 }, // 10: gliders
  { 210, 210, 160, 160,  80,  80 }, // 11: swarm
  { 130, 130, 130, 130, 190, 190 }, // 12: colonies
  { 260, 260, 130, 130,  60,  60 }, // 13: crystals
  { 260, 260, 130, 130,  60,  60 }, // 14: amoeba
};

const uint8_t presetRadius[NUM_PRESETS][MAX_TYPES] = {
  { 60, 60, 60, 60, 60, 60 }, // 0: cells
  { 36, 36, 36, 36, 36, 36 }, // 1: chains
  { 50, 50, 50, 50, 50, 50 }, // 2: orbits
  { 50, 50, 50, 50, 50, 50 }, // 3: crawlers
  { 60, 60, 60, 60, 60, 60 }, // 4: microbes
  { 50, 50, 50, 50, 50, 50 }, // 5: mitosis
  { 55, 55, 55, 55, 55, 55 }, // 6: membranes
  { 42, 42, 42, 42, 42, 42 }, // 7: worms
  { 55, 55, 55, 55, 55, 55 }, // 8: vortex
  { 50, 50, 50, 50, 50, 50 }, // 9: planets
  { 50, 50, 50, 50, 50, 50 }, // 10: gliders
  { 55, 55, 55, 55, 55, 55 }, // 11: swarm
  { 60, 60, 60, 60, 60, 60 }, // 12: colonies
  { 50, 50, 50, 50, 50, 50 }, // 13: crystals
  { 60, 60, 60, 60, 60, 60 }, // 14: amoeba
};

const uint8_t presetTypeCount[NUM_PRESETS] = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
};

volatile int currentPreset = DEFAULT_PRESET;
float viscosity = 0.64f;

const char* presetNames[NUM_PRESETS] = {
  "cells", "chains", "orbits", "crawlers",
  "microbes", "mitosis", "membranes", "worms",
  "vortex", "planets", "gliders", "swarm",
  "colonies", "crystals", "amoeba"
};

// プリセット適用
void setPreset(int p) {
  for (int i = 0; i < MAX_TYPES; i++) {
    for (int j = 0; j < MAX_TYPES; j++) {
      int16_t g = (int16_t)lrintf(presets[p][i][j] * 256.0f);
      ruleGq8[i][j]  = g;
      ruleGq8T[j][i] = g;
    }
  }
  if (forceScale) {
    for (int ti = 0; ti < MAX_TYPES; ti++) {
      for (int r = 0; r < invLevelsCount; r++) {
        for (int tj = 0; tj < MAX_TYPES; tj++) {
          int16_t fij = (int16_t)(((int32_t)ruleGq8[ti][tj] * invValue[r]) >> 8);
          int16_t fji = (int16_t)(((int32_t)ruleGq8[tj][ti] * invValue[r]) >> 8);
          forceScale[ti][r][tj] = (uint16_t)fij | ((uint32_t)(uint16_t)fji << 16);
        }
      }
    }
  }
  viscosity = presetViscosity[p];
  dampQ12 = (int32_t)lrintf((1.0f - viscosity) * 4096.0f);
  const int radius = presetRadius[p][0];
  currentRadiusQ4  = radius << FPB;
  currentRadius2Q8 = ((int32_t)radius * radius) << (2 * FPB);
  for (int i = 0; i < MAX_TYPES; i++) {
    radius2Q8[i] = ((int32_t)presetRadius[p][i] * presetRadius[p][i]) << (2 * FPB);
  }
}

// 疑似乱数生成器（Xorshift32）
uint32_t testRandom = 0x12345678;
static inline uint32_t fixedRandom() {
  testRandom ^= testRandom << 13;
  testRandom ^= testRandom >> 17;
  testRandom ^= testRandom << 5;
  return testRandom;
}

// 粒子初期化
void initParticles(int preset) {
  const int typeCount = presetTypeCount[preset];
  int type = 0;
  int nextType = presetCounts[preset][0];
  testRandom = 0x12345678;
  for (int k = 0; k < NUM_PARTICLES; k++) {
    idCur[k] = (uint16_t)k;
    while (type < typeCount - 1 && k >= nextType) {
      type++;
      nextType += presetCounts[preset][type];
    }
    partsCur[k].type = type;
    partsCur[k].x = (int16_t)((fixedRandom() % WORLD_W) << FPB);
    partsCur[k].y = (int16_t)((fixedRandom() % WORLD_H) << FPB);
    velXq12[k] = 0;
    velYq12[k] = 0;
  }
}

// 2Dセルリスト構築 ＆ 各コアの処理負荷に応じた動的ジョブ振り分け
void IRAM_ATTR __attribute__((optimize("O3"))) buildCellList() {
  memset(cellCount, 0, sizeof(cellCount));
  const Particle* __restrict__ p = partsCur;

  for (int k = 0; k < NUM_PARTICLES; k++) {
    int cx = (p[k].x >> FPB) >> CELL_SHIFT;
    int cy = (p[k].y >> FPB) >> CELL_SHIFT;
    if ((unsigned)cx >= GRID_W) cx = (cx < 0) ? 0 : GRID_W - 1;
    if ((unsigned)cy >= GRID_H) cy = (cy < 0) ? 0 : GRID_H - 1;
    cellCount[cy * GRID_W + cx]++;
  }

  cellStart[0] = 0;
  for (int c = 0; c < NUM_CELLS; c++) {
    cellStart[c + 1] = cellStart[c] + cellCount[c];
  }

  uint16_t cellCursor[NUM_CELLS];
  memcpy(cellCursor, cellStart, sizeof(cellCursor));

  const uint16_t* __restrict__ pid = idCur;
  for (int k = 0; k < NUM_PARTICLES; k++) {
    int cx = (p[k].x >> FPB) >> CELL_SHIFT;
    int cy = (p[k].y >> FPB) >> CELL_SHIFT;
    if ((unsigned)cx >= GRID_W) cx = (cx < 0) ? 0 : GRID_W - 1;
    if ((unsigned)cy >= GRID_H) cy = (cy < 0) ? 0 : GRID_H - 1;
    int cellIdx = cy * GRID_W + cx;
    const int dst = cellCursor[cellIdx]++;
    partsNext[dst] = p[k];
    idNext[dst] = pid[k];
  }

  // ポインタスワップ
  Particle* tmp = partsCur;
  partsCur = partsNext;
  partsNext = tmp;
  uint16_t* tmpId = idCur;
  idCur = idNext;
  idNext = tmpId;

  // ジョブ初期化
  // Core 1側の描画/DMA負荷とCore 0側のソート/積分負荷の差分をオフセットとして
  // workUnits[1]に加算し、以降のジョブ振り分け（workUnits最小コアへ割当）で
  // 両コアの完了タイミングが揃うようバランシングする。
  numJobs[0] = 0;
  numJobs[1] = 0;
  {
    const int32_t netBiasUs = (int32_t)(buildUsAvg + pushUsAvg) - (int32_t)(sortUsAvg + integrateUsAvg);
    const uint32_t NS_PER_CAND = 240; // 1候補あたりの実測計算コスト（ns）
    workUnits[1] = (netBiasUs > 0) ? (uint32_t)((int64_t)netBiasUs * 1000 / NS_PER_CAND) : 0;
  }
  workUnits[0] = 0;
  forceJobOverflow = false;

  const int32_t radiusQ4 = currentRadiusQ4;
  const int radiusCells = (int)((radiusQ4 >> FPB) + CELL_SIZE - 1) >> CELL_SHIFT;
  const int radiusPx = (int)(radiusQ4 >> FPB);
  const int radiusPx2 = radiusPx * radiusPx;

  for (int cellA = 0; cellA < NUM_CELLS; cellA++) {
    int countA = cellCount[cellA];
    if (countA == 0) continue;
    int aBegin = cellStart[cellA];
    const int cellEnd = cellStart[cellA + 1];
    int cyA = cellA / GRID_W, cxA = cellA % GRID_W;

    // 1. 同一セル内ペア
    if (countA > 1) {
      int curA = aBegin;
      while (curA < cellEnd - 1) {
        uint32_t work = 0;
        int nextA = curA;
        while (nextA < cellEnd - 1) {
          uint32_t rowWork = cellEnd - 1 - nextA;
          if (nextA > curA && (work + rowWork) > MAX_PAIRS_PER_JOB) break;
          work += rowWork;
          nextA++;
        }

        int targetCore = (workUnits[0] <= workUnits[1]) ? 0 : 1;
        if (numJobs[targetCore] >= MAX_JOBS_PER_CORE) {
          forceJobOverflow = true;
          return;
        }
        ForceJob& job = jobs[targetCore][numJobs[targetCore]++];
        job.aBegin = curA;
        job.aEnd   = nextA;
        job.bBegin = curA;
        job.bEnd   = cellEnd;
        job.sameCell = true;
        workUnits[targetCore] += work;

        curA = nextA;
      }
    }

    // 2. 近傍セル間ペア
    for (int dyCell = 0; dyCell <= radiusCells; dyCell++) {
      int cyB = cyA + dyCell;
      if (cyB >= GRID_H) break;
      int dxMin = (dyCell == 0) ? 1 : -radiusCells;
      int dxMax = radiusCells;

      for (int dxCell = dxMin; dxCell <= dxMax; dxCell++) {
        int cxB = cxA + dxCell;
        if (cxB < 0 || cxB >= GRID_W) continue;
        int cellB = cyB * GRID_W + cxB;
        int countB = cellCount[cellB];
        if (countB == 0) continue;

        int dX = abs(cxB - cxA) - 1; if (dX < 0) dX = 0;
        int dY = abs(cyB - cyA) - 1; if (dY < 0) dY = 0;
        if ((dX * dX + dY * dY) * (CELL_SIZE * CELL_SIZE) >= radiusPx2) continue;

        int bBegin = cellStart[cellB];
        int bEnd   = cellStart[cellB + 1];

        const int maxPairsPerJob = MAX_PAIRS_PER_JOB;
        int aChunk = maxPairsPerJob / countB;
        if (aChunk < 1) aChunk = 1;

        for (int a = aBegin; a < cellEnd; a += aChunk) {
          int aSubEnd = a + aChunk;
          if (aSubEnd > cellEnd) aSubEnd = cellEnd;
          uint32_t w = (uint32_t)(aSubEnd - a) * countB;

          int targetCore = (workUnits[0] <= workUnits[1]) ? 0 : 1;
          if (numJobs[targetCore] >= MAX_JOBS_PER_CORE) {
            forceJobOverflow = true;
            return;
          }
          ForceJob& job = jobs[targetCore][numJobs[targetCore]++];
          job.aBegin = a;
          job.aEnd   = aSubEnd;
          job.bBegin = bBegin;
          job.bEnd   = bEnd;
          job.sameCell = false;
          workUnits[targetCore] += w;
        }
      }
    }
  }
}

// 粒子間力計算（各コア担当分を実行）
void IRAM_ATTR __attribute__((optimize("O3"))) computeForces(int coreId,
                                                              int32_t* __restrict__ fxArr,
                                                              int32_t* __restrict__ fyArr) {
  const Particle* __restrict__ p = partsCur;
  const int32_t radiusQ4 = currentRadiusQ4;
  const int32_t radius2Q8 = currentRadius2Q8;
  const uint32_t dyLimit = (uint32_t)(2 * radiusQ4 - 1);
  const int jobCount = numJobs[coreId];
  const ForceJob* __restrict__ jobList = jobs[coreId];

  for (int jIdx = 0; jIdx < jobCount; jIdx++) {
    const ForceJob job = jobList[jIdx];
    const int aBegin = job.aBegin;
    const int aEnd   = job.aEnd;
    const int bBegin = job.bBegin;
    const int bEnd   = job.bEnd;

    if (job.sameCell) {
      const int cellEnd = bEnd;
      for (int i = aBegin; i < aEnd; i++) {
        const Particle pi = p[i];
        const uint32_t (* __restrict__ scaleRow)[MAX_TYPES] = forceScale[pi.type];
        int32_t fxI = 0, fyI = 0;

        for (int j = i + 1; j < cellEnd; j++) {
          const Particle pj = p[j];
          const int32_t dy = pi.y - pj.y;
          if ((uint32_t)(dy + radiusQ4 - 1) >= dyLimit) continue;
          const int32_t dx = pi.x - pj.x;
          if ((uint32_t)(dx + radiusQ4 - 1) >= dyLimit) continue;

          const int32_t d2 = dx * dx + dy * dy;
          if (d2 < radius2Q8) {
            const uint8_t rank = invRank[d2 >> 8];
            const uint32_t packed = scaleRow[rank][pj.type];
            const int32_t fij = (int16_t)packed;
            const int32_t fji = (int16_t)(packed >> 16);

            fxI += fij * dx;
            fyI += fij * dy;
            fxArr[j] -= fji * dx;
            fyArr[j] -= fji * dy;
          }
        }
        fxArr[i] += fxI;
        fyArr[i] += fyI;
      }
    } else {
      for (int i = aBegin; i < aEnd; i++) {
        const Particle pi = p[i];
        const uint32_t (* __restrict__ scaleRow)[MAX_TYPES] = forceScale[pi.type];
        int32_t fxI = 0, fyI = 0;

        for (int j = bBegin; j < bEnd; j++) {
          const Particle pj = p[j];
          const int32_t dy = pi.y - pj.y;
          if ((uint32_t)(dy + radiusQ4 - 1) >= dyLimit) continue;
          const int32_t dx = pi.x - pj.x;
          if ((uint32_t)(dx + radiusQ4 - 1) >= dyLimit) continue;

          const int32_t d2 = dx * dx + dy * dy;
          if (d2 < radius2Q8) {
            const uint8_t rank = invRank[d2 >> 8];
            const uint32_t packed = scaleRow[rank][pj.type];
            const int32_t fij = (int16_t)packed;
            const int32_t fji = (int16_t)(packed >> 16);

            fxI += fij * dx;
            fyI += fij * dy;
            fxArr[j] -= fji * dx;
            fyArr[j] -= fji * dy;
          }
        }
        fxArr[i] += fxI;
        fyArr[i] += fyI;
      }
    }
  }
}

// 運動積分・壁境界反発処理（全固定小数点）
void IRAM_ATTR __attribute__((optimize("O3"))) integrate() {
  const int32_t WALLQ4 = 12 << FPB;
  const int32_t WQ4 = WORLD_W << FPB, HQ4 = WORLD_H << FPB;
  const int32_t WALL_KQ8 = 26;   // バネ定数 Q8
  for (int k = 0; k < NUM_PARTICLES; k++) {
    Particle& a = partsCur[k];
    const uint16_t id = idCur[k];
    int32_t vx = velXq12[id], vy = velYq12[id];
    int32_t fx = forceXq12[0][k] + forceXq12[1][k];
    int32_t fy = forceYq12[0][k] + forceYq12[1][k];

    if (a.x < WALLQ4)         fx += (WALLQ4 - a.x) * WALL_KQ8;
    if (a.x > WQ4 - WALLQ4)   fx += (WQ4 - WALLQ4 - a.x) * WALL_KQ8;
    if (a.y < WALLQ4)         fy += (WALLQ4 - a.y) * WALL_KQ8;
    if (a.y > HQ4 - WALLQ4)   fy += (HQ4 - WALLQ4 - a.y) * WALL_KQ8;

    vx = (int32_t)(((int64_t)vx * dampQ12) >> 12) + fx;
    vy = (int32_t)(((int64_t)vy * dampQ12) >> 12) + fy;

    int32_t nx = a.x + (vx >> 8), ny = a.y + (vy >> 8);
    if (nx < 0)          { nx = -nx;          vx = -vx; }
    else if (nx >= WQ4)  { nx = 2 * WQ4 - nx; vx = -vx; }
    if (ny < 0)          { ny = -ny;          vy = -vy; }
    else if (ny >= HQ4)  { ny = 2 * HQ4 - ny; vy = -vy; }

    a.x = (int16_t)nx;
    a.y = (int16_t)ny;
    velXq12[id] = vx;
    velYq12[id] = vy;
  }
}

// 描画用スナップショットの生成（32bitパッキング）
void IRAM_ATTR __attribute__((optimize("O3"))) snapshotParticles(int w) {
  Snap* __restrict__ s = snap[w];
  const Particle* __restrict__ p = partsCur;
  for (int k = 0; k < NUM_PARTICLES; k++) {
    s[k].u = (uint32_t)(uint16_t)(p[k].x >> FPB)
           | ((uint32_t)(uint8_t)(p[k].y >> FPB) << 16)
           | ((uint32_t)p[k].type << 24);
  }
}

// --- Core 1: 力計算並列ワーカータスク ---
void forceWorker(void* arg) {
  (void)arg;
  while (1) {
    xSemaphoreTake(semForceGo, portMAX_DELAY);
    memset(forceXq12[1], 0, sizeof forceXq12[1]);
    memset(forceYq12[1], 0, sizeof forceYq12[1]);
    computeForces(1, forceXq12[1], forceYq12[1]);
    xSemaphoreGive(semForceDone);
  }
}

// --- Core 0: シミュレーションマスタータスク ---
void simMaster(void* arg) {
  (void)arg;
  int w = 0;
  while (1) {
    if (wantNextPreset) {
      wantNextPreset = false;
      currentPreset = (currentPreset + 1) % NUM_PRESETS;
      initParticles(currentPreset);
      setPreset(currentPreset);
    }

    unsigned long stage0 = micros();
    buildCellList();
    sortUsAvg = (sortUsAvg * 7 + (uint32_t)(micros() - stage0)) / 8;
    if (forceJobOverflow) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    // Core 0の力配列をクリアして直ちにCore 1ワーカーを起動
    memset(forceXq12[0], 0, sizeof forceXq12[0]);
    memset(forceYq12[0], 0, sizeof forceYq12[0]);
    xSemaphoreGive(semForceGo);

    computeForces(0, forceXq12[0], forceYq12[0]);
    xSemaphoreTake(semForceDone, portMAX_DELAY);

    stage0 = micros();
    integrate();
    integrateUsAvg = (integrateUsAvg * 7 + (uint32_t)(micros() - stage0)) / 8;

    xSemaphoreTake(semEmpty, portMAX_DELAY);
    snapshotParticles(w);
    xSemaphoreGive(semFull);
    w ^= 1;
  }
}

// 粒子発光スタンプ（残光バッファへの高速書き込み）
static inline void stampParticleGlowFast(int x, int y, uint8_t type) {
  uint8_t* __restrict__ cell = &trail8[y * WORLD_W + x];
  const uint8_t tag = (uint8_t)((type + 1) << 4);

  #define STAMP_P(ptr, lvl) do { if ((*(ptr) & TRAIL_MASK) < (lvl)) *(ptr) = (uint8_t)(tag | (lvl)); } while(0)
  STAMP_P(cell, 15);
  STAMP_P(cell - 1, 10);
  STAMP_P(cell + 1, 10);
  STAMP_P(cell - WORLD_W, 10);
  STAMP_P(cell + WORLD_W, 10);
  STAMP_P(cell - WORLD_W - 1, 7);
  STAMP_P(cell - WORLD_W + 1, 7);
  STAMP_P(cell + WORLD_W - 1, 7);
  STAMP_P(cell + WORLD_W + 1, 7);
  STAMP_P(cell - 2, 4);
  STAMP_P(cell + 2, 4);
  STAMP_P(cell - 2 * WORLD_W, 4);
  STAMP_P(cell + 2 * WORLD_W, 4);
  #undef STAMP_P
}

// フレームバッファ生成（残光減衰・RGB565展開）
void __attribute__((optimize("O3"))) buildFrame(int r) {
  uint16_t* __restrict__ buf = frame565;
  const Snap* __restrict__ s = snap[r];

  for (int k = 0; k < NUM_PARTICLES; k++) {
    const uint32_t u = s[k].u;
    unsigned px = u & 0xFFFF, py = (u >> 16) & 0xFF;
    const uint8_t type = (uint8_t)(u >> 24);
    int cx = (int)px, cy = (int)py;
    if (cx < 2) cx = 2; else if (cx > WORLD_W - 3) cx = WORLD_W - 3;
    if (cy < 2) cy = 2; else if (cy > WORLD_H - 3) cy = WORLD_H - 3;
    stampParticleGlowFast(cx, cy, type);
  }

  // 4画素アンロールで残光減衰LUT展開
  for (int y = 0; y < WORLD_H; y++) {
    uint16_t* __restrict__ row = &buf[y * WORLD_W];
    uint8_t* __restrict__ trailRow = &trail8[y * WORLD_W];
    int x = 0;
    for (; x + 4 <= WORLD_W; x += 4) {
      const uint32_t s0 = trailStep[trailRow[x]];
      const uint32_t s1 = trailStep[trailRow[x + 1]];
      const uint32_t s2 = trailStep[trailRow[x + 2]];
      const uint32_t s3 = trailStep[trailRow[x + 3]];
      trailRow[x]     = (uint8_t)s0; row[x]     = (uint16_t)(s0 >> 16);
      trailRow[x + 1] = (uint8_t)s1; row[x + 1] = (uint16_t)(s1 >> 16);
      trailRow[x + 2] = (uint8_t)s2; row[x + 2] = (uint16_t)(s2 >> 16);
      trailRow[x + 3] = (uint8_t)s3; row[x + 3] = (uint16_t)(s3 >> 16);
    }
    for (; x < WORLD_W; x++) {
      const uint32_t step = trailStep[trailRow[x]];
      trailRow[x] = (uint8_t)step;
      row[x] = (uint16_t)(step >> 16);
    }
  }

  for (int x = 0; x < WORLD_W; x++) {
    buf[x] = arenaBorder565;
    buf[(WORLD_H - 1) * WORLD_W + x] = arenaBorder565;
  }
  for (int y = 0; y < WORLD_H; y++) {
    buf[y * WORLD_W] = arenaBorder565;
    buf[y * WORLD_W + WORLD_W - 1] = arenaBorder565;
  }
}

// DMA送信制御
static inline void startPush() {
  lcd.push565DMA(0, ORIGIN_Y, WORLD_W, WORLD_H, frame565);
}

static inline void endPush() {
  lcd.waitDMA();
  lcd.endWrite();
}

// HUD描画
void drawClockButton(int x, int w, const char* label) {
  const uint16_t muted = lcd.color565(130, 138, 156);
  const uint16_t panel = lcd.color565(4, 7, 18);
  lcd.setTextColor(muted, panel);
  lcd.setTextSize(1);
  lcd.setCursor(x + (w - lcd.textWidth(label)) / 2, HUD_Y + 14);
  lcd.print(label);
}

void drawHUD() {
  const uint16_t muted = lcd.color565(130, 138, 156);
  const uint16_t panel = lcd.color565(4, 7, 18);
  static char lastPreset[32] = "";
  static char lastClock[16] = "";
  static bool controlsDrawn = false;
  char buf[48];

  snprintf(buf, sizeof buf, "P%d/%d %-9s", currentPreset + 1,
           NUM_PRESETS, presetNames[currentPreset]);
  if (strcmp(buf, lastPreset) != 0) {
    lcd.setTextColor(muted, panel);
    lcd.setTextSize(1);
    lcd.setCursor(4, 13);
    lcd.print(buf);
    strcpy(lastPreset, buf);
  }

  snprintf(buf, sizeof buf, "%02d:%02d:%02d", clockHour, clockMin, clockSec);
  if (strcmp(buf, lastClock) != 0) {
    lcd.setTextColor(muted, panel);
    lcd.setTextSize(2);
    lcd.setCursor(5, HUD_Y + (HUD_H - 16) / 2);
    lcd.print(buf);
    strcpy(lastClock, buf);
  }

  if (!controlsDrawn) {
    drawClockButton(165, 42, "H+");
    drawClockButton(215, 42, "M+");
    drawClockButton(265, 46, "00s");
    controlsDrawn = true;
  }
}

void setup() {
  lcd.init();
  lcd.setRotation(1);
  lcd.setBrightness(255);
  lcd.fillScreen(0x0000);

  const size_t framePixels = (size_t)WORLD_W * WORLD_H;
  const size_t frameBytes = framePixels * sizeof(*frame565);

  frame565 = (uint16_t*)heap_caps_malloc(frameBytes, MALLOC_CAP_DMA | MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  trail8   = (uint8_t*)heap_caps_malloc(framePixels, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  jobs[0]  = (ForceJob*)heap_caps_malloc(sizeof(ForceJob) * MAX_JOBS_PER_CORE, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  jobs[1]  = (ForceJob*)heap_caps_malloc(sizeof(ForceJob) * MAX_JOBS_PER_CORE, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  partsCur = (Particle*)heap_caps_malloc(sizeof(Particle) * NUM_PARTICLES, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  partsNext= (Particle*)heap_caps_malloc(sizeof(Particle) * NUM_PARTICLES, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  idCur    = (uint16_t*)heap_caps_malloc(sizeof(uint16_t) * NUM_PARTICLES, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  idNext   = (uint16_t*)heap_caps_malloc(sizeof(uint16_t) * NUM_PARTICLES, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  forceScale = (uint32_t (*)[INV_LEVELS][MAX_TYPES])heap_caps_malloc(
      sizeof(uint32_t) * MAX_TYPES * INV_LEVELS * MAX_TYPES, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

  if (!frame565 || !trail8 || !jobs[0] || !jobs[1] || !partsCur || !partsNext || !idCur || !idNext || !forceScale) {
    while (1) delay(1000);
  }

  memset(frame565, 0, frameBytes);
  memset(trail8, 0, framePixels);

  const uint16_t hudPanel = lcd.color565(4, 7, 18);
  lcd.fillRect(0, 0, WORLD_W, ORIGIN_Y - 1, hudPanel);
  lcd.fillRect(0, HUD_Y + 1, WORLD_W, HUD_H - 1, hudPanel);
  lcd.drawFastHLine(0, ORIGIN_Y - 1, WORLD_W, lcd.color565(28, 32, 44));
  lcd.drawFastHLine(0, HUD_Y,     WORLD_W, lcd.color565(28, 32, 44));

  initGammaLUT();
  // 6タイプ分の基色（緑 / 赤 / 黄 / シアン / マゼンタ / 青）
  const uint8_t baseR[MAX_TYPES] = {
    gammaLut[60], gammaLut[255], gammaLut[255],
    gammaLut[60], gammaLut[255], gammaLut[90],
  };
  const uint8_t baseG[MAX_TYPES] = {
    gammaLut[255], gammaLut[60], gammaLut[220],
    gammaLut[220], gammaLut[60], gammaLut[110],
  };
  const uint8_t baseB[MAX_TYPES] = {
    gammaLut[80], gammaLut[60], gammaLut[60],
    gammaLut[255], gammaLut[255], gammaLut[255],
  };

  for (int i = 0; i < 256; i++) {
    pal565[i] = 0;
    trail565[i] = 0;
  }
  for (int t = 0; t < MAX_TYPES; t++) {
    const uint8_t colorIndex = (uint8_t)(t + 1);
    pal565[colorIndex] = lgfx::swap565(baseR[t], baseG[t], baseB[t]);
    for (int level = 1; level < TRAIL_LEVELS; level++) {
      const uint8_t r = (uint16_t)baseR[t] * level / (TRAIL_LEVELS - 1);
      const uint8_t g = (uint16_t)baseG[t] * level / (TRAIL_LEVELS - 1);
      const uint8_t b = (uint16_t)baseB[t] * level / (TRAIL_LEVELS - 1);
      trail565[(colorIndex << 4) | level] = lgfx::swap565(r, g, b);
    }
  }
  arenaBorder565 = lgfx::swap565(10, 28, 50);
  arenaBackground565 = lgfx::swap565(2, 5, 14);

  for (size_t i = 0; i < framePixels; i++) {
    frame565[i] = arenaBackground565;
  }
  for (int x = 0; x < WORLD_W; x++) {
    frame565[x] = arenaBorder565;
    frame565[(WORLD_H - 1) * WORLD_W + x] = arenaBorder565;
  }
  for (int y = 0; y < WORLD_H; y++) {
    frame565[y * WORLD_W] = arenaBorder565;
    frame565[y * WORLD_W + WORLD_W - 1] = arenaBorder565;
  }

  for (int old = 0; old < 256; old++) {
    uint8_t next = 0;
    if (old) {
      uint8_t level = (uint8_t)((old & TRAIL_MASK) - 1);
      if (level) {
        next = (uint8_t)((old & 0xF0) | level);
      }
    }
    uint16_t color = next ? trail565[next] : arenaBackground565;
    trailStep[old] = (uint32_t)next | ((uint32_t)color << 16);
  }

  touchSpi.begin(TOUCH_CLK, TOUCH_MISO, TOUCH_MOSI, TOUCH_CS);
  ts.begin(touchSpi);
  ts.setRotation(1);

  clockHour = 0; clockMin = 0; clockSec = 0;
  lastClockTick = millis();

  for (int t = 0; t < MAX_TYPES; t++) {
    typeColor[t] = t + 1;
  }

  initLUT();
  initParticles(currentPreset);
  setPreset(currentPreset);

  semEmpty     = xSemaphoreCreateCounting(2, 2);
  semFull      = xSemaphoreCreateCounting(2, 0);
  semForceGo   = xSemaphoreCreateBinary();
  semForceDone = xSemaphoreCreateBinary();

  if (!semEmpty || !semFull || !semForceGo || !semForceDone) {
    while (1) delay(1000);
  }

  xTaskCreatePinnedToCore(forceWorker, "force", 4096, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(simMaster, "master", 4096, NULL, 2, NULL, 0);
}

// --- Core 1: メイン描画・入力処理ループ ---
void loop() {
  static bool wasTouched = false;
  static unsigned long lastTap = 0;
  static unsigned long touchReadyAt = 0;

  if (touchReadyAt == 0) touchReadyAt = millis() + 1000;

  tickClock();

  // タッチ操作処理
  if (ts.touched()) {
    TS_Point p = ts.getPoint();
    int tx = map(p.x, TOUCH_MIN_X, TOUCH_MAX_X, 0, WORLD_W);
    int ty = map(p.y, TOUCH_MIN_Y, TOUCH_MAX_Y, 0, 240);

    if (millis() >= touchReadyAt && !wasTouched && millis() - lastTap > 200) {
      if (ty >= HUD_Y) {
        if      (tx >= 165 && tx < 207) clockHour = (clockHour + 1) % 24;
        else if (tx >= 215 && tx < 257) clockMin  = (clockMin  + 1) % 60;
        else if (tx >= 265 && tx < 311) clockSec  = 0;
      } else if (ty >= ORIGIN_Y) {
        wantNextPreset = true;
      }
      lastTap = millis();
    }
    wasTouched = true;
  } else {
    wasTouched = false;
  }

  static int r = 0;
  static bool pushed = false;

  if (pushed) {
    endPush();
    r ^= 1;
  }

  // HUD定期更新（250ms間隔）
  unsigned long nowMs = millis();
  static unsigned long lastHud = 0;
  if (nowMs - lastHud >= 250) {
    lastHud = nowMs;
    drawHUD();
  }

  if (xSemaphoreTake(semFull, portMAX_DELAY) == pdTRUE) {
    unsigned long stage0 = micros();
    buildFrame(r);
    buildUsAvg = (buildUsAvg * 7 + (uint32_t)(micros() - stage0)) / 8;
    stage0 = micros();
    startPush();
    pushUsAvg = (pushUsAvg * 7 + (uint32_t)(micros() - stage0)) / 8;
    xSemaphoreGive(semEmpty);
    pushed = true;
  }

  // 描画周期を一定化し、Core 1 の力計算時間を確保するため TARGET_FPS で待機
  static TickType_t frameWake;
  static bool frameWakeInit = false;
  if (!frameWakeInit) {
    frameWake = xTaskGetTickCount();
    frameWakeInit = true;
  }
  vTaskDelayUntil(&frameWake, pdMS_TO_TICKS(1000 / TARGET_FPS));
}
