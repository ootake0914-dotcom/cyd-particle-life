# Build and Installation Guide

This document describes how to compile and flash CYD Particle Life using the Arduino IDE.

---

## Hardware Requirements
- **Target Board:** ESP32-2432S028R (Cheap Yellow Display / CYD)
- **Display:** 2.8" SPI TFT (240x320 resolution, ST7789)
- **Touch Controller:** XPT2046 Resistive Touch Controller

---

## Software Prerequisites
1. **Arduino IDE 2.x** (or Arduino IDE 1.8.x)
2. **ESP32 Arduino Core** (version 2.0.x or 3.x)
   - Board Manager URL: https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

---

## Required Libraries
Install the following libraries via the Arduino Library Manager:

1. **LovyanGFX** (by lovyan03)
   - Search LovyanGFX in the Library Manager and install the latest version.
   - Used for high-speed asynchronous DMA SPI display output.
2. **XPT2046_Touchscreen** (by Paul Stoffregen)
   - Search XPT2046_Touchscreen in the Library Manager and install it.
   - Used for reading resistive touch inputs on the CYD.

---

## Arduino IDE Settings
Select the following board configuration in Arduino IDE:

- **Board:** ESP32 Dev Module
- **CPU Frequency:** 240MHz (WiFi/BT)
- **Flash Frequency:** 80MHz
- **Flash Mode:** QIO
- **Flash Size:** 4MB (32Mb)
- **Partition Scheme:** Huge APP (3MB No OTA/1MB SPIFFS) or Default 4MB with ffat
- **Core Debug Level:** None
- **PSRAM:** Disabled (or Enabled if your board has PSRAM)

---

## Display Variant Configuration

Open `config.h` and select your display panel type before compiling (default is `DISP_ST7789` for CYD2 USB-C):

- **CYD2 (ST7789):** Default configuration (`#define DISP_ST7789`) for the original source code.
- **CYD1 (ILI9341):** Add `#define DISP_ILI9341` at the top of `config.h`.
- **CYD1 (ILI9341 Inverted Colors):** Add `#define DISP_ILI9341_INV` at the top of `config.h`.

---

## Compilation and Flashing Steps
1. Clone or download this repository.
2. Open `cyd-particle-life.ino` in the Arduino IDE.
3. Configure your display variant in `config.h` if needed.
4. Connect your CYD board to your PC via a USB cable.
5. Select the corresponding COM port in the Arduino IDE.
6. Click **Upload** to compile and flash the firmware.
