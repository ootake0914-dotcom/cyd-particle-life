#ifndef CONFIG_H
#define CONFIG_H

#define LGFX_USE_V1
#include <LovyanGFX.hpp>

// ============================================================================
//  CYD Display Configuration for LovyanGFX
//  Target Board: ESP32-2432S028R (Cheap Yellow Display / CYD)
//
//  Supported Hardware Variants:
//    - DISP_ST7789: CYD2 (ST7789 panel, default)
//    - DISP_ILI9341: CYD1 (Standard ILI9341 panel)
//    - DISP_ILI9341_INV: CYD1 (Inverted colors ILI9341 panel)
//
//  Powered by LovyanGFX (FreeBSD License by @lovyan03)
// ============================================================================

#if !defined(DISP_ILI9341) && !defined(DISP_ILI9341_INV) && !defined(DISP_ST7789)
#define DISP_ST7789
#endif

class LGFX_CYD : public lgfx::LGFX_Device {
#if defined(DISP_ILI9341) || defined(DISP_ILI9341_INV)
  lgfx::Panel_ILI9341 _panel;
#else
  lgfx::Panel_ST7789  _panel;
#endif
  lgfx::Bus_SPI       _bus;
  lgfx::Light_PWM     _light;
public:
  LGFX_CYD() {
    {
      auto cfg = _bus.config();
      cfg.spi_host = VSPI_HOST;
      cfg.spi_mode = 0;
      cfg.freq_write = 80000000; // 80MHz SPI DMA
      cfg.freq_read  = 16000000;
      cfg.spi_3wire  = false;
      cfg.use_lock   = true;
      cfg.dma_channel = 1;
      cfg.pin_sclk = 14;
      cfg.pin_mosi = 13;
      cfg.pin_miso = 12;
      cfg.pin_dc   = 2;
      _bus.config(cfg);
      _panel.setBus(&_bus);
    }
    {
      auto cfg = _panel.config();
      cfg.pin_cs           = 15;
      cfg.pin_rst          = -1;
      cfg.pin_busy         = -1;
      cfg.panel_width      = 240;
      cfg.panel_height     = 320;
      cfg.offset_x         = 0;
      cfg.offset_y         = 0;
      cfg.offset_rotation  = 0;
      cfg.dummy_read_pixel = 8;
      cfg.dummy_read_bits  = 1;
      cfg.readable         = true;
#if defined(DISP_ILI9341_INV)
      cfg.invert           = true;
#else
      cfg.invert           = false;
#endif
      cfg.rgb_order        = false;
      cfg.dlen_16bit       = false;
      cfg.bus_shared       = true;
      _panel.config(cfg);
    }
    {
      auto cfg = _light.config();
      cfg.pin_bl = 21;
      cfg.invert = false;
      cfg.freq   = 1000;
      cfg.pwm_channel = 7;
      _light.config(cfg);
      _panel.setLight(&_light);
    }
    setPanel(&_panel);
  }

  // DMA direct RGB565 push without per-pixel conversion
  void push565DMA(int32_t x, int32_t y, int32_t w, int32_t h,
                  const uint16_t* data)
  {
    auto pc = create_pc_fast(data, false);
    pc.src_bitwidth = w;
    startWrite();
    panel()->writeImage(x, y, w, h, &pc, true);
  }
};

#endif // CONFIG_H
