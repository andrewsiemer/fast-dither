/*
 *  DTPalette.h
 *  dither Utility
 *
 *  Palette generation, type and function declarations.
 *
 */

#ifndef DT_PALETTE
#define DT_PALETTE

#include <stddef.h>
#include <DTImage.h>

typedef struct {
    size_t size;
    DTPixel *colors;
} DTPalette;

typedef struct {
    unsigned long long search_time;
    unsigned long long search_units;
} palette_time_t;

DTPalette *StandardPaletteBW(size_t size);
DTPalette *StandardPaletteRGB(void);

DTPixel FindClosestColorFromPalette(DTPixel pixel, DTPalette *palette, palette_time_t *time);

void PaletteTimeInit(palette_time_t *time);
void PaletteTimeReport(palette_time_t *time);

#endif
