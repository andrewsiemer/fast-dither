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
    size_t size;
    int *colors;
} DTPalettePacked;

typedef struct {
    unsigned long long search_time;
    unsigned long long search_units;
} palette_time_t;

DTPalettePacked *StandardPaletteBW(size_t size);
DTPalettePacked *StandardPaletteRGB(void);

DTPixel FindClosestColorFromPalette(DTPixel pixel, DTPalettePacked *palette, palette_time_t *time);

void PaletteTimeInit(palette_time_t *time);
void PaletteTimeReport(palette_time_t *time);

#endif
