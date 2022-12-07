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

DTPalettePacked *StandardPaletteBW(size_t size);
DTPalettePacked *StandardPaletteRGB(void);

DTPixel FindClosestColorFromPalette(DTPixel pixel, DTPalettePacked *palette);
DTPixel FindClosestColorFromPaletteDiff(DTPixelDiff pixel, DTPalettePacked *palette);

#endif
