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

DTPalette *StandardPaletteBW(size_t size);
DTPalette *StandardPaletteRGB(void);

DTPixel FindClosestColorFromPalette(DTPixel pixel, DTPalette *palette);

#endif
