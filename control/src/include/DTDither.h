/*
 *  DTDither.h
 *  dither Utility
 *
 *  Dithering methods declarations
 *
 */

#ifndef DT_DITHER
#define DT_DITHER

#include <DTImage.h>
#include <DTPalette.h>

typedef struct {
    unsigned long long shift_time;
    unsigned long long shift_units;
    unsigned long long dither_time;
    unsigned long long dither_units;
    unsigned long long deshift_time;
    unsigned long long deshift_units;
} dt_time_t;

void DTTimeInit(dt_time_t *time);
void DTTimeReport(dt_time_t *time);

void ApplyFloydSteinbergDither(DTImage *image, DTPalette *palette, palette_time_t *palette_time);

#endif
