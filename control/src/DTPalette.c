/*
 *  DTPalette.c
 *  dither Utility
 *
 *  Palette functions implementation.
 *
 */

#include <DTPalette.h>
#include <stdlib.h>
#include <math.h>
#include <UtilMacro.h>
#include <assert.h>

DTPalette *
StandardPaletteBW(size_t size)
{
    if (size < 2) return NULL;

    DTPalette *palette = malloc(sizeof(DTPalette));
    palette->size = size;
    palette->colors = malloc(sizeof(DTPixel) * size);

    float step = 255.0f / (size - 1);
    for (size_t i = 0; i < size; i++)
        palette->colors[i] = PixelFromRGB(
            (byte) (float) roundf(i*step),
            (byte) (float) roundf(i*step),
            (byte) (float) roundf(i*step)
        );

    return palette;
}

DTPalette *
StandardPaletteRGB()
{
    DTPalette *palette = malloc(sizeof(DTPalette));
    palette->size = 8;
    palette->colors = malloc(sizeof(DTPixel) * 3);
    palette->colors[0] = PixelFromRGB(0xFF, 0x00, 0x00);
    palette->colors[1] = PixelFromRGB(0x00, 0xFF, 0x00);
    palette->colors[2] = PixelFromRGB(0x00, 0x00, 0xFF);
    palette->colors[3] = PixelFromRGB(0x00, 0xFF, 0xFF);
    palette->colors[4] = PixelFromRGB(0xFF, 0x00, 0xFF);
    palette->colors[5] = PixelFromRGB(0xFF, 0xFF, 0x00);
    palette->colors[6] = PixelFromRGB(0x00, 0x00, 0x00);
    palette->colors[7] = PixelFromRGB(0xFF, 0xFF, 0xFF);

    return palette;
}

DTPixel
FindClosestColorFromPalette(DTPixel needle, DTPalette *palette, palette_time_t *time)
{
    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);
    // search for smallest Euclidean distance
    size_t index = 0;
    int d, minimal = 255 * 255 * 3 + 1;
    int dR, dG, dB;
    DTPixel current;

    for (size_t i = 0; i < palette->size; i++) {
        current = palette->colors[i];
        dR = needle.r - current.r;
        dG = needle.g - current.g;
        dB = needle.b - current.b;
        d = dR * dR + dG * dG + dB * dB;
        if (d < minimal) {
            minimal = d;
            index = i;
        }
    }

    TIMESTAMP(ts2);
    time->search_time += (ts2 - ts1);
    time->search_units += palette->size*3;

    return palette->colors[index];
}

void
PaletteTimeInit(palette_time_t *time) {
    assert(time);
    *time = (palette_time_t) { .search_time = 0 };
}

void
PaletteTimeReport(palette_time_t *time) {
    const double ops_per_pix = 3.0;
    const double pix_per_kernel = 32.0;
    const double ops_per_kernel = pix_per_kernel*ops_per_pix; // 6 mullo * 16 way SIMD
    const double op_throughput = 0.5; // mullo: 2 every 1 cycles
    const double search_theoretical = ops_per_kernel/TIME_NORM(0, ops_per_kernel*op_throughput); // iops/cycle

    double search_time = TIME_NORM(0, time->search_time);
    double search_perf = (((double)time->search_units) / search_time); // iops/cycle
    double search_pix = search_perf/3; // pixels/cycle
    double search_peak = (search_perf / search_theoretical) * 100;

    printf("Palette Search%6s%-20.6lf%-20.6lf%.2lf%%\n", "", search_time, search_pix, search_peak);
}
