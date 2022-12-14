/*
 *  DTDither.c
 *  dither Utility
 *
 *  Dithering algorithm Implementation.
 *
 */

#include <DTDither.h>
#include <UtilMacro.h>

#include <assert.h>

typedef struct {
    short int r, g, b;
} DTDiff;

DTDiff CalculateDifference(DTPixel original, DTPixel new);
void ApplyDifference(DTPixel *pixel, DTDiff diff, int factor);
byte ByteCap(int n);

void DTTimeInit(dt_time_t *time)
{
    assert(time);
    *time = (dt_time_t) {0};
}

void DTTimeReport(dt_time_t *time)
{
    const double dither_theoretical = (1/1.3125);

    double dither_time = TIME_NORM(0, time->dither_time);
    double dither_pix = ((double)time->dither_units) / dither_time;
    double dither_peak = (dither_pix / dither_theoretical) * 100;

    printf("Dither%19s%-20.6lf%-20.6lf%.2lf%%\n", "", dither_time, dither_pix, dither_peak);
}

void
ApplyFloydSteinbergDither(DTImage *image, DTPalette *palette, palette_time_t *palette_time)
{
    dt_time_t t;
    DTTimeInit(&t);

    unsigned long long ts1, ts2;

    for (size_t i = 0; i < image->height; i++) {
        for (size_t j = 0; j < image->width; j++) {
            DTPixel original = image->pixels[i*image->width + j];
            DTPixel new = FindClosestColorFromPalette(original, palette, palette_time);

            TIMESTAMP(ts1);

            DTDiff diff = CalculateDifference(original, new);

            // disperse
            if (j+1 < image->width)
                ApplyDifference( (
                    image->pixels + i*image->width + j + 1
                ), diff, 7);
            if (i+1 < image->height) {
                if (j-1) ApplyDifference( (
                    image->pixels + (i+1)*image->width + j - 1
                ), diff, 3);
                ApplyDifference( (
                    image->pixels + (i+1)*image->width + j
                ), diff, 5);
                if (j+1 < image->width) ApplyDifference( (
                    image->pixels + (i+1)*image->width + j + 1
                ), diff, 1);
            }

            image->pixels[i*image->width + j] = new;

            TIMESTAMP(ts2);
            t.dither_time += (ts2 - ts1);
            t.dither_units += 1;
        }
    }

    DTTimeReport(&t);
}

DTDiff
CalculateDifference(DTPixel original, DTPixel new)
{
    DTDiff diff;

    diff.r = original.r - new.r;
    diff.g = original.g - new.g;
    diff.b = original.b - new.b;

    return diff;
}

void
ApplyDifference(DTPixel *pixel, DTDiff diff, int factor)
{
    pixel->r = ByteCap(pixel->r + (diff.r * factor / 16));
    pixel->g = ByteCap(pixel->g + (diff.g * factor / 16));
    pixel->b = ByteCap(pixel->b + (diff.b * factor / 16));
}

byte
ByteCap(int num)
{
    if (num > 255) return 255;
    if (num < 0) return 0;

    return (byte) (unsigned int) num;
}
