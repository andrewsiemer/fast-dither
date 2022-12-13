/*
 *  DTImage.h
 *  dither Utility
 *
 *  Basic image related structures and function declarations.
 *
 */

#ifndef DT_IMAGE
#define DT_IMAGE

#include <stddef.h>
#include <stdint.h>

#define e_PPM ".ppm"
#define e_PNG ".png"

typedef unsigned char byte;

typedef struct {
    byte r, g, b;
} DTPixel;

typedef struct {
    int16_t r, g, b;
} DTDiff;

typedef enum {
    t_PPM,
    t_PNG,
    t_UNKNOWN
} DTImageType;

typedef struct {
    size_t width;
    size_t height;
    DTImageType type;
    unsigned long resolution;
    DTPixel *pixels;
} DTImage;

DTImage *CreateImageFromFile(char *filename);
void WriteImageToFile(DTImage *img, char *filename);

DTPixel PixelFromRGB(byte r, byte g, byte b);

#endif
