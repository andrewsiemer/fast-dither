/*
 *  main.c
 *  dither Utility
 *
 *  Created on August 2017 by Cesar Tessarin.
 *  Rewritten and expanded from the 2010 version.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <DTImage.h>
#include <DTDither.h>
#include <DTPalette.h>
#include <MCQuantization.h>
#include <XMalloc.h>
#include <UtilMacro.h>

DTPalettePacked *PaletteForIdentifier(char *s, DTImage *img);
DTPalettePacked *ReadPaletteFromStdin(size_t size);
DTPalettePacked *QuantizedPaletteForImage(DTImage *image, size_t size);

int
main(int argc, char ** argv)
{
    char *paletteID = NULL;
    char *inputFile, *outputFile;
    int verbose = 0;
    int dither = 1;
    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "dvp:")) != -1) {
        switch (c) {
            case 'p':
                paletteID = optarg;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'd':
                dither = 0;
                break;
            case '?':
                fprintf(stderr, "Ignoring unknown option: -%c\n", optopt);
        }
    }

    /* check if there is still two arguments remaining, for i/o */
    if (argc - optind != 2) {
        fprintf(stderr,
            "Usage: %s [-p palette[.size]] [-dv] input output\n", argv[0]);
        return 1;
    }

    /* all arguments ok */
    inputFile = argv[optind];
    outputFile = argv[optind+1];

    DTImage *input = CreateImageFromFile(inputFile);
    if (input == NULL) return 2;

    DTPalettePacked *palette = PaletteForIdentifier(paletteID, input);
    if (palette == NULL) return 3;

    /* dump palette if verbose option was set */
    if (verbose)
        for (size_t i = 0; i < palette->size; i++)
            printf("%d %d %d\n",
                palette->colors[i],
                palette->colors[palette->size+i],
                palette->colors[palette->size*2+i]
            );

    palette_time_t palette_time;
    PaletteTimeInit(&palette_time);
    if (dither) {
        ApplyFloydSteinbergDither(input, palette, &palette_time);
    } else {
        /* closest color only */
        DTPixel *pixel;
        for (size_t i = 0; i < input->height; i++) {
            for (size_t j = 0; j < input->width; j++) {
                pixel = &input->pixels[i*input->width + j];
                *pixel = FindClosestColorFromPalette(*pixel, palette, &palette_time);
            }
        }
    }
    PaletteTimeReport(&palette_time);

    WriteImageToFile(input, outputFile);

    XFree(palette->colors);
    XFree(palette);

    return 0;
}

DTPalettePacked *
PaletteForIdentifier(char *str, DTImage *image)
{
    // if (str == NULL) return StandardPaletteRGB();

    char *name, *sizeStr;
    char *sep = (char*) ".";
    name = strtok(str, sep);
    sizeStr = strtok(NULL, sep);
    size_t size = 0;

    /* if size was inserted, transform to int and check if is valid */
    if (sizeStr) {
        size = strtoul(sizeStr, (char **)NULL, 10);
        if (size <= 0) {
            fprintf(stderr, "Invalid palette size, aborting.\n");
            return NULL;
        }
    }

    /* RGB */
    if (strcmp(name, "rgb") == 0) {
        if (size) fprintf(stderr, "Ignored palette size.\n");
        return StandardPaletteRGB();
    }

    if (strcmp(name, "bw") == 0) {
        if (size == 1) {
            fprintf(stderr,
                    "Invalid palette size for B&W. Must be at least 2.\n");
            return NULL;
        }
        if (!size) size = 2;
        return StandardPaletteBW(size);
    }

    if (strcmp(name, "custom") == 0) {
        if (!size) {
            fprintf(stderr, "Size required for custom palette, aborting.\n");
            return NULL;
        }
        return ReadPaletteFromStdin(size);
    }

    if (strcmp(name, "auto") == 0) {
        if (!size) {
            fprintf(stderr,
                    "Size required for automatic palette, aborting.\n");
            return NULL;
        }
        if (size % 16 != 0) { // PalletSearch requires pow 16
            fprintf(stderr, "Size must be a power of 16, aborting.\n");
            return NULL;
        }
        return QuantizedPaletteForImage(image, size);
    }

    /* unknown palette */
    fprintf(stderr, "Unrecognized palette identifier, aborting.\n");
    return NULL;
}

DTPalettePacked *
ReadPaletteFromStdin(size_t size)
{
    DTPalettePacked *palette = XMalloc(sizeof(DTPalettePacked));
    palette->size = size;
    palette->colors = XMalloc(size*sizeof(int)*3);

    unsigned int r, g, b;
    for (size_t i = 0; i < size; i++) {
        assert(scanf(" %d %d %d", &r, &g, &b) == 3);
        palette->colors[i] = (byte) r;
        palette->colors[palette->size+i] = (byte) g;
        palette->colors[palette->size*2+i] = (byte) b;
    }

    return palette;
}

DTPalettePacked *
QuantizedPaletteForImage(DTImage *image, size_t size)
{
    MCWorkspace *ws = MCWorkspaceMake((mc_byte_t) (double) log2(size), image->width * image->height);
    DTPalettePacked *palette = XMalloc(sizeof(DTPalettePacked));

    palette->colors = XMalloc(size*sizeof(int)*3);
    palette->size = size;
    printf("Image size: (w, h) = (%zu, %zu)\n", image->width, image->height);

    mc_time_t mc_time;
    MCTimeInit(&mc_time);
    SplitImage *img = CreateSplitImage(image, &mc_time);
    DTPalette *mc = MCQuantizeData(img, ws, &mc_time);
    MCTimeReport(&mc_time);

    for (size_t i = 0; i < palette->size; i++) {
        palette->colors[i] = mc->colors[i].r;
        palette->colors[palette->size+i] = mc->colors[i].g;
        palette->colors[palette->size*2+i] = mc->colors[i].b;
    }

    MCWorkspaceDestroy(ws);
    DestroySplitImage(img);
    XFree(mc->colors);
    XFree(mc);

    return palette;
}

/* vim:set ts=8 sts=4 sw=4 */
