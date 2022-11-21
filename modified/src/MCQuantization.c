/*
 *  MCQuantization.c
 *  dither Utility
 *
 *  Quantization algorithm implementation.
 *
 */

#include <MCQuantization.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <MedianPartition.h>
#include <XMalloc.h>

#define NUM_DIM 3u

/**
 * Enumeration of color dimensions in pixels.
 *
 * Used to track which dimension has the greatest range.
 */
typedef enum {
    DIM_RED,
    DIM_GREEN,
    DIM_BLUE
} color_dim_t;

typedef struct {
    DTPixel min;
    DTPixel max;
    size_t size;
    uint8_t *r;
    uint8_t *g;
    uint8_t *b;
} MCCube;

struct mc_workspace_t {
    mc_byte_t level;
    MCCube *cubes;
    DTPalette *palette;
};

MCWorkspace *
MCWorkspaceMake(mc_byte_t level)
{
    MCWorkspace *ws = XMalloc(sizeof(MCWorkspace));
    ws->level = level;
    ws->palette = XMalloc(sizeof(DTPalette));
    ws->palette->size = 1 << level;
    ws->palette->colors = XMalloc(sizeof(DTPixel) * ws->palette->size);
    ws->cubes = XMalloc(sizeof(MCCube) * ws->palette->size);
    return ws;
}

void
MCWorkspaceDestroy(MCWorkspace *ws)
{
    free(ws->cubes);
    free(ws->palette);
    free(ws);
}

static void
MCShrinkCube(
    MCCube *cube
) {
    mc_byte_t r, g, b;

    cube->min = (DTPixel) { .r = 255, .g = 255, .b = 255 };
    cube->max = (DTPixel) { .r = 0, .g = 0, .b = 0 };

    for (size_t i = 0; i < cube->size; i++) {
        r = cube->r[i];
        g = cube->g[i];
        b = cube->b[i];

        if (r < cube->min.r) cube->min.r = r;
        if (g < cube->min.g) cube->min.g = g;
        if (b < cube->min.b) cube->min.b = b;

        if (r > cube->max.r) cube->max.r = r;
        if (g > cube->max.g) cube->max.g = g;
        if (b > cube->max.b) cube->max.b = b;
    }
}

static color_dim_t
MCCalculateBiggestDimension(
    MCCube *cube
) {
    mc_byte_t r = cube->max.r - cube->min.r;
    mc_byte_t g = cube->max.g - cube->min.g;
    mc_byte_t b = cube->max.b - cube->min.b;

    if (r >= g && r >= b) {
        return DIM_RED;
    } else if (g >= r && g >= b) {
        return DIM_GREEN;
    } else {
        return DIM_BLUE;
    }
}

/**
 * @brief Splits lo across its median into [lo, hi].
 * @param lo The cube to be split and the output lo half.
 * @param hi The output hi half.
 */
static void
MCSplit(
    MCCube *lo,
    MCCube *hi
) {
    assert(lo);
    assert(hi);

    // Determine which color has the biggest range.
    color_dim_t dim = MCCalculateBiggestDimension(lo);

    // Partition across the median in the selected dimension.
    size_t mid = 0;
    switch (dim) {
        case DIM_RED:
            mid = MedianPartition(lo->r, lo->g, lo->b, lo->size);
            break;
        case DIM_GREEN:
            mid = MedianPartition(lo->g, lo->r, lo->b, lo->size);
            break;
        case DIM_BLUE:
            mid = MedianPartition(lo->b, lo->g, lo->r, lo->size);
            break;
        default:
            assert(0);
    }

    // Split the cubes by size.
    *hi = *lo;
    lo->size = mid + 1;
    hi->r += lo->size;
    hi->g += lo->size;
    hi->b += lo->size;
    hi->size -= lo->size;

    // Shrink the value range of the cubes.
    MCShrinkCube(lo);
    MCShrinkCube(hi);
}

static DTPixel
MCCubeAverage(
    MCCube *cube
) {
    return (DTPixel) {
        .r = (cube->max.r + cube->min.r) >> 1,
        .g = (cube->max.g + cube->min.g) >> 1,
        .b = (cube->max.b + cube->min.b) >> 1
    };
}

DTPalette *
MCQuantizeData(
    SplitImage *img,
    MCWorkspace *ws
) {
    assert(img);
    assert(ws);

    size_t size = img->w * img->h;

    /* first cube */
    ws->cubes[0] = (MCCube) {
       .r = img->r,
       .g = img->g,
       .b = img->b,
       .size = size
    };
    MCShrinkCube(ws->cubes);

    /* remaining cubes */
    size_t parentIndex = 0;
    int iLevel = 1; /* iteration level */
    size_t offset;
    MCCube *parentCube;
    while (iLevel <= ws->level)
    {
        // Partition the cube across the median.
        parentCube = &ws->cubes[parentIndex];
        offset = ws->palette->size >> iLevel;
        MCSplit(parentCube, &ws->cubes[parentIndex + offset]);

        /* check if iLevel must be increased by analysing if the next
         * offset is within palette size boundary. If not, change level
         * and reset parent to 0. If it is, set next element as parent. */
        if (parentIndex + (offset * 2) < ws->palette->size) {
            parentIndex = parentIndex + (offset * 2);
        } else {
            parentIndex = 0;
            iLevel++;
        }
    }

    /* find final cube averages */
    for (size_t i = 0; i < ws->palette->size; i++)
        ws->palette->colors[i] = MCCubeAverage(&ws->cubes[i]);

    DTPalette *ret = ws->palette;
    ws->palette = NULL;
    return ret;
}
