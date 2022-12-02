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
#include <UtilMacro.h>

#include <immintrin.h>

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

// LOOOOOOONG BOI
static void
MCShrinkCube(
    MCCube *cube
) {
    unsigned long long ts1, ts2;
    mc_byte_t r, g, b;
    size_t r_pre, r_post;
    register __m256i r_min, r_max, g_min, g_max, b_min, b_max;
    register __m256i r_tmp1, r_tmp2, r_tmp3;
    register __m256i g_tmp1, g_tmp2, g_tmp3;
    register __m256i b_tmp1, b_tmp2, b_tmp3;

    TIMESTAMP(ts1);

    // Figure out which parts are unaligned, so their min/max can be collected.
    r_pre = 32 - (((uintptr_t) cube->r) & 0x1F);
    r_pre = (r_pre == 32) ? 0 : r_pre;
    r_pre = MIN(r_pre, cube->size);
    r_post = (cube->size - r_pre) % 32;

    cube->min = (DTPixel) { .r = 255, .g = 255, .b = 255 };
    cube->max = (DTPixel) { .r = 0, .g = 0, .b = 0 };

    // Perform the aligned min/max.
    if ((r_pre + r_post) < cube->size) {
        size_t chunks = (cube->size - (r_pre + r_post)) / 32;
        __m256i *r_align = (__m256i*) &cube->r[r_pre];
        __m256i *g_align = (__m256i*) &cube->g[r_pre];
        __m256i *b_align = (__m256i*) &cube->b[r_pre];
        r_max = g_max = b_max = _mm256_setzero_si256();
        r_min = g_min = b_min = _mm256_cmpeq_epi8(r_max, r_max);

        for (size_t i = 0; i < (chunks % 3); i++) {
            r_tmp1 = _mm256_load_si256(&r_align[i]);
            g_tmp1 = _mm256_load_si256(&g_align[i]);
            b_tmp1 = _mm256_load_si256(&b_align[i]);
            r_min = _mm256_min_epu8(r_min, r_tmp1);
            g_min = _mm256_min_epu8(g_min, g_tmp1);
            b_min = _mm256_min_epu8(b_min, b_tmp1);
            r_max = _mm256_max_epu8(r_max, r_tmp1);
            g_max = _mm256_max_epu8(g_max, g_tmp1);
            b_max = _mm256_max_epu8(b_max, b_tmp1);
        }

        for (size_t i = chunks % 3; i < chunks; i += 3) {
            r_tmp1 = _mm256_load_si256(&r_align[i]);
            g_tmp1 = _mm256_load_si256(&g_align[i]);
            b_tmp1 = _mm256_load_si256(&b_align[i]);
            r_tmp2 = _mm256_load_si256(&r_align[i+1]);
            g_tmp2 = _mm256_load_si256(&g_align[i+1]);
            b_tmp2 = _mm256_load_si256(&b_align[i+1]);
            r_tmp3 = _mm256_load_si256(&r_align[i+2]);
            g_tmp3 = _mm256_load_si256(&g_align[i+2]);
            b_tmp3 = _mm256_load_si256(&b_align[i+2]);
            r_min = _mm256_min_epu8(r_min, r_tmp1);
            g_min = _mm256_min_epu8(g_min, g_tmp1);
            b_min = _mm256_min_epu8(b_min, b_tmp1);
            r_min = _mm256_min_epu8(r_min, r_tmp2);
            g_min = _mm256_min_epu8(g_min, g_tmp2);
            b_min = _mm256_min_epu8(b_min, b_tmp2);
            r_min = _mm256_min_epu8(r_min, r_tmp3);
            g_min = _mm256_min_epu8(g_min, g_tmp3);
            b_min = _mm256_min_epu8(b_min, b_tmp3);
            r_max = _mm256_max_epu8(r_max, r_tmp1);
            g_max = _mm256_max_epu8(g_max, g_tmp1);
            b_max = _mm256_max_epu8(b_max, b_tmp1);
            r_max = _mm256_max_epu8(r_max, r_tmp2);
            g_max = _mm256_max_epu8(g_max, g_tmp2);
            b_max = _mm256_max_epu8(b_max, b_tmp2);
            r_max = _mm256_max_epu8(r_max, r_tmp3);
            g_max = _mm256_max_epu8(g_max, g_tmp3);
            b_max = _mm256_max_epu8(b_max, b_tmp3);
        }

        __attribute__((aligned(32))) uint8_t r_min_a[32];
        __attribute__((aligned(32))) uint8_t g_min_a[32];
        __attribute__((aligned(32))) uint8_t b_min_a[32];
        __attribute__((aligned(32))) uint8_t r_max_a[32];
        __attribute__((aligned(32))) uint8_t g_max_a[32];
        __attribute__((aligned(32))) uint8_t b_max_a[32];
        _mm256_store_si256((__m256i*) r_min_a, r_min);
        _mm256_store_si256((__m256i*) g_min_a, g_min);
        _mm256_store_si256((__m256i*) b_min_a, b_min);
        _mm256_store_si256((__m256i*) r_max_a, r_max);
        _mm256_store_si256((__m256i*) g_max_a, g_max);
        _mm256_store_si256((__m256i*) b_max_a, b_max);

        for (size_t i = 0; i < 32; i++) {
            cube->min.r = MIN(r_min_a[i], cube->min.r);
            cube->min.g = MIN(g_min_a[i], cube->min.g);
            cube->min.b = MIN(b_min_a[i], cube->min.b);

            cube->max.r = MAX(r_max_a[i], cube->max.r);
            cube->max.g = MAX(g_max_a[i], cube->max.g);
            cube->max.b = MAX(b_max_a[i], cube->max.b);
        }
    }

    for (size_t i = 0; i < r_pre; i++) {
        r = cube->r[i];
        g = cube->g[i];
        b = cube->b[i];

        cube->min.r = MIN(r, cube->min.r);
        cube->min.g = MIN(g, cube->min.g);
        cube->min.b = MIN(b, cube->min.b);

        cube->max.r = MAX(r, cube->max.r);
        cube->max.g = MAX(g, cube->max.g);
        cube->max.b = MAX(b, cube->max.b);
    }

    for (size_t i = cube->size - r_post; i < cube->size; i++) {
        r = cube->r[i];
        g = cube->g[i];
        b = cube->b[i];

        cube->min.r = MIN(r, cube->min.r);
        cube->min.g = MIN(g, cube->min.g);
        cube->min.b = MIN(b, cube->min.b);

        cube->max.r = MAX(r, cube->max.r);
        cube->max.g = MAX(g, cube->max.g);
        cube->max.b = MAX(b, cube->max.b);
    }

    TIMESTAMP(ts2);
    TIME_REPORT("MCShrinkCube", ts1, ts2);
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
    unsigned long long ts1, ts2;
    switch (dim) {
        case DIM_RED:
            TIMESTAMP(ts1);
            mid = MedianPartition(lo->r, lo->g, lo->b, lo->size);
            TIMESTAMP(ts2);
            break;
        case DIM_GREEN:
            TIMESTAMP(ts1);
            mid = MedianPartition(lo->g, lo->r, lo->b, lo->size);
            TIMESTAMP(ts2);
            break;
        case DIM_BLUE:
            TIMESTAMP(ts1);
            mid = MedianPartition(lo->b, lo->g, lo->r, lo->size);
            TIMESTAMP(ts2);
            break;
        default:
            assert(0);
    }
    TIME_REPORT("Median Partition", ts1, ts2);

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
