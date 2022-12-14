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
    DIM_RGB = 0x7,
    DIM_RBG = 0x6,
    DIM_GRB = 0x3,
    DIM_GBR = 0x1,
    DIM_BRG = 0x4,
    DIM_BGR = 0x0
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
    mp_workspace_t mp;
};

MCWorkspace *
MCWorkspaceMake(mc_byte_t level, size_t img_size)
{
    MCWorkspace *ws = XMalloc(sizeof(MCWorkspace));
    ws->level = level;
    ws->palette = XMalloc(sizeof(DTPalette));
    ws->palette->size = 1 << level;
    ws->palette->colors = XMalloc(sizeof(DTPixel) * ws->palette->size);
    ws->cubes = XMalloc(sizeof(MCCube) * ws->palette->size);
    MPWorkspaceInit(&ws->mp, img_size);
    return ws;
}

void
MCWorkspaceDestroy(MCWorkspace *ws)
{
    free(ws->cubes);
    free(ws->palette);
    MPWorkspaceDestroy(&ws->mp);
    free(ws);
}

// LOOOOOOONG BOI
static void
MCShrinkCube(
    MCCube *cube,
    mc_time_t *time
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

        // Load chunks to get us a multiple of three chunks on each cycle.
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

        // Perform remaining cycles, 3 chunks at a time.
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

#if 0
            __builtin_prefetch(&r_align[i+256]);
            __builtin_prefetch(&g_align[i+256]);
            __builtin_prefetch(&b_align[i+256]);
#endif

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

        // Min/max with the unaligned parts.
        r_tmp1 = _mm256_loadu_si256((__m256i*) cube->r);
        g_tmp1 = _mm256_loadu_si256((__m256i*) cube->g);
        b_tmp1 = _mm256_loadu_si256((__m256i*) cube->b);
        r_tmp2 = _mm256_loadu_si256((__m256i*) &cube->r[cube->size - 32]);
        g_tmp2 = _mm256_loadu_si256((__m256i*) &cube->g[cube->size - 32]);
        b_tmp2 = _mm256_loadu_si256((__m256i*) &cube->b[cube->size - 32]);
        r_min = _mm256_min_epu8(r_min, r_tmp1);
        g_min = _mm256_min_epu8(g_min, g_tmp1);
        b_min = _mm256_min_epu8(b_min, b_tmp1);
        r_min = _mm256_min_epu8(r_min, r_tmp2);
        g_min = _mm256_min_epu8(g_min, g_tmp2);
        b_min = _mm256_min_epu8(b_min, b_tmp2);
        r_max = _mm256_max_epu8(r_max, r_tmp1);
        g_max = _mm256_max_epu8(g_max, g_tmp1);
        b_max = _mm256_max_epu8(b_max, b_tmp1);
        r_max = _mm256_max_epu8(r_max, r_tmp2);
        g_max = _mm256_max_epu8(g_max, g_tmp2);
        b_max = _mm256_max_epu8(b_max, b_tmp2);

        // Reduce the remaining vector down.
        r_tmp1 = _mm256_permute2x128_si256(r_min, r_min, 0x01);
        g_tmp1 = _mm256_permute2x128_si256(g_min, g_min, 0x01);
        b_tmp1 = _mm256_permute2x128_si256(b_min, b_min, 0x01);
        r_tmp2 = _mm256_permute2x128_si256(r_max, r_max, 0x01);
        g_tmp2 = _mm256_permute2x128_si256(g_max, g_max, 0x01);
        b_tmp2 = _mm256_permute2x128_si256(b_max, b_max, 0x01);
        r_min = _mm256_min_epu8(r_min, r_tmp1);
        g_min = _mm256_min_epu8(g_min, g_tmp1);
        b_min = _mm256_min_epu8(b_min, b_tmp1);
        r_max = _mm256_max_epu8(r_max, r_tmp2);
        g_max = _mm256_max_epu8(g_max, g_tmp2);
        b_max = _mm256_max_epu8(b_max, b_tmp2);
        r_tmp1 = _mm256_srli_si256(r_min, 8);
        g_tmp1 = _mm256_srli_si256(g_min, 8);
        b_tmp1 = _mm256_srli_si256(b_min, 8);
        r_tmp2 = _mm256_srli_si256(r_max, 8);
        g_tmp2 = _mm256_srli_si256(g_max, 8);
        b_tmp2 = _mm256_srli_si256(b_max, 8);
        r_min = _mm256_min_epu8(r_min, r_tmp1);
        g_min = _mm256_min_epu8(g_min, g_tmp1);
        b_min = _mm256_min_epu8(b_min, b_tmp1);
        r_max = _mm256_max_epu8(r_max, r_tmp2);
        g_max = _mm256_max_epu8(g_max, g_tmp2);
        b_max = _mm256_max_epu8(b_max, b_tmp2);
        r_tmp1 = _mm256_srli_si256(r_min, 4);
        g_tmp1 = _mm256_srli_si256(g_min, 4);
        b_tmp1 = _mm256_srli_si256(b_min, 4);
        r_tmp2 = _mm256_srli_si256(r_max, 4);
        g_tmp2 = _mm256_srli_si256(g_max, 4);
        b_tmp2 = _mm256_srli_si256(b_max, 4);
        r_min = _mm256_min_epu8(r_min, r_tmp1);
        g_min = _mm256_min_epu8(g_min, g_tmp1);
        b_min = _mm256_min_epu8(b_min, b_tmp1);
        r_max = _mm256_max_epu8(r_max, r_tmp2);
        g_max = _mm256_max_epu8(g_max, g_tmp2);
        b_max = _mm256_max_epu8(b_max, b_tmp2);
        r_tmp1 = _mm256_srli_si256(r_min, 2);
        g_tmp1 = _mm256_srli_si256(g_min, 2);
        b_tmp1 = _mm256_srli_si256(b_min, 2);
        r_tmp2 = _mm256_srli_si256(r_max, 2);
        g_tmp2 = _mm256_srli_si256(g_max, 2);
        b_tmp2 = _mm256_srli_si256(b_max, 2);
        r_min = _mm256_min_epu8(r_min, r_tmp1);
        g_min = _mm256_min_epu8(g_min, g_tmp1);
        b_min = _mm256_min_epu8(b_min, b_tmp1);
        r_max = _mm256_max_epu8(r_max, r_tmp2);
        g_max = _mm256_max_epu8(g_max, g_tmp2);
        b_max = _mm256_max_epu8(b_max, b_tmp2);

        // Store the min/max vectors to get the final output.
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

        cube->min.r = MIN(r_min_a[0], r_min_a[1]);
        cube->min.g = MIN(g_min_a[0], g_min_a[1]);
        cube->min.b = MIN(b_min_a[0], b_min_a[1]);
        cube->max.r = MAX(r_max_a[0], r_max_a[1]);
        cube->max.g = MAX(g_max_a[0], g_max_a[1]);
        cube->max.b = MAX(b_max_a[0], b_max_a[1]);
    } else {
        // Not enough values to fill a SIMD vector.
        for (size_t i = 0; i < cube->size; i++) {
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
    }

    TIMESTAMP(ts2);
    time->shrink_time += ts2 - ts1;
    time->shrink_units += cube->size;
}

static color_dim_t
MCCalculateBiggestDimension(
    MCCube *cube
) {
    mc_byte_t r = cube->max.r - cube->min.r;
    mc_byte_t g = cube->max.g - cube->min.g;
    mc_byte_t b = cube->max.b - cube->min.b;

    int r_gt_g = r >= g;
    int r_gt_b = r >= b;
    int g_gt_b = g >= b;

    return (color_dim_t) ((r_gt_g << 2) | (r_gt_b << 1) | g_gt_b);
}

/**
 * @brief Splits lo across its median into [lo, hi].
 * @param lo The cube to be split and the output lo half.
 * @param hi The output hi half.
 */
static void
MCSplit(
    MCCube *lo,
    MCCube *hi,
    mp_workspace_t *ws,
    mc_time_t *time
) {
    assert(lo);
    assert(hi);

    // Determine which color has the biggest range.
    color_dim_t dim = MCCalculateBiggestDimension(lo);

    // Partition across the median in the selected dimension.
    size_t mid = 0;
    unsigned long long ts1, ts2;
    switch (dim) {
        case DIM_RGB:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->r, lo->g, lo->b, lo->size, time);
            TIMESTAMP(ts2);
            break;
        case DIM_RBG:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->r, lo->b, lo->g, lo->size, time);
            TIMESTAMP(ts2);
            break;
        case DIM_GRB:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->g, lo->r, lo->b, lo->size, time);
            TIMESTAMP(ts2);
            break;
        case DIM_GBR:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->g, lo->b, lo->r, lo->size, time);
            TIMESTAMP(ts2);
            break;
        case DIM_BRG:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->b, lo->r, lo->g, lo->size, time);
            TIMESTAMP(ts2);
            break;
        case DIM_BGR:
            TIMESTAMP(ts1);
            mid = MedianPartition(ws, lo->b, lo->g, lo->r, lo->size, time);
            TIMESTAMP(ts2);
            break;
        default:
            assert(0);
    }

    time->mid_time += (ts2 - ts1);
    time->mid_units += lo->size;

    // Split the cubes by size.
    *hi = *lo;
    lo->size = mid + 1;
    hi->r += lo->size;
    hi->g += lo->size;
    hi->b += lo->size;
    hi->size -= lo->size;
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

static void
MCTimeAdd(
    mc_time_t *dst,
    mc_time_t *src
) {
    dst->shrink_time += src->shrink_time;
    dst->shrink_units += src->shrink_units;
    dst->part_time += src->part_time;
    dst->part_units += src->part_units;
    dst->mid_time += src->mid_time;
    dst->mid_units += src->mid_units;
    dst->mc_time += src->mc_time;
    dst->mc_units += src->mc_units;
    dst->align_time += src->align_time;
    dst->align_units += src->align_units;
    dst->sub_time += src->sub_time;
    dst->sub_units += src->sub_units;
    dst->full_time += src->full_time;
    dst->full_units += src->full_units;
}

static void
MCQuantizeNext(
    MCCube *cubes,
    size_t size,
    mc_byte_t level,
    mp_workspace_t *ws,
    mc_time_t *time
) {
    if (level == 0) { return; }

    size_t offset = size >> 1;
    MCSplit(&cubes[0], &cubes[offset], ws, time);
    MCShrinkCube(&cubes[0], time);
    MCShrinkCube(&cubes[offset], time);
    MCQuantizeNext(&cubes[0], offset, level - 1, ws, time);
    MCQuantizeNext(&cubes[offset], size - offset, level - 1, ws, time);
}

static void
ParallelMCQuantizeNext(
    MCCube *cubes,
    size_t size,
    mc_byte_t level,
    mp_workspace_t *ws,
    mc_time_t *time
) {
    if (level == 0) { return; }

    mc_time_t t1, t2;
    MCTimeInit(&t1);
    MCTimeInit(&t2);
    size_t offset = size >> 1;
    MCSplit(&cubes[0], &cubes[offset], ws, time);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mp_workspace_t local_ws = {
                .counts = &ws->counts[0]
            };

            MCShrinkCube(&cubes[0], &t1);
            ParallelMCQuantizeNext(&cubes[0], offset, level - 1, &local_ws, &t1);
        }
        #pragma omp section
        {
            mp_workspace_t local_ws = {
                .counts = &ws->counts[(cubes[offset].size / 32)]
            };

            MCShrinkCube(&cubes[offset], &t2);
            ParallelMCQuantizeNext(&cubes[offset], size - offset, level - 1, &local_ws, &t2);
        }
    }

    MCTimeAdd(time, &t1);
    MCTimeAdd(time, &t2);
}

DTPalette *
MCQuantizeData(
    SplitImage *img,
    MCWorkspace *ws,
    mc_time_t *time
) {
    assert(img);
    assert(ws);

    // This seems to be around the point where creating more than one thread
    // has a real performance benefit. Images smaller than this are eaten
    // alive by the cost of creating threads.
    const size_t parallel_threshold = 2000000;

    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);

    size_t size = img->w * img->h;

    /* first cube */
    ws->cubes[0] = (MCCube) {
       .r = img->r,
       .g = img->g,
       .b = img->b,
       .size = size
    };

    MCShrinkCube(&ws->cubes[0], time);
    if (size >= parallel_threshold) {
        ParallelMCQuantizeNext(ws->cubes, ws->palette->size, ws->level, &ws->mp, time);
    } else {
        MCQuantizeNext(ws->cubes, ws->palette->size, ws->level, &ws->mp, time);
    }

    /* find final cube averages */
    for (size_t i = 0; i < ws->palette->size; i++) {
        ws->palette->colors[i] = MCCubeAverage(&ws->cubes[i]);
    }

    DTPalette *ret = ws->palette;
    ws->palette = NULL;

    TIMESTAMP(ts2);
    time->mc_time += (ts2 - ts1);
    time->mc_units += size;

    return ret;
}

void
MCTimeInit(
    mc_time_t *time
) {
    assert(time);
    *time = (mc_time_t) { .part_time = 0, .shrink_time = 0 };
}

void
MCTimeReport(
    mc_time_t *time
) {
    const double split_theoretical = (32.0/28.0);
    const double sub_theoretical = (32.0/10.0);
    const double full_theoretical = (32.0/11.0);
    const double part_theoretical = (32.0/21.0);
    const double shrink_theoretical = (32.0/3.0);

    double mc_time = TIME_NORM(0, time->mc_time);
    double mc_pix = ((double)time->mc_units) / mc_time;
    double mc_peak = (((((double)time->part_units) / part_theoretical) + (((double)time->shrink_units) / shrink_theoretical)) / mc_time) * 100;

    double split_time = TIME_NORM(0, time->split_time);
    double split_pix = ((double)time->split_units) / split_time;
    double split_peak = (split_pix / split_theoretical) * 100;

    double mid_time = TIME_NORM(0, time->mid_time);
    double mid_pix = ((double)time->mid_units) / mid_time;
    double mid_peak = ((((double)time->part_units) / part_theoretical) / mid_time) * 100;

    double part_time = TIME_NORM(0, time->part_time);
    double part_pix = ((double)time->part_units) / part_time;
    double part_peak = (part_pix / part_theoretical) * 100;

    double align_time = TIME_NORM(0, time->align_time);
    double align_pix = ((double)time->align_units) / align_time;
    double align_peak = (align_pix / part_theoretical) * 100;

    double full_time = TIME_NORM(0, time->full_time);
    double full_pix = ((double)time->full_units) / full_time;
    double full_peak = (full_pix / full_theoretical) * 100;

    double sub_time = TIME_NORM(0, time->sub_time);
    double sub_pix = ((double)time->sub_units) / sub_time;
    double sub_peak = (sub_pix / sub_theoretical) * 100;

    double shrink_time = TIME_NORM(0, time->shrink_time);
    double shrink_pix = ((double)time->shrink_units) / shrink_time;
    double shrink_peak = (shrink_pix / shrink_theoretical) * 100;

    printf("Kernel%19sCycles%14sPix/cyc%13s%%Peak\n", "", "", "");
    printf("MCQuantization%11s%-20.6lf%-20.6lf%.2lf%%\n", "", mc_time, mc_pix, mc_peak);
    printf(" Split%19s%-20.6lf%-20.6lf%.2lf%%\n", "", split_time, split_pix, split_peak);
    printf(" Median Partition%8s%-20.6lf%-20.6lf%.2lf%%\n", "", mid_time, mid_pix, mid_peak);
    printf("  Partition%14s%-20.6lf%-20.6lf%.2lf%%\n", "", part_time, part_pix, part_peak);
    printf("   Align Partition%7s%-20.6lf%-20.6lf%.2lf%%\n", "", align_time, align_pix, align_peak);
    printf("    Align Full-Partition%1s%-20.6lf%-20.6lf%.2lf%%\n", "", full_time, full_pix, full_peak);
    printf("    Align Sub-Partition%2s%-20.6lf%-20.6lf%.2lf%%\n", "", sub_time, sub_pix, sub_peak);
    printf(" Shrink%18s%-20.6lf%-20.6lf%.2lf%%\n", "", shrink_time, shrink_pix, shrink_peak);
}
