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
#include <assert.h>
#include <UtilMacro.h>
#include <immintrin.h>
#include <XMalloc.h>

DTPalettePacked *
StandardPaletteBW(size_t size)
{
    if (size < 2) return NULL;

    DTPalettePacked *palette = malloc(sizeof(DTPalettePacked));
    palette->size = size;
    palette->colors = XMalloc(size*sizeof(int)*3);

    float step = 255.0f / (size - 1);
    for (size_t i = 0; i < size; i++) {
        palette->colors[i] = (byte) (float) roundf(i*step);
        palette->colors[palette->size+i] = (byte) (float) roundf(i*step);
        palette->colors[palette->size*2+i] = (byte) (float) roundf(i*step);
    }
    return palette;
}

DTPalettePacked *
StandardPaletteRGB()
{
    DTPalettePacked *palette = malloc(sizeof(DTPalettePacked));
    palette->size = 8;
    palette->colors = XMalloc(palette->size*sizeof(int)*3);

    palette->colors[0] = 0xFF;
    palette->colors[1] = 0x00;
    palette->colors[2] = 0x00;
    palette->colors[3] = 0x00;
    palette->colors[4] = 0xFF;
    palette->colors[5] = 0xFF;
    palette->colors[6] = 0x00;
    palette->colors[7] = 0xFF;

    palette->colors[8] = 0x00;
    palette->colors[9] = 0xFF;
    palette->colors[10] = 0x00;
    palette->colors[11] = 0xFF;
    palette->colors[12] = 0x00;
    palette->colors[13] = 0xFF;
    palette->colors[14] = 0x00;
    palette->colors[15] = 0xFF;

    palette->colors[16] = 0x00;
    palette->colors[17] = 0x00;
    palette->colors[18] = 0xFF;
    palette->colors[19] = 0xFF;
    palette->colors[20] = 0xFF;
    palette->colors[21] = 0x00;
    palette->colors[22] = 0x00;
    palette->colors[23] = 0xFF;

    return palette;
}

DTPixel
FindClosestColorFromPalette(DTPixel needle, DTPalettePacked *palette, palette_time_t *time)
{
    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);
    // indices on the current iteration
    __m256i curr_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // the current minimum for each slice
    __m256i min_val = _mm256_set1_epi32(255*255*3+1);
    // index (argmin) for each slice
    __m256i min_idx = _mm256_setzero_si256();
    // const register for index increase
    const __m256i eight = _mm256_set1_epi32(8);
    // broadcast needle channels across individual registers
    const __m256i needle_r = _mm256_set1_epi32(needle.r);
    const __m256i needle_g = _mm256_set1_epi32(needle.g);
    const __m256i needle_b = _mm256_set1_epi32(needle.b);

    __m256i curr_r, curr_g, curr_b, dist, curr_r2, curr_g2, curr_b2, dist2, mask;
    for (size_t i = 0; i < palette->size; i += 16) {
        // load next 16 palette colors
        curr_r = _mm256_load_si256((__m256i*)&palette->colors[i]);
        curr_g = _mm256_load_si256((__m256i*)&palette->colors[palette->size+i]);
        curr_b = _mm256_load_si256((__m256i*)&palette->colors[palette->size*2+i]);
        curr_r2 = _mm256_load_si256((__m256i*)&palette->colors[i+8]);
        curr_g2 = _mm256_load_si256((__m256i*)&palette->colors[palette->size+i+8]);
        curr_b2 = _mm256_load_si256((__m256i*)&palette->colors[palette->size*2+i+8]);
        // subtract difference
        curr_r = _mm256_sub_epi32(needle_r, curr_r);
        curr_g = _mm256_sub_epi32(needle_g, curr_g);
        curr_b = _mm256_sub_epi32(needle_b, curr_b);
        curr_r2 = _mm256_sub_epi32(needle_r, curr_r2);
        curr_g2 = _mm256_sub_epi32(needle_g, curr_g2);
        curr_b2 = _mm256_sub_epi32(needle_b, curr_b2);
        // square difference
        curr_r = _mm256_mullo_epi32(curr_r, curr_r);
        curr_g = _mm256_mullo_epi32(curr_g, curr_g);
        curr_b = _mm256_mullo_epi32(curr_b, curr_b);
        curr_r2 = _mm256_mullo_epi32(curr_r2, curr_r2);
        curr_g2 = _mm256_mullo_epi32(curr_g2, curr_g2);
        curr_b2 = _mm256_mullo_epi32(curr_b2, curr_b2);
        // add squares of differences
        dist = _mm256_add_epi32(curr_r, curr_g);
        dist = _mm256_add_epi32(dist, curr_b);
        dist2 = _mm256_add_epi32(curr_r2, curr_g2);
        dist2 = _mm256_add_epi32(dist2, curr_b2);
        // find the slices where the minimum is updated
        mask = _mm256_cmpgt_epi32(min_val, dist);
        // update the indices
        min_idx = _mm256_blendv_epi8(min_idx, curr_idx, mask);
        // update the minimum (could use a "blend" here, but min is faster)
        min_val = _mm256_min_epi32(dist, min_val);
        // update the current indices
        curr_idx = _mm256_add_epi32(curr_idx, eight);
        // find the slices where the minimum is updated
        mask = _mm256_cmpgt_epi32(min_val, dist2);
        // update the indices
        min_idx = _mm256_blendv_epi8(min_idx, curr_idx, mask);
        // update the minimum (could use a "blend" here, but min is faster)
        min_val = _mm256_min_epi32(dist2, min_val);
        // update the current indices
        curr_idx = _mm256_add_epi32(curr_idx, eight);
    }

    // find the argmin in the "min" register and return its real index
    int min[8], idx[8];
    _mm256_storeu_si256((__m256i*)min, min_val);
    _mm256_storeu_si256((__m256i*)idx, min_idx);

    int k = 0, m = min[0];
    for (size_t i = 1; i < 8; i++) {
        if (min[i] < m) {
            m = min[k = (int)i];
        }
    }

    // return the pixel to original DTPixel format (rbg)
    DTPixel ret = {
        (byte)palette->colors[idx[k]],
        (byte)palette->colors[palette->size+(size_t)idx[k]],
        (byte)palette->colors[palette->size*2+(size_t)idx[k]]
    };

    TIMESTAMP(ts2);
    time->search_time += (ts2 - ts1);
    time->search_units += palette->size*3;

    return ret;
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

    printf("Palette Search%11s%-20.6lf%-20.6lf%.2lf%%\n", "", search_time, search_pix, search_peak);
}