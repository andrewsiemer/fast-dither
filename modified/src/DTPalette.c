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
#include <immintrin.h>
#include <XMalloc.h>

DTPalettePacked *
StandardPaletteBW(size_t size)
{
    if (size < 2) return NULL;

    DTPalettePacked *palette = malloc(sizeof(DTPalettePacked));
    palette->size = size;
    palette->rgb = XMalloc(size*sizeof(int)*3);

    float step = 255.0f / (size - 1);
    for (size_t i = 0; i < size; i++) {
        palette->rgb[i] = (byte) (float) roundf(i*step);
        palette->rgb[palette->size+i] = (byte) (float) roundf(i*step);
        palette->rgb[palette->size*2+i] = (byte) (float) roundf(i*step);
    }
    return palette;
}

DTPalettePacked *
StandardPaletteRGB()
{
    DTPalettePacked *palette = malloc(sizeof(DTPalettePacked));
    palette->size = 8;
    palette->rgb = XMalloc(palette->size*sizeof(int)*3);

    palette->rgb[0] = 0xFF;
    palette->rgb[1] = 0x00;
    palette->rgb[2] = 0x00;
    palette->rgb[3] = 0x00;
    palette->rgb[4] = 0xFF;
    palette->rgb[5] = 0xFF;
    palette->rgb[6] = 0x00;
    palette->rgb[7] = 0xFF;

    palette->rgb[8] = 0x00;
    palette->rgb[9] = 0xFF;
    palette->rgb[10] = 0x00;
    palette->rgb[11] = 0xFF;
    palette->rgb[12] = 0x00;
    palette->rgb[13] = 0xFF;
    palette->rgb[14] = 0x00;
    palette->rgb[15] = 0xFF;

    palette->rgb[16] = 0x00;
    palette->rgb[17] = 0x00;
    palette->rgb[18] = 0xFF;
    palette->rgb[19] = 0xFF;
    palette->rgb[20] = 0xFF;
    palette->rgb[21] = 0x00;
    palette->rgb[22] = 0x00;
    palette->rgb[23] = 0xFF;

    return palette;
}

DTPixel
FindClosestColorFromPalette(DTPixel needle, DTPalettePacked *palette)
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
        curr_r = _mm256_loadu_si256((__m256i*)&palette->rgb[i]);
        curr_g = _mm256_loadu_si256((__m256i*)&palette->rgb[palette->size+i]);
        curr_b = _mm256_loadu_si256((__m256i*)&palette->rgb[palette->size*2+i]);
        curr_r2 = _mm256_loadu_si256((__m256i*)&palette->rgb[i+8]);
        curr_g2 = _mm256_loadu_si256((__m256i*)&palette->rgb[palette->size+i+8]);
        curr_b2 = _mm256_loadu_si256((__m256i*)&palette->rgb[palette->size*2+i+8]);
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
        // could use a "blend" here, but add is faster

        // find the slices where the minimum is updated
        mask = _mm256_cmpgt_epi32(min_val, dist2);
        // update the indices
        min_idx = _mm256_blendv_epi8(min_idx, curr_idx, mask);
        // update the minimum (could use a "blend" here, but min is faster)
        min_val = _mm256_min_epi32(dist2, min_val);
        // update the current indices
        curr_idx = _mm256_add_epi32(curr_idx, eight);
        // could use a "blend" here, but add is faster
    }

    // find the argmin in the "min" register and return its real index
    int min[8], idx[8];
    _mm256_storeu_si256((__m256i*)min, min_val);
    _mm256_storeu_si256((__m256i*)idx, min_idx);

    int k = 0, m = min[0];
    for (int i = 1; i < 8; i++) {
        if (min[i] < m) {
            m = min[k = i];
        }
    }

    DTPixel ret = {palette->rgb[idx[k]], palette->rgb[palette->size+idx[k]], palette->rgb[palette->size*2+idx[k]]};

    TIMESTAMP(ts2);
    TIME_REPORT("PaletteSearch", ts1, ts2);

    return ret;
}
