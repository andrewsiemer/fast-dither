/**
 * @file SplitImage.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Implementation of the split image module.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>
#include <stdint.h>

#include <SplitImage.h>

#include <immintrin.h>

#include <MCQuantization.h>
#include <XMalloc.h>
#include <CompilerGoop.h>

/// @brief Documentation is a liberal myth.
align(32) static const uint8_t shuffle_mask[32] = {
    0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14,
    0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14
};

/// @brief You want answers? So do I...
align(32) static const uint8_t blend_mask[3][32] = {
    { 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};

SplitImage *
CreateSplitImage(
    DTImage *img,
    mc_time_t *time
) {
    assert(img);

    SplitImage *ret = XMalloc(sizeof(SplitImage));
    ret->w = img->width;
    ret->h = img->height;
    ret->type = img->type;
    ret->resolution = img->resolution;
    ret->r = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);
    ret->g = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);
    ret->b = XMemalign(32, sizeof(uint8_t) * ret->w * ret->h);

    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);

    register __m256i r, g, b, m1, m2, m3, rev1, rev3, tmp, mtmp;
    register __m256i rgb_gbr, brg_rgb, gbr_brg, smask;
    smask = _mm256_load_si256((__m256i*) shuffle_mask);

    size_t size = ret->w * ret->h;
    size_t align_size = size - (size % 32);
    for (size_t i = 0; i < align_size; i += 32) {
        rgb_gbr = _mm256_loadu_si256(((__m256i*) &img->pixels[i]) + 0);
        brg_rgb = _mm256_loadu_si256(((__m256i*) &img->pixels[i]) + 1);
        gbr_brg = _mm256_loadu_si256(((__m256i*) &img->pixels[i]) + 2);
        m1 = _mm256_load_si256((__m256i*) blend_mask[0]);
        m2 = _mm256_load_si256((__m256i*) blend_mask[1]);
        m3 = _mm256_load_si256((__m256i*) blend_mask[2]);
        rgb_gbr = _mm256_shuffle_epi8(rgb_gbr, smask);
        brg_rgb = _mm256_shuffle_epi8(brg_rgb, smask);
        gbr_brg = _mm256_shuffle_epi8(gbr_brg, smask);
        rev1 = _mm256_permute2x128_si256(rgb_gbr, rgb_gbr, 0x01);
        rev3 = _mm256_permute2x128_si256(gbr_brg, gbr_brg, 0x01);

        /* First group */

        r = rgb_gbr;
        g = _mm256_srli_si256(rgb_gbr, 6);
        b = _mm256_srli_si256(rgb_gbr, 11);

        /* Second group */

        tmp = _mm256_srli_si256(rev1, 5);
        tmp = _mm256_and_si256(m2, tmp);
        r = _mm256_andnot_si256(m2, r);
        r = _mm256_or_si256(r, tmp);

        tmp = _mm256_slli_si256(rev1, 5);
        mtmp = _mm256_slli_si256(m1, 5);
        tmp = _mm256_and_si256(mtmp, tmp);
        g = _mm256_andnot_si256(mtmp, g);
        g = _mm256_or_si256(g, tmp);

        tmp = _mm256_srli_si256(rev1, 1);
        mtmp = _mm256_srli_si256(m2, 1);
        tmp = _mm256_and_si256(mtmp, tmp);
        b = _mm256_andnot_si256(mtmp, b);
        b = _mm256_or_si256(b, tmp);

        /* Third group */

        tmp = _mm256_slli_si256(brg_rgb, 5);
        tmp = _mm256_and_si256(m3, tmp);
        r = _mm256_andnot_si256(m3, r);
        r = _mm256_or_si256(r, tmp);

        tmp = _mm256_and_si256(m3, brg_rgb);
        g = _mm256_andnot_si256(m3, g);
        g = _mm256_or_si256(g, tmp);

        tmp = _mm256_slli_si256(brg_rgb, 10);
        mtmp = _mm256_slli_si256(m1, 10);
        tmp = _mm256_and_si256(mtmp, tmp);
        b = _mm256_andnot_si256(mtmp, b);
        b = _mm256_or_si256(b, tmp);

        /* Fix the masks */

        m1 = _mm256_permute2x128_si256(m1, m1, 0x01);
        m2 = _mm256_permute2x128_si256(m2, m2, 0x01);
        m3 = _mm256_permute2x128_si256(m3, m3, 0x01);

        /* Fourth group */

        tmp = _mm256_and_si256(m1, brg_rgb);
        r = _mm256_andnot_si256(m1, r);
        r = _mm256_or_si256(r, tmp);

        tmp = _mm256_srli_si256(brg_rgb, 6);
        mtmp = _mm256_srli_si256(m2, 6);
        tmp = _mm256_and_si256(mtmp, tmp);
        g = _mm256_andnot_si256(mtmp, g);
        g = _mm256_or_si256(tmp, g);

        tmp = _mm256_srli_si256(brg_rgb, 11);
        tmp = _mm256_and_si256(mtmp, tmp);
        b = _mm256_andnot_si256(mtmp, b);
        b = _mm256_or_si256(tmp, b);

        /* Fifth group */

        tmp = _mm256_srli_si256(rev3, 5);
        tmp = _mm256_and_si256(m2, tmp);
        r = _mm256_andnot_si256(m2, r);
        r = _mm256_or_si256(r, tmp);

        tmp = _mm256_slli_si256(rev3, 5);
        mtmp = _mm256_slli_si256(m1, 5);
        tmp = _mm256_and_si256(mtmp, tmp);
        g = _mm256_andnot_si256(mtmp, g);
        g = _mm256_or_si256(g, tmp);

        tmp = _mm256_srli_si256(rev3, 1);
        mtmp = _mm256_srli_si256(m2, 1);
        tmp = _mm256_and_si256(mtmp, tmp);
        b = _mm256_andnot_si256(mtmp, b);
        b = _mm256_or_si256(b, tmp);

        /* Sixth group */

        tmp = _mm256_slli_si256(gbr_brg, 5);
        tmp = _mm256_and_si256(m3, tmp);
        r = _mm256_andnot_si256(m3, r);
        r = _mm256_or_si256(r, tmp);

        tmp = _mm256_and_si256(m3, gbr_brg);
        g = _mm256_andnot_si256(m3, g);
        g = _mm256_or_si256(g, tmp);

        tmp = _mm256_slli_si256(gbr_brg, 10);
        mtmp = _mm256_slli_si256(m1, 10);
        tmp = _mm256_and_si256(mtmp, tmp);
        b = _mm256_andnot_si256(mtmp, b);
        b = _mm256_or_si256(b, tmp);

        /* Store it */

        _mm256_storeu_si256((__m256i*) &ret->r[i], r);
        _mm256_storeu_si256((__m256i*) &ret->g[i], g);
        _mm256_storeu_si256((__m256i*) &ret->b[i], b);
    }

    for (size_t i = align_size; i < size; i++) {
        ret->r[i] = img->pixels[i].r;
        ret->g[i] = img->pixels[i].g;
        ret->b[i] = img->pixels[i].b;
    }

    TIMESTAMP(ts2);
    time->mc_time += (ts2 - ts1);
    time->split_time += (ts2 - ts1);
    time->split_units += size;

    return ret;
}

void
DestroySplitImage(
    SplitImage *img
) {
    assert(img);
    XFree(img->r);
    XFree(img->g);
    XFree(img->b);
    XFree(img);
}
