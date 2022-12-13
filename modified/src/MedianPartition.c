/**
 * @file MedianPartition.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Exposes a median partitioning function for multi-channel arrays.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>

#include <MedianPartition.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#include <immintrin.h>

#include <UtilMacro.h>
#include <XMalloc.h>
#include <CompilerGoop.h>

/*** Special look-up table for partition. Only used in this file. ***/
#include <sort_lut.h>

/// @brief Used to adjust a shuffle vector which operates on 8-byte subvectors.
__attribute__((aligned (32))) static const uint8_t shuffle_adjust[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8
};

/// @brief Used to reverse the 8-bit elements in a register.
__attribute__((aligned (32))) static const uint8_t shuffle_reverse[32] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
};

/// @brief Used to adjust the signed cmp_epi8 to unsigned.
__attribute__((aligned (32))) static const uint8_t cmp_adjust[32] = {
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128
};

/// @brief Load mask setting the upper-low half of the vector.
align(32) static const uint8_t quarter_mask[32] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    255, 255, 255, 255, 255, 255, 255, 255,
    0, 0, 0, 0, 0, 0, 0, 0
};

/// @brief Load mask setting the upper half of the vector.
align(32) static const uint8_t half_mask[32] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255
};

/**
 * @brief Shuffle vectors to sort two 8-element sorted vectors into a 16-element
 *        sorted vector.
 *
 * indexed by the number of high-elements in the lower 8-element vector.
 */
__attribute__((aligned (16))) static const uint8_t sort1b_2x16[9][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 7 },
    { 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7 },
    { 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 5, 6, 7 },
    { 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7 },
    { 0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 3, 4, 5, 6, 7 },
    { 0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7 },
    { 0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7 },
    { 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 }
};

/** @brief A shuffle mask, rolled right by the index value many bytes. */
__attribute__((aligned (32))) static const uint8_t rrl_shuffle[16][32] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 },
    { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1,
      2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1 },
    { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2,
      3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2 },
    { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3,
      4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3 },
    { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4 },
    { 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5,
      6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5 },
    { 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 },
    { 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 },
    { 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8 },
    { 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
    { 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    { 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },
    { 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
    { 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
      14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
    { 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
      15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }
};

/** @brief Rolls only the top half of the vector. */
__attribute__((aligned (32))) static const uint8_t rrl_shuffle_hi[16][32] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }
};

/**
 * @brief Gathers 4 64-bit elements into an m256i.
 *
 * This used to be a load into a uint64_t array and then a load from that
 * array into an m256i, which Clang happily transformed into something like
 * this code. Unfortunately, GCC is not smart enough to make that optimization.
 * Even more unfortunately, neither GCC nor Clang are smart enough to realize
 * that xmmN = ymmN, so we just pick some and then warn the compiler that we
 * smashed them.
 */
#define GATHER64_SI256(i0, i1, i2, i3)\
({\
    register __m256i _ret, _tmp;\
    __asm__ (\
        "vmovdqa %6, %%ymm0\n\t"\
        "vpmaskmovq %4, %%ymm0, %0\n\t"\
        "vpmaskmovq %5, %%ymm0, %1\n\t"\
        "vmovq %2, %%xmm0\n\t"\
        "vpor %%ymm0, %0, %0\n\t"\
        "vmovq %3, %%xmm0\n\t"\
        "vpor %%ymm0, %1, %1\n\t"\
        "vpunpcklqdq %1, %0, %0"\
        : "=x" (_ret),\
          "=x" (_tmp)\
        : "m" (*i0),\
          "m" (*i1),\
          "m" (*((i2) - 2)),\
          "m" (*((i3) - 2)),\
          "m" (* (__m256i*) quarter_mask)\
        : "xmm0"\
    );\
    _ret;\
})

/** @brief Gathers 2 128-bit elements into an m256i. */
#define GATHER128_SI256(i0, i1)\
({\
    register __m256i _ret;\
    __asm__ (\
        "vmovdqa %3, %0\n\t"\
        "movdqa %1, %%xmm0\n\t"\
        "vpmaskmovq %2, %0, %0\n\t"\
        "vpor %%ymm0, %0, %0\n\t"\
        : "=x" (_ret)\
        : "m" (*i0),\
          "m" (* (__m256i*) ((i1) - 1)),\
          "m" (* (__m256i*) half_mask)\
        : "xmm0"\
    );\
    _ret;\
})

/**
 * @brief Sub-partitions the given arrays.
 * @param ws The MedianPartition workspace for this partition.
 * @param ch1 The first channel to partition across.
 * @param ch2 The second channel to partition across.
 * @param ch3 The third channel to partition across.
 * @param size The size of each channel.
 */
static void
AlignSubPartition32(
    mp_workspace_t *ws,
    __m256i *ch1,
    __m256i *ch2,
    __m256i *ch3,
    size_t size,
    uint8_t pivot
) {
    register __m256i a1, pa1, ppa1, ppa2, ppa3, adjust, pivots;
    register __m256i sort_mask, p_sort_mask, tmp1, tmp2;
    register __m256i a1_lo, a2_lo, a3_lo;
    register uint32_t am, pam;
    register uint32_t loc, lloc, hloc, ploc, plloc, phloc, pploc;

    adjust = _mm256_load_si256((__m256i*) cmp_adjust);
    pivots = _mm256_set1_epi8(pivot);
    pivots = _mm256_add_epi8(adjust, pivots);

    a1 = _mm256_load_si256(&ch1[0]);
    tmp1 = _mm256_add_epi8(a1, adjust);
    tmp1 = _mm256_cmpgt_epi8(tmp1, pivots);
    am = (unsigned int) _mm256_movemask_epi8(tmp1);

    lloc = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFF);
    hloc = (uint8_t) (unsigned int) __builtin_popcount((am >> 16) & 0xFF);
    loc  = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFFFF);
    ws->counts[0] = (uint8_t) (unsigned int) __builtin_popcount(am);

    pa1 = a1;
    pam = am;
    ploc = loc;
    plloc = lloc;
    phloc = hloc;

    if (size > 1) {
        sort_mask = _mm256_load_si256((__m256i*) shuffle_adjust);

        a1 = _mm256_load_si256(&ch1[1]);
        tmp1 = _mm256_add_epi8(a1, adjust);
        tmp1 = _mm256_cmpgt_epi8(tmp1, pivots);
        am = (unsigned int) _mm256_movemask_epi8(tmp1);

        lloc = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFF);
        hloc = (uint8_t) (unsigned int) __builtin_popcount((am >> 16) & 0xFF);
        loc  = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFFFF);
        ws->counts[1] = (uint8_t) (unsigned int) __builtin_popcount(am);

        // Load in the 8-element sort vector, and the shuffle vector for
        // 32-element sorting.
        tmp1 = GATHER64_SI256(
            (uint64_t*) sort1b_4x8[(pam >> 0) & 0xFF],
            (uint64_t*) sort1b_4x8[(pam >> 8) & 0xFF],
            (uint64_t*) sort1b_4x8[(pam >> 16) & 0xFF],
            (uint64_t*) sort1b_4x8[(pam >> 24) & 0xFF]
        );
        sort_mask = _mm256_add_epi8(sort_mask, tmp1);

        // Load in the 16-element sort vectors pieces.
        tmp2 = _mm256_load_si256((__m256i*) rrl_shuffle_hi[ploc & 0xF]);
        tmp1 = GATHER128_SI256(
            (__m128i*) sort1b_2x16[plloc],
            (__m128i*) sort1b_2x16[phloc]
        );

        // Blend the 8/16 sort evcotrs together, then preroll the resulting
        // shuffle vector to create a vector that will 16-sort and pre-roll
        // the a vectors.
        sort_mask = _mm256_shuffle_epi8(sort_mask, tmp1);
        sort_mask = _mm256_shuffle_epi8(sort_mask, tmp2);

        p_sort_mask = sort_mask;
        ppa1 = pa1;
        pploc = ploc;
        pa1 = a1;
        pam = am;
        ploc = loc;
        plloc = lloc;
        phloc = hloc;

        for (size_t i = 2; i < size; i++) {
            a1 = _mm256_load_si256(&ch1[i]);
            sort_mask = _mm256_load_si256((__m256i*) shuffle_adjust);
            ppa2 = _mm256_load_si256(&ch2[i-2]);
            ppa3 = _mm256_load_si256(&ch3[i-2]);

            tmp1 = _mm256_add_epi8(a1, adjust);
            tmp1 = _mm256_cmpgt_epi8(tmp1, pivots);
            am = (unsigned int) _mm256_movemask_epi8(tmp1);

            lloc = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFF);
            hloc = (uint8_t) (unsigned int) __builtin_popcount((am >> 16) & 0xFF);
            loc  = (uint8_t) (unsigned int) __builtin_popcount(am & 0xFFFF);
            ws->counts[i] = (uint8_t) (unsigned int) __builtin_popcount(am);

            /******** ITERATION SPLIT ********/

            // Load in the 8-element sort vector, and the shuffle vector for
            // 32-element sorting.
            tmp1 = GATHER64_SI256(
                (uint64_t*) sort1b_4x8[(pam >> 0) & 0xFF],
                (uint64_t*) sort1b_4x8[(pam >> 8) & 0xFF],
                (uint64_t*) sort1b_4x8[(pam >> 16) & 0xFF],
                (uint64_t*) sort1b_4x8[(pam >> 24) & 0xFF]
            );
            sort_mask = _mm256_add_epi8(sort_mask, tmp1);

            // Load in the 16-element sort vectors pieces.
            tmp1 = GATHER128_SI256(
                (__m128i*) sort1b_2x16[plloc],
                (__m128i*) sort1b_2x16[phloc]
            );
            tmp2 = _mm256_load_si256((__m256i*) rrl_shuffle_hi[ploc & 0xF]);

            // Blend the 8/16 sort evcotrs together, then preroll the resulting
            // shuffle vector to create a vector that will 16-sort and pre-roll
            // the a vectors.
            sort_mask = _mm256_shuffle_epi8(sort_mask, tmp1);
            sort_mask = _mm256_shuffle_epi8(sort_mask, tmp2);

            /******** ITERATION SPLIT ********/

            tmp1 = _mm256_load_si256((__m256i*) srl_blend[pploc]);

            // Apply the shuffle to the a vectors.
            ppa1 = _mm256_shuffle_epi8(ppa1, p_sort_mask);
            ppa2 = _mm256_shuffle_epi8(ppa2, p_sort_mask);
            ppa3 = _mm256_shuffle_epi8(ppa3, p_sort_mask);

            // Reverse each channel to allow it to be blended. The rolling of the
            // high channel was already done. Note that the srl_blend vector has the
            // low half of each SiMD vector inverted in anticipation of this permute
            // time save (where we simply reverse instead of creating a
            // high-high/low-low vector).
            a1_lo = _mm256_permute2x128_si256(ppa1, ppa1, 0x01);
            a2_lo = _mm256_permute2x128_si256(ppa2, ppa2, 0x01);
            a3_lo = _mm256_permute2x128_si256(ppa3, ppa3, 0x01);

            // Use the mask and rolled high-high vector to insert the upper half
            // of each vector in between the low halfs low and high values.
            ppa1 = _mm256_and_si256(tmp1, ppa1);
            ppa2 = _mm256_and_si256(tmp1, ppa2);
            ppa3 = _mm256_and_si256(tmp1, ppa3);
            a1_lo = _mm256_andnot_si256(tmp1, a1_lo);
            a2_lo = _mm256_andnot_si256(tmp1, a2_lo);
            a3_lo = _mm256_andnot_si256(tmp1, a3_lo);
            ppa1 = _mm256_or_si256(ppa1, a1_lo);
            ppa2 = _mm256_or_si256(ppa2, a2_lo);
            ppa3 = _mm256_or_si256(ppa3, a3_lo);

            _mm256_store_si256(&ch1[i-2], ppa1);
            _mm256_store_si256(&ch2[i-2], ppa2);
            _mm256_store_si256(&ch3[i-2], ppa3);

            p_sort_mask = sort_mask;
            ppa1 = pa1;
            pploc = ploc;
            pa1 = a1;
            pam = am;
            ploc = loc;
            plloc = lloc;
            phloc = hloc;
        }

        ppa2 = _mm256_load_si256(&ch2[size-2]);
        ppa3 = _mm256_load_si256(&ch3[size-2]);

        // Get the mask used to blend the a vectors.
        tmp1 = _mm256_load_si256((__m256i*) srl_blend[pploc]);

        // Apply the shuffle to the a vectors.
        ppa1 = _mm256_shuffle_epi8(ppa1, p_sort_mask);
        ppa2 = _mm256_shuffle_epi8(ppa2, p_sort_mask);
        ppa3 = _mm256_shuffle_epi8(ppa3, p_sort_mask);

        // Reverse each channel to allow it to be blended. The rolling of the
        // high channel was already done. Note that the srl_blend vector has the
        // low half of each SiMD vector inverted in anticipation of this permute
        // time save (where we simply reverse instead of creating a
        // high-high/low-low vector).
        a1_lo = _mm256_permute2x128_si256(ppa1, ppa1, 0x01);
        a2_lo = _mm256_permute2x128_si256(ppa2, ppa2, 0x01);
        a3_lo = _mm256_permute2x128_si256(ppa3, ppa3, 0x01);

        // Use the mask and rolled high-high vector to insert the upper half
        // of each vector in between the low halfs low and high values.
        ppa1 = _mm256_and_si256(tmp1, ppa1);
        ppa2 = _mm256_and_si256(tmp1, ppa2);
        ppa3 = _mm256_and_si256(tmp1, ppa3);
        a1_lo = _mm256_andnot_si256(tmp1, a1_lo);
        a2_lo = _mm256_andnot_si256(tmp1, a2_lo);
        a3_lo = _mm256_andnot_si256(tmp1, a3_lo);
        ppa1 = _mm256_or_si256(ppa1, a1_lo);
        ppa2 = _mm256_or_si256(ppa2, a2_lo);
        ppa3 = _mm256_or_si256(ppa3, a3_lo);

        _mm256_store_si256(&ch1[size-2], ppa1);
        _mm256_store_si256(&ch2[size-2], ppa2);
        _mm256_store_si256(&ch3[size-2], ppa3);
    }

    sort_mask = _mm256_load_si256((__m256i*) shuffle_adjust);

    // Load in the 8-element sort vector, and the shuffle vector for
    // 32-element sorting.
    tmp1 = GATHER64_SI256(
        (uint64_t*) sort1b_4x8[(pam >> 0) & 0xFF],
        (uint64_t*) sort1b_4x8[(pam >> 8) & 0xFF],
        (uint64_t*) sort1b_4x8[(pam >> 16) & 0xFF],
        (uint64_t*) sort1b_4x8[(pam >> 24) & 0xFF]
    );
    sort_mask = _mm256_add_epi8(sort_mask, tmp1);

    // Load in the 16-element sort vectors pieces.
    tmp2 = _mm256_load_si256((__m256i*) rrl_shuffle_hi[ploc & 0xF]);
    tmp1 = GATHER128_SI256(
        (__m128i*) sort1b_2x16[plloc],
        (__m128i*) sort1b_2x16[phloc]
    );

    // Blend the 8/16 sort evcotrs together, then preroll the resulting
    // shuffle vector to create a vector that will 16-sort and pre-roll
    // the a vectors.
    sort_mask = _mm256_shuffle_epi8(sort_mask, tmp1);
    sort_mask = _mm256_shuffle_epi8(sort_mask, tmp2);

    // Apply the shuffle to the a vectors.
    p_sort_mask = sort_mask;
    ppa1 = pa1;
    pploc = ploc;

    ppa2 = _mm256_load_si256(&ch2[size-1]);
    ppa3 = _mm256_load_si256(&ch3[size-1]);

    // Get the mask used to blend the a vectors.
    tmp1 = _mm256_load_si256((__m256i*) srl_blend[pploc]);

    // Apply the shuffle to the a vectors.
    ppa1 = _mm256_shuffle_epi8(ppa1, p_sort_mask);
    ppa2 = _mm256_shuffle_epi8(ppa2, p_sort_mask);
    ppa3 = _mm256_shuffle_epi8(ppa3, p_sort_mask);

    // Reverse each channel to allow it to be blended. The rolling of the
    // high channel was already done. Note that the srl_blend vector has the
    // low half of each SiMD vector inverted in anticipation of this permute
    // time save (where we simply reverse instead of creating a
    // high-high/low-low vector).
    a1_lo = _mm256_permute2x128_si256(ppa1, ppa1, 0x01);
    a2_lo = _mm256_permute2x128_si256(ppa2, ppa2, 0x01);
    a3_lo = _mm256_permute2x128_si256(ppa3, ppa3, 0x01);

    // Use the mask and rolled high-high vector to insert the upper half
    // of each vector in between the low halfs low and high values.
    ppa1 = _mm256_and_si256(tmp1, ppa1);
    ppa2 = _mm256_and_si256(tmp1, ppa2);
    ppa3 = _mm256_and_si256(tmp1, ppa3);
    a1_lo = _mm256_andnot_si256(tmp1, a1_lo);
    a2_lo = _mm256_andnot_si256(tmp1, a2_lo);
    a3_lo = _mm256_andnot_si256(tmp1, a3_lo);
    ppa1 = _mm256_or_si256(ppa1, a1_lo);
    ppa2 = _mm256_or_si256(ppa2, a2_lo);
    ppa3 = _mm256_or_si256(ppa3, a3_lo);

    _mm256_store_si256(&ch1[size-1], ppa1);
    _mm256_store_si256(&ch2[size-1], ppa2);
    _mm256_store_si256(&ch3[size-1], ppa3);
}

/**
 * @brief Fully partitions the given sub-partitioned arrays.
 * @param ws The MedianPartition workspace for this partition.
 * @param ch1 The first channel to partition across.
 * @param ch2 The second channel to partition across.
 * @param ch3 The third channel to partition across.
 * @param size The size of each channel.
 */
static size_t
AlignFullPartition(
    mp_workspace_t *ws,
    __m256i *ch1,
    __m256i *ch2,
    __m256i *ch3,
    size_t size
) {
    register __m256i a1, a2, a3, b1, b2, b3;
    register uint32_t ac, bc;
    size_t lo = 0, hi = size - 1, next = size - 1;

    a1 = _mm256_load_si256(&ch1[0]);
    a2 = _mm256_load_si256(&ch2[0]);
    a3 = _mm256_load_si256(&ch3[0]);
    ac = ws->counts[0];

    while (lo < hi) {
        b1 = _mm256_load_si256(&ch1[next]);
        b2 = _mm256_load_si256(&ch2[next]);
        b3 = _mm256_load_si256(&ch3[next]);
        bc = ws->counts[next];

        /* Sort across both chunks. */

        // Roll the b vectors right by the high-value count of a
        {
            register __m256i b1_lo, b2_lo, b3_lo, roll_mask, shuffle_mask;

            // Generate the 32-byte roll blending mask.
            // And create the 16-byte roll shuffle mask.
            shuffle_mask = _mm256_load_si256((__m256i*) rrl_shuffle[(ac) & 0xF]);
            roll_mask = _mm256_load_si256((__m256i*) srl_blend[ac]);

            // Roll the b vectors by 16-byte rolling each one and then blending
            // with the roll mask
            b1 = _mm256_shuffle_epi8(b1, shuffle_mask);
            b2 = _mm256_shuffle_epi8(b2, shuffle_mask);
            b3 = _mm256_shuffle_epi8(b3, shuffle_mask);

            // Generate the lo/hi b vectors to be blended.
            b1_lo = _mm256_permute2x128_si256(b1, b1, 0x01);
            b2_lo = _mm256_permute2x128_si256(b2, b2, 0x01);
            b3_lo = _mm256_permute2x128_si256(b3, b3, 0x01);

            b1 = _mm256_and_si256(b1, roll_mask);
            b2 = _mm256_and_si256(b2, roll_mask);
            b3 = _mm256_and_si256(b3, roll_mask);
            b1_lo = _mm256_andnot_si256(roll_mask, b1_lo);
            b2_lo = _mm256_andnot_si256(roll_mask, b2_lo);
            b3_lo = _mm256_andnot_si256(roll_mask, b3_lo);
            b1 = _mm256_or_si256(b1, b1_lo);
            b2 = _mm256_or_si256(b2, b2_lo);
            b3 = _mm256_or_si256(b3, b3_lo);
        }

        // Use the rolled b vectors to sort across { a, b }.
        {
            register __m256i a1_tmp1, a2_tmp1, a3_tmp1, b_blend;
            register __m256i a1_tmp2, a2_tmp2, a3_tmp2;

            // Load the 64-byte blending masks.
            b_blend = _mm256_load_si256((__m256i*) shifted_set_mask[ac]);

            /* Sort the a and b vectors across each other. */
            a1_tmp1 = _mm256_andnot_si256(b_blend, b1);
            a1_tmp2 = _mm256_and_si256(b_blend, a1);
            a2_tmp1 = _mm256_andnot_si256(b_blend, b2);
            a2_tmp2 = _mm256_and_si256(b_blend, a2);
            a3_tmp1 = _mm256_andnot_si256(b_blend, b3);
            a3_tmp2 = _mm256_and_si256(b_blend, a3);
            a1_tmp1 = _mm256_or_si256(a1_tmp1, a1_tmp2);
            a2_tmp1 = _mm256_or_si256(a2_tmp1, a2_tmp2);
            a3_tmp1 = _mm256_or_si256(a3_tmp1, a3_tmp2);
            b1 = _mm256_and_si256(b_blend, b1);
            a1 = _mm256_andnot_si256(b_blend, a1);
            b2 = _mm256_and_si256(b_blend, b2);
            a2 = _mm256_andnot_si256(b_blend, a2);
            b3 = _mm256_and_si256(b_blend, b3);
            a3 = _mm256_andnot_si256(b_blend, a3);
            b1 = _mm256_or_si256(a1, b1);
            b2 = _mm256_or_si256(a2, b2);
            b3 = _mm256_or_si256(a3, b3);
            a1 = a1_tmp1;
            a2 = a2_tmp1;
            a3 = a3_tmp1;
        }

        // Update the high-value counts.
        register uint32_t hvc_tmp = MIN(32, (ac) + (bc));
        ac = (uint32_t) MAX((int32_t)0, (int32_t) (((ac) + (bc)) - 32));
        bc = hvc_tmp;

        // Determine which side is full. If ac is zero, then the a vectors are
        // low values and should be stored. Otherwise, the b vectors are all
        // high values and should be stored.
        if (ac == 0) {
            _mm256_store_si256(&ch1[lo], a1);
            _mm256_store_si256(&ch2[lo], a2);
            _mm256_store_si256(&ch3[lo], a3);
            a1 = b1;
            a2 = b2;
            a3 = b3;
            ac = bc;
            next = ++lo;
        } else {
            _mm256_store_si256(&ch1[hi], b1);
            _mm256_store_si256(&ch2[hi], b2);
            _mm256_store_si256(&ch3[hi], b3);
            next = --hi;
        }
    }

    _mm256_store_si256(&ch1[lo], a1);
    _mm256_store_si256(&ch2[lo], a2);
    _mm256_store_si256(&ch3[lo], a3);

    return lo;
}

/**
 * @brief Performs a partition on the given arrays where the arrays are 32
 *        byte aligned and the size is a multiple of 32.
 * @param ch1 The first array to partition, and the one to be compared against
 *            the pivot.
 * @param ch2 The second array to partition, moved the same as ch1.
 * @param ch3 The third array to be partitioned, moved the same as ch1.
 * @param size The size of each channel, in 32-byte chunks.
 * @param pivot The value to pivot across.
 * @return The index of the vector containing the pivot crossing.
 */
static size_t
AlignPartition(
    mp_workspace_t *ws,
    __m256i *ch1,
    __m256i *ch2,
    __m256i *ch3,
    size_t size,
    uint8_t pivot,
    mc_time_t *time
) {
    assert(size > 0);

    // Partition each 32-element sub-group.
    MC_TIME(
        time->sub,
        size * 32,
        AlignSubPartition32(ws, ch1, ch2, ch3, size, pivot);
    );

    // Partition the full array.
    size_t ret;
    MC_TIME(
        time->full,
        size * 32,
        ret = AlignFullPartition(ws, ch1, ch2, ch3, size);
    );

    return ret;
}

/**
 * @brief Partitions a single 1X32 group.
 * @param ch1 The first channel group.
 * @param ch2 The second channel group.
 * @param ch3 The third channel group.
 * @param pivot The pivot to partition across.
 * @param count Returns the high-value count for the group.
 */
static void
SinglePartition1X32(
    __m256i *ch1,
    __m256i *ch2,
    __m256i *ch3,
    uint8_t pivot,
    uint32_t *count
) {
    mp_workspace_t ws = { .counts = count };
    __m256i ch[3] = {
        _mm256_loadu_si256(ch1),
        _mm256_loadu_si256(ch2),
        _mm256_loadu_si256(ch3)
    };

    AlignSubPartition32(&ws, &ch[0], &ch[1], &ch[2], 1, pivot);

    _mm256_storeu_si256(ch1, ch[0]);
    _mm256_storeu_si256(ch2, ch[1]);
    _mm256_storeu_si256(ch3, ch[2]);
}

static size_t
Partition(
    mp_workspace_t *ws,
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    uint8_t pivot,
    mc_time_t *time
) {
    // Get the offsets necessary to align the size and arrays to a 32-byte
    // bound for the aligned partition function.
    size_t pre_align = 32 - (((uintptr_t) ch1) & 0x1F);
    pre_align = (pre_align == 32) ? 0 : pre_align;
    pre_align = MIN(pre_align, size);
    size_t post_align = (size - pre_align) % 32;

    // Perform the aligned partition, if possible.
    size_t bound = size;
    if ((pre_align + post_align) < size) {
        size_t align_size = (size - (pre_align + post_align)) / 32;

        MC_TIME(
            time->align,
            align_size * 32,
            bound = AlignPartition(
                ws,
                (__m256i*) &ch1[pre_align],
                (__m256i*) &ch2[pre_align],
                (__m256i*) &ch3[pre_align],
                align_size,
                pivot,
                time
            );
        );

        // Find the actual bound, now that we're mostly sorted.
        bound = (bound * 32) + pre_align;
        while ((bound < size) && (ch1[bound] <= pivot)) bound++;

        // Sort the not-aligned pre parts.
        __m256i u1, u2, u3;
        register __m256i t1, t2, t3;
        size_t target = (size_t) MAX(0, ((ptrdiff_t)bound) - 32);
        u1 = _mm256_loadu_si256((__m256i*) &ch1[0]);
        u2 = _mm256_loadu_si256((__m256i*) &ch2[0]);
        u3 = _mm256_loadu_si256((__m256i*) &ch3[0]);
        t1 = _mm256_loadu_si256((__m256i*) &ch1[target]);
        t2 = _mm256_loadu_si256((__m256i*) &ch2[target]);
        t3 = _mm256_loadu_si256((__m256i*) &ch3[target]);

        uint32_t count;
        SinglePartition1X32(&u1, &u2, &u3, pivot, &count);
        bound = (size_t) MAX(32 - ((ptrdiff_t)count), (ptrdiff_t) (bound - count));
        do {
            register __m256i tmp = _mm256_load_si256((__m256i*) shuffle_reverse);
            t1 = _mm256_shuffle_epi8(t1, tmp);
            t2 = _mm256_shuffle_epi8(t2, tmp);
            t3 = _mm256_shuffle_epi8(t3, tmp);
            t1 = _mm256_permute2x128_si256(t1, t1, 0x01);
            t2 = _mm256_permute2x128_si256(t2, t2, 0x01);
            t3 = _mm256_permute2x128_si256(t3, t3, 0x01);
        } while (0);

        _mm256_storeu_si256((__m256i*) &ch1[0], t1);
        _mm256_storeu_si256((__m256i*) &ch2[0], t2);
        _mm256_storeu_si256((__m256i*) &ch3[0], t3);
        _mm256_storeu_si256((__m256i*) &ch1[target], u1);
        _mm256_storeu_si256((__m256i*) &ch2[target], u2);
        _mm256_storeu_si256((__m256i*) &ch3[target], u3);

        // Sort the not-aligned post parts.
        size_t base = size - 32;
        target = MIN(base, bound);
        u1 = _mm256_loadu_si256((__m256i*) &ch1[base]);
        u2 = _mm256_loadu_si256((__m256i*) &ch2[base]);
        u3 = _mm256_loadu_si256((__m256i*) &ch3[base]);
        t1 = _mm256_loadu_si256((__m256i*) &ch1[target]);
        t2 = _mm256_loadu_si256((__m256i*) &ch2[target]);
        t3 = _mm256_loadu_si256((__m256i*) &ch3[target]);

        SinglePartition1X32(&u1, &u2, &u3, pivot, &count);
        bound = MIN(size - count, bound + (32 - count));
        do {
            register __m256i tmp = _mm256_load_si256((__m256i*) shuffle_reverse);
            t1 = _mm256_shuffle_epi8(t1, tmp);
            t2 = _mm256_shuffle_epi8(t2, tmp);
            t3 = _mm256_shuffle_epi8(t3, tmp);
            t1 = _mm256_permute2x128_si256(t1, t1, 0x01);
            t2 = _mm256_permute2x128_si256(t2, t2, 0x01);
            t3 = _mm256_permute2x128_si256(t3, t3, 0x01);
        } while (0);

        _mm256_storeu_si256((__m256i*) &ch1[base], t1);
        _mm256_storeu_si256((__m256i*) &ch2[base], t2);
        _mm256_storeu_si256((__m256i*) &ch3[base], t3);
        _mm256_storeu_si256((__m256i*) &ch1[target], u1);
        _mm256_storeu_si256((__m256i*) &ch2[target], u2);
        _mm256_storeu_si256((__m256i*) &ch3[target], u3);
    } else {
        size_t lo = 0, hi = size;
        while (lo < hi) {
            if (ch1[lo] > pivot) {
                hi--;
                SWAP(ch1[lo], ch1[hi]);
                SWAP(ch2[lo], ch2[hi]);
                SWAP(ch3[lo], ch3[hi]);
            } else {
                lo++;
            }
        }

        bound = hi;
    }

#if 0
    // Verify that the array was partitioned correctly.
    for (size_t i = 0; i < size; i++) {
        if (!(((i < bound) && (ch1[i] <= pivot)) ||
               ((i >= bound) && (ch1[i] > pivot)))) {
            fprintf(stderr, "Bad value %u at %zu for pivot = %u, size = %zu,"
                    " bound = %zu, pre = %zu, post = %zu\n",
                    ch1[i], i, pivot, size, bound, pre_align, post_align);
            abort();
        }
    }
#endif

    return bound;
}

/**
 * @brief Selects the kth sorted element in the given array.
 * @param ch1 The channel to search for the kth element of.
 * @param ch2 The first channel to arg-partition with ch1.
 * @param ch3 The second channel to arg-partition with ch2.
 * @param size The size of the array.
 * @param k The k index to search for.
 * @param time The time measured for partition.
 * @return The median for channel 1.
 */
static uint8_t
QSelect(
    mp_workspace_t *ws,
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    size_t k,
    mc_time_t *time
) {
    assert(ch1);
    assert(ch2);
    assert(ch3);
    assert(size > 0);
    assert(k < size);

    // Bound the values of pivots we can choose to ensure termination, since
    // partition may not actually divide the array if, e.g., all the elements
    // are the same.
    uint8_t min_pivot = 0;
    uint8_t max_pivot = (uint8_t) ~0u;

    while ((size > 1) && (min_pivot <= max_pivot)) {
        // Get our pivot. Random is "good enough" for O(n) in most cases.
        size_t pivot_idx = ((size_t) (unsigned int) rand()) % size;
        uint8_t pivot = ch1[pivot_idx];
        assert(pivot >= min_pivot);
        pivot = MIN(pivot, max_pivot);

        // Partition across our pivot.
        size_t mid;
        MC_TIME(
            time->part,
            size,
            mid = Partition(ws, ch1, ch2, ch3, size, pivot, time);
        );
        assert(mid <= size);

        if (k < mid) {
            size = mid;
            max_pivot = pivot - 1;
        } else if (k >= mid) {
            assert(mid < size);
            ch1 = &ch1[mid];
            ch2 = &ch2[mid];
            ch3 = &ch3[mid];
            size -= mid;
            k -= mid;
            min_pivot = pivot + 1;
        }
    }

    return ch1[0];
}

size_t
MedianPartition(
    mp_workspace_t *ws,
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    mc_time_t *time
) {
    // Use the quickselect algorithm to find the median.
    size_t mid = size >> 1;
    uint8_t m1 = QSelect(ws, ch1, ch2, ch3, size, mid, time);

    // Partition across the median.
    size_t lo_size;
    MC_TIME(
        time->part,
        size,
        lo_size = Partition(ws, ch1, ch2, ch3, size, m1, time);
    );
    assert(lo_size > mid);

    // Partition again across (median - 1) to force all median values to the
    // middle of the array.
    uint8_t median = m1;
    if (median-- > 0) {
        size_t tmp;
        MC_TIME(
            time->part,
            lo_size,
            tmp = Partition(ws, ch1, ch2, ch3, lo_size, median, time);
        );
        assert(tmp <= mid);
    }

    assert(ch1[mid] == m1);

    return mid;
}

void
MPWorkspaceInit(
    mp_workspace_t *ws,
    size_t size
) {
    assert(ws);
    assert(size > 0);
    ws->counts = XMalloc((size / 32) * sizeof(uint32_t));
}

void
MPWorkspaceDestroy(
    mp_workspace_t *ws
) {
    assert(ws);
    XFree(ws->counts);
}
