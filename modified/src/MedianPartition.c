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

/**
 * @brief Shuffle vectors to sort two 8-element sorted vectors into a 16-element
 *        sorted vector.
 *
 * Indexed by the number of high-elements in the lower 8-element vector.
 *
 * Padded by two elements, since we unaligned load this twice to blend it
 * and create a single sorting vector.
 */
__attribute__((aligned (16))) static const uint8_t sort1b_2x16[11][16] = {
    { 0 },
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
 * @brief Sorts 16-byte chunks of a vector based on the pivot and a1.
 *
 * Note that the vectors returned by this will automaticically have their
 * high 16-byte chunk rolled by loc. This is an optimization, as this is
 * required by ARGMSORT1X32 anyway, and doing it here means less shuffles
 * are necessary overall.
 *
 * @param pivots The signed-adjusted pivot vector.
 * @param adjust The vector used to adjust a1 before the pivot comparison.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 * @param loc Returns the number of high values in the lower 16 bytes.
 * @param hic Returns the number of high values in the higher 16 bytes.
 */
#define ARGMSORT2X16(pivots, adjust, a1, a2, a3, loc, hic)\
do {\
    register __m256i _sort_mask = _mm256_load_si256((__m256i*) shuffle_adjust);\
    union { int32_t i; uint8_t b[4]; uint16_t s[2]; } _move_mask;\
    unsigned int _lo_loc, _hi_loc;\
    \
    /* Reduce the high-value count and the offset into the 8-byte sort. */\
    do {\
        register __m256i _cmp_mask;\
        _cmp_mask = _mm256_add_epi8(adjust, a1);\
        _cmp_mask = _mm256_cmpgt_epi8(_cmp_mask, pivots);\
        _move_mask.i = _mm256_movemask_epi8(_cmp_mask);\
    } while (0);\
    \
    /* Determine how many high elements are in each section. */\
    loc = (unsigned int) __builtin_popcount(_move_mask.s[0]);\
    _lo_loc = (uint8_t) (unsigned int) __builtin_popcount(_move_mask.b[0]);\
    _hi_loc = (uint8_t) (unsigned int) __builtin_popcount(_move_mask.b[2]);\
    hic = (unsigned int) __builtin_popcount(_move_mask.s[1]);\
    \
    /* Load in the 8-element sort vectors. */\
    __attribute__((aligned(32))) uint64_t _sort8[4];\
    _sort8[0] = * (uint64_t*) sort1b_4x8[_move_mask.b[0]];\
    _sort8[1] = * (uint64_t*) sort1b_4x8[_move_mask.b[1]];\
    _sort8[2] = * (uint64_t*) sort1b_4x8[_move_mask.b[2]];\
    _sort8[3] = * (uint64_t*) sort1b_4x8[_move_mask.b[3]];\
    \
    do {\
        register __m256i _tmp8, _pre_roll, _tmp16_lo, _tmp16_hi;\
        /* Load in the 64-bit vectors that will sort each 64-bit subvector. */\
        _pre_roll = _mm256_load_si256((__m256i*) rrl_shuffle_hi[(loc) & 0xF]);\
        _tmp8 = _mm256_load_si256((__m256i*) _sort8);\
        \
        /* Do the same for the 128-bit sorting vectors */\
        _tmp16_lo = _mm256_loadu_si256((__m256i*) sort1b_2x16[_lo_loc + 1]);\
        _tmp16_hi = _mm256_loadu_si256((__m256i*) sort1b_2x16[_hi_loc]);\
        \
        /* Blend the 8/16 element sort vectors together. Then 16-sort the */\
        /* 8-sort vector to get a 16-sort vector. */\
        _tmp16_lo = _mm256_blend_epi32(_tmp16_lo, _tmp16_hi, 0xF0);\
        _sort_mask = _mm256_add_epi8(_sort_mask, _tmp8);\
        _sort_mask = _mm256_shuffle_epi8(_sort_mask, _tmp16_lo);\
        _sort_mask = _mm256_shuffle_epi8(_sort_mask, _pre_roll);\
    } while (0);\
    \
    a1 = _mm256_shuffle_epi8(a1, _sort_mask);\
    a2 = _mm256_shuffle_epi8(a2, _sort_mask);\
    a3 = _mm256_shuffle_epi8(a3, _sort_mask);\
} while (0)

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 * @param pivots The signed-adjusted pivot vector.
 * @param adjust The vector used to adjust a1 before the pivot comparison.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 * @param count Returns the number of high elements in the vector.
 */
#define ARGMSORT1X32(pivots, adjust, a1, a2, a3, count)\
do {\
    /* 16 sort the array after performing the comparison */\
    uint64_t _loc, _hic;\
    ARGMSORT2X16(pivots, adjust, a1, a2, a3, _loc, _hic);\
    count = _loc + _hic;\
    \
    register __m256i _roll_mask, _a1_lo, _a2_lo, _a3_lo;\
    \
    /* Create a mask to use to blend the lo and hi vectors. */\
    _roll_mask = _mm256_load_si256((__m256i*) srl_blend[_loc]);\
    \
    /* Reverse each channel to allow it to be blended. The rolling of the */\
    /* high channel was already done in argmsort2x16, as doing it there */\
    /* reduces the number of shuffles necessary and allows the load to */\
    /* happen further from its use. Note that the srl_blend vector has the */\
    /* low half of each SIMD vector inverted in anticipation of this permute */\
    /* time save (where we simply reverse instead of creating a */\
    /* high-high/low-low vector). */\
    _a1_lo = _mm256_permute2x128_si256(a1, a1, 0x01);\
    _a2_lo = _mm256_permute2x128_si256(a2, a2, 0x01);\
    _a3_lo = _mm256_permute2x128_si256(a3, a3, 0x01);\
    \
    /* Use the mask and rolled high-high vector to insert the upper half */\
    /* of each vector in between the low halfs low and high values. */\
    a1 = _mm256_and_si256(_roll_mask, a1);\
    a2 = _mm256_and_si256(_roll_mask, a2);\
    a3 = _mm256_and_si256(_roll_mask, a3);\
    _a1_lo = _mm256_andnot_si256(_roll_mask, _a1_lo);\
    _a2_lo = _mm256_andnot_si256(_roll_mask, _a2_lo);\
    _a3_lo = _mm256_andnot_si256(_roll_mask, _a3_lo);\
    a1 = _mm256_or_si256(a1, _a1_lo);\
    a2 = _mm256_or_si256(a2, _a2_lo);\
    a3 = _mm256_or_si256(a3, _a3_lo);\
} while (0)

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 *
 * Assumes each input is already 1x32 sorted.
 *
 * @param ac The number of high-values in a.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 * @param bc The number of high-values in b.
 * @param b1 An array to apply the masks sort to.
 * @param b2 An array to apply the masks sort to.
 * @param b3 An array to apply the masks sort to.
 */
#define ARGMSORT2X32(ac, a1, a2, a3, bc, b1, b2, b3)\
do {\
    /* Roll the b vectors right by the high-value count of a */\
    do {\
        register __m256i _b1_lo, _b2_lo, _b3_lo, _roll_mask, _shuffle_mask;\
        \
        /* Generate the 32-byte roll blending mask. */\
        /* And create the 16-byte roll shuffle mask. */\
        _shuffle_mask = _mm256_load_si256((__m256i*) rrl_shuffle[(ac) & 0xF]);\
        _roll_mask = _mm256_load_si256((__m256i*) srl_blend[ac]);\
        \
        /* Roll the b vectors by 16-byte rolling each one and then blending */\
        /* with the roll mask */\
        b1 = _mm256_shuffle_epi8(b1, _shuffle_mask);\
        b2 = _mm256_shuffle_epi8(b2, _shuffle_mask);\
        b3 = _mm256_shuffle_epi8(b3, _shuffle_mask);\
        \
        /* Generate the lo/hi b vectors to be blended. */\
        _b1_lo = _mm256_permute2x128_si256(b1, b1, 0x01);\
        _b2_lo = _mm256_permute2x128_si256(b2, b2, 0x01);\
        _b3_lo = _mm256_permute2x128_si256(b3, b3, 0x01);\
        \
        b1 = _mm256_and_si256(b1, _roll_mask);\
        b2 = _mm256_and_si256(b2, _roll_mask);\
        b3 = _mm256_and_si256(b3, _roll_mask);\
        _b1_lo = _mm256_andnot_si256(_roll_mask, _b1_lo);\
        _b2_lo = _mm256_andnot_si256(_roll_mask, _b2_lo);\
        _b3_lo = _mm256_andnot_si256(_roll_mask, _b3_lo);\
        b1 = _mm256_or_si256(b1, _b1_lo);\
        b2 = _mm256_or_si256(b2, _b2_lo);\
        b3 = _mm256_or_si256(b3, _b3_lo);\
    } while (0);\
    \
    do {\
        register __m256i _a1_tmp1, _a2_tmp1, _a3_tmp1, _b_blend;\
        register __m256i _a1_tmp2, _a2_tmp2, _a3_tmp2;\
        \
        /* Load the 64-byte blending masks. */\
        _b_blend = _mm256_load_si256((__m256i*) shifted_set_mask[ac]);\
        \
        /* Sort the a and b vectors across each other. */\
        _a1_tmp1 = _mm256_andnot_si256(_b_blend, b1);\
        _a1_tmp2 = _mm256_and_si256(_b_blend, a1);\
        _a2_tmp1 = _mm256_andnot_si256(_b_blend, b2);\
        _a2_tmp2 = _mm256_and_si256(_b_blend, a2);\
        _a3_tmp1 = _mm256_andnot_si256(_b_blend, b3);\
        _a3_tmp2 = _mm256_and_si256(_b_blend, a3);\
        _a1_tmp1 = _mm256_or_si256(_a1_tmp1, _a1_tmp2);\
        _a2_tmp1 = _mm256_or_si256(_a2_tmp1, _a2_tmp2);\
        _a3_tmp1 = _mm256_or_si256(_a3_tmp1, _a3_tmp2);\
        b1 = _mm256_and_si256(_b_blend, b1);\
        a1 = _mm256_andnot_si256(_b_blend, a1);\
        b2 = _mm256_and_si256(_b_blend, b2);\
        a2 = _mm256_andnot_si256(_b_blend, a2);\
        b3 = _mm256_and_si256(_b_blend, b3);\
        a3 = _mm256_andnot_si256(_b_blend, a3);\
        b1 = _mm256_or_si256(a1, b1);\
        b2 = _mm256_or_si256(a2, b2);\
        b3 = _mm256_or_si256(a3, b3);\
        a1 = _a1_tmp1;\
        a2 = _a2_tmp1;\
        a3 = _a3_tmp1;\
    } while (0);\
    \
    /* Update the high-value counts. */\
    uint64_t _hvc_tmp = MIN(32, (ac) + (bc));\
    ac = (uint64_t) MAX((int64_t)0, (int64_t) (((ac) + (bc)) - 32));\
    bc = _hvc_tmp;\
} while (0)

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
    __m256i *ch1,
    __m256i *ch2,
    __m256i *ch3,
    size_t size,
    uint8_t pivot
) {
    assert(size > 0);

    // Set up the pivot vector.
    union {
        float f;
        uint8_t p[4];
    } crime = { .p = { pivot, pivot, pivot, pivot } };
    register __m256i pivots = _mm256_castps_si256(_mm256_broadcast_ss(&crime.f));
    register __m256i adjust = _mm256_load_si256((__m256i*) cmp_adjust);
    pivots = _mm256_add_epi8(pivots, adjust);

    // Load and sort the first chunk. This is done here to avoid repeat work.
    size_t lo = 0, hi = size - 1;
    size_t next = hi;
    uint64_t ac;
    register __m256i a1, a2, a3;
    a1 = _mm256_load_si256(&ch1[lo]);
    a2 = _mm256_load_si256(&ch2[lo]);
    a3 = _mm256_load_si256(&ch3[lo]);
    ARGMSORT1X32(pivots, adjust, a1, a2, a3, ac);

    // Perform the partition for all except the last step.
    while (hi > lo) {
        // Load in and sort new chunk.
        uint64_t bc = 0;
        register __m256i b1, b2, b3;
        b1 = _mm256_load_si256(&ch1[next]);
        b2 = _mm256_load_si256(&ch2[next]);
        b3 = _mm256_load_si256(&ch3[next]);
        ARGMSORT1X32(pivots, adjust, b1, b2, b3, bc);

        // Sort across both chunks.
        ARGMSORT2X32(ac, a1, a2, a3, bc, b1, b2, b3);

        // Determine which side is full.
        if (ac == 0) {
            // amask is all zero, so the lo regs are full. Store them and then
            // move the b regs into the lo regs.
            _mm256_store_si256(&ch1[lo], a1);
            _mm256_store_si256(&ch2[lo], a2);
            _mm256_store_si256(&ch3[lo], a3);
            ac = bc;
            a1 = b1;
            a2 = b2;
            a3 = b3;
            next = ++lo;
        } else {
            // bmask must be filled with ones, so store those values.
            _mm256_store_si256(&ch1[hi], b1);
            _mm256_store_si256(&ch2[hi], b2);
            _mm256_store_si256(&ch3[hi], b3);
            next = --hi;
        }
    }

    // Store the final vector.
    _mm256_store_si256(&ch1[lo], a1);
    _mm256_store_si256(&ch2[lo], a2);
    _mm256_store_si256(&ch3[lo], a3);

    // Return the index.
    return lo;
}

static size_t
Partition(
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    uint8_t pivot,
    mc_time_t *time
) {
    // FIXME: There is probably a better way to deal with the unaligned bits.

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

        unsigned long long ts1, ts2;
        TIMESTAMP(ts1);
        bound = AlignPartition(
            (__m256i*) &ch1[pre_align],
            (__m256i*) &ch2[pre_align],
            (__m256i*) &ch3[pre_align],
            align_size,
            pivot
        );
        TIMESTAMP(ts2);
        time->align_time += (ts2 - ts1);
        time->align_units += align_size * 32;

        // Find the actual bound, now that we're mostly sorted.
        bound = (bound * 32) + pre_align;
        while ((bound < size) && (ch1[bound] <= pivot)) bound++;

        // Sort the not-aligned pre parts.
        register __m256i u1, u2, u3, t1, t2, t3, p, adj;
        size_t target = (size_t) MAX(0, ((ptrdiff_t)bound) - 32);
        adj = _mm256_load_si256((__m256i*) cmp_adjust);
        union {
            float f;
            uint8_t p[4];
        } crime = { .p = { pivot, pivot, pivot, pivot } };
        p = _mm256_castps_si256(_mm256_broadcast_ss(&crime.f));
        p = _mm256_add_epi8(adj, p);
        u1 = _mm256_loadu_si256((__m256i*) &ch1[0]);
        u2 = _mm256_loadu_si256((__m256i*) &ch2[0]);
        u3 = _mm256_loadu_si256((__m256i*) &ch3[0]);
        t1 = _mm256_loadu_si256((__m256i*) &ch1[target]);
        t2 = _mm256_loadu_si256((__m256i*) &ch2[target]);
        t3 = _mm256_loadu_si256((__m256i*) &ch3[target]);

        size_t count;
        ARGMSORT1X32(p, adj, u1, u2, u3, count);
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

        ARGMSORT1X32(p, adj, u1, u2, u3, count);
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
 * @return The kth element in a sorted version of buf.
 */
static uint8_t
QSelect(
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
        unsigned long long ts1, ts2;

        // Get our pivot. Random is "good enough" for O(n) in most cases.
        size_t pivot_idx = ((size_t) (unsigned int) rand()) % size;
        uint8_t pivot = ch1[pivot_idx];
        pivot = MIN(pivot, max_pivot);
        pivot = MAX(pivot, min_pivot);

        // Partition across our pivot.
        TIMESTAMP(ts1);
        size_t mid = Partition(ch1, ch2, ch3, size, pivot, time);
        TIMESTAMP(ts2);
        assert(mid <= size);

        time->part_units += size;
        time->part_time += (ts2 - ts1);

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
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    mc_time_t *time
) {
    unsigned long long ts1, ts2;

    // Use the quickselect algorithm to find the median.
    size_t mid = size >> 1;
    uint8_t median = QSelect(ch1, ch2, ch3, size, mid, time);

    // Partition across the median.
    TIMESTAMP(ts1);
    size_t lo_size = Partition(ch1, ch2, ch3, size, median, time);
    TIMESTAMP(ts2);
    assert(lo_size > mid);

    time->part_units += size;
    time->part_time += (ts2 - ts1);

    // Partition again across (median - 1) to force all median values to the
    // middle of the array.
    if (median > 0) {
        TIMESTAMP(ts1);
        assert(Partition(ch1, ch2, ch3, lo_size, median - 1, time) <= mid);
        TIMESTAMP(ts2);

        time->part_units += lo_size;
        time->part_time += (ts2 - ts1);
    }

    return mid;
}
