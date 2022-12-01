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

#include <emmintrin.h>
#include <immintrin.h>

#include <UtilMacro.h>

/*** Special look-up table for bitonic sort. Only used in this file. ***/
#include <sort_lut.h>

/// @brief Used to adjust a shuffle vector which operates on 8-byte subvectors.
__attribute__((aligned (32))) static const uint8_t shuffle_adjust[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8
};

/// @brief Used to adjust the signed cmp_epi8 to unsigned.
__attribute__((aligned (32))) static const uint8_t cmp_adjust[32] = {
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128
};

/// @brief Loads a mask which sets the low 128-bits to one.
__attribute__((aligned (32))) static const uint8_t half_mask[32] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

/// @brief Masks to set all bits in a register based on the index bit.
__attribute__((aligned (32))) static const uint8_t maybe_not[2][32] = {
    { 0 },
    { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }
};

/**
 * @brief Shuffle vectors to sort two 8-element sorted vectors into a 16-element
 *        sorted vector.
 *
 * Indexed by the number of high-elements in the lower 8-element vector.
 *
 * An extra element is added, since we use an unaligned 256-bit load to
 * read this array.
 */
__attribute__((aligned (16))) static const uint8_t sort1b_2x16[10][16] = {
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
__attribute__((aligned (32))) static const uint8_t rrl_shuffle[17][32] = {
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
      15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
};

/**
 * @brief Sorts 16-bytes chunks of a vector based on the mask in arg.
 * @param arg The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 * @param loc Returns the number of high values in the lower 16 bytes.
 * @param hic Returns the number of high values in the higher 16 bytes.
 */
#define ARGMSORT2X16(mask, a1, a2, a3, loc, hic)\
do {\
    register __m256i _sort_mask;\
    union { uint32_t u; int32_t i; uint8_t b[4]; } _move_mask, _hi_counts;\
    \
    /* Reduce the high-value count and the offset into the 8-byte sort. */\
    do {\
        register __m256i _tmp;\
        __attribute__((aligned(32))) uint64_t _sort8[4];\
        _move_mask.i = _mm256_movemask_epi8(mask);\
        \
        /* For some reason, this is much faster than popcount */\
        _hi_counts.u = _move_mask.u;\
        _hi_counts.u = (_hi_counts.u & 0x55555555)\
                     + ((_hi_counts.u & 0xAAAAAAAA) >> 1);\
        _hi_counts.u = (_hi_counts.u & 0x33333333)\
                     + ((_hi_counts.u & 0xCCCCCCCC) >> 2);\
        _hi_counts.u = (_hi_counts.u & 0x0F0F0F0F)\
                     + ((_hi_counts.u & 0xF0F0F0F0) >> 4);\
        \
        _sort8[0] = * (uint64_t*) sort1b_4x8[_move_mask.b[0]];\
        _sort8[1] = * (uint64_t*) sort1b_4x8[_move_mask.b[1]];\
        _sort8[2] = * (uint64_t*) sort1b_4x8[_move_mask.b[2]];\
        _sort8[3] = * (uint64_t*) sort1b_4x8[_move_mask.b[3]];\
        _tmp = _mm256_load_si256((__m256i*) shuffle_adjust);\
        _sort_mask = _mm256_load_si256((__m256i*) _sort8);\
        _sort_mask = _mm256_add_epi8(_sort_mask, _tmp);\
        \
        loc = _hi_counts.b[0] + _hi_counts.b[1];\
        hic = _hi_counts.b[2] + _hi_counts.b[3];\
    } while (0);\
    \
    /* Generate and apply a 16-byte sort shuffle to the 8-byte sort shuffle */\
    /* to create a "true" 16-byte sort shuffle vector. */\
    do {\
        register __m256i _tmp_lo, _tmp_hi;\
        _tmp_lo = _mm256_loadu_si256((__m256i*) sort1b_2x16[_hi_counts.b[0]]);\
        _tmp_hi = _mm256_loadu_si256((__m256i*) sort1b_2x16[_hi_counts.b[2]]);\
        _tmp_lo = _mm256_permute2x128_si256(_tmp_lo, _tmp_hi, 0x20);\
        _sort_mask = _mm256_shuffle_epi8(_sort_mask, _tmp_lo);\
    } while (0);\
    \
    a1 = _mm256_shuffle_epi8(a1, _sort_mask);\
    a2 = _mm256_shuffle_epi8(a2, _sort_mask);\
    a3 = _mm256_shuffle_epi8(a3, _sort_mask);\
} while (0)

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 * @param pivots The signed-adjusted pivot vector.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 */
#define ARGMSORT1X32(pivots, a1, a2, a3, count)\
do {\
    uint64_t _loc, _hic;\
    \
    /* 16 sort the array after performing the comparison */\
    do {\
        register __m256i _mask;\
        _mask = _mm256_load_si256((__m256i*) cmp_adjust);\
        _mask = _mm256_add_epi8(_mask, a1);\
        _mask = _mm256_cmpgt_epi8(_mask, pivots);\
        ARGMSORT2X16(_mask, a1, a2, a3, _loc, _hic);\
    } while (0);\
    \
    register __m256i _move_mask, _a1_hi, _a2_hi, _a3_hi;\
    \
    /* Create a mask to use to blend the lo and hi vectors. */\
    _move_mask = _mm256_load_si256((__m256i*) srl_blend[_loc]);\
    \
    /* Roll the high vectors by the high count value, then create a */\
    /* high-high and low-low vector for each vector */\
    do {\
        register __m256i _roll_shuffle;\
        _roll_shuffle = _mm256_load_si256((__m256i*) rrl_shuffle[_loc]);\
        \
        _a1_hi = _mm256_permute2x128_si256(a1, a1, 0x11);\
        _a2_hi = _mm256_permute2x128_si256(a2, a2, 0x11);\
        _a3_hi = _mm256_permute2x128_si256(a3, a3, 0x11);\
        a1 = _mm256_permute2x128_si256(a1, a1, 0x00);\
        a2 = _mm256_permute2x128_si256(a2, a2, 0x00);\
        a3 = _mm256_permute2x128_si256(a3, a3, 0x00);\
        _a1_hi = _mm256_shuffle_epi8(_a1_hi, _roll_shuffle);\
        _a2_hi = _mm256_shuffle_epi8(_a2_hi, _roll_shuffle);\
        _a3_hi = _mm256_shuffle_epi8(_a3_hi, _roll_shuffle);\
    } while (0);\
    \
    /* Use the mask and rolled high-high vector to insert the upper half */\
    /* of each vector in between the low halfs low and high values. */\
    a1 = _mm256_blendv_epi8(a1, _a1_hi, _move_mask);\
    a2 = _mm256_blendv_epi8(a2, _a2_hi, _move_mask);\
    a3 = _mm256_blendv_epi8(a3, _a3_hi, _move_mask);\
    count = _loc + _hic;\
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
    register __m256i _a_blend, _b_blend;\
    /* Roll the b vectors right by the high-value count of a */\
    do {\
        register __m256i _b1_lo, _b2_lo, _b3_lo, _maybe_blend;\
        register __m256i _roll_mask, _tmp, _rtmp;\
        size_t _hi_sel = ((ac) >> 4) > 0;\
        size_t _hic = ((ac) & 0xF) + (((ac) >> 1) & 0x10);\
        \
        /* Begin preparing the 64-byte blending masks. */\
        _rtmp = _mm256_load_si256((__m256i*) half_mask);\
        _a_blend = _mm256_setzero_si256();\
        _a_blend = _mm256_cmpeq_epi8(_a_blend, _a_blend);\
        \
        /* Generate the 32-byte roll blending mask. */\
        _b_blend = _mm256_load_si256((__m256i*) srl_blend[_hic]);\
        _tmp = _mm256_load_si256((__m256i*) &maybe_not[_hi_sel]);\
        _roll_mask = _mm256_xor_si256(_tmp, _b_blend);\
        \
        /* Finish creating the final blending mask by making an (ac % 16) */\
        /* shifted set vector and an (ac % 16) + 16 shifted vector and */\
        /* selecting the vector using the 16ths place in ac. */\
        _maybe_blend = _mm256_permute2x128_si256(_rtmp, _b_blend, 0x03);\
        _b_blend = _mm256_permute2x128_si256(_rtmp, _b_blend, 0x31);\
        \
        /* Generate the b vectors to be 16-byte rolled and blended. */\
        _b1_lo = _mm256_permute2x128_si256(b1, b1, 0x00);\
        _b2_lo = _mm256_permute2x128_si256(b2, b2, 0x00);\
        _b3_lo = _mm256_permute2x128_si256(b3, b3, 0x00);\
        b1 = _mm256_permute2x128_si256(b1, b1, 0x11);\
        b2 = _mm256_permute2x128_si256(b2, b2, 0x11);\
        b3 = _mm256_permute2x128_si256(b3, b3, 0x11);\
        \
        /* Create the 16-byte roll shuffle mask. */\
        _rtmp = _mm256_load_si256((__m256i*) rrl_shuffle[_hic]);\
        \
        /* Select the blend vector */\
        _maybe_blend = _mm256_and_si256(_maybe_blend, _tmp);\
        _tmp = _mm256_xor_si256(_tmp, _a_blend);\
        _b_blend = _mm256_and_si256(_b_blend, _tmp);\
        _b_blend = _mm256_or_si256(_b_blend, _maybe_blend);\
        _a_blend = _mm256_xor_si256(_a_blend, _b_blend);\
        \
        /* Roll the b vectors by 16-byte rolling each one and then blending */\
        /* with the roll mask */\
        b1 = _mm256_shuffle_epi8(b1, _rtmp);\
        b2 = _mm256_shuffle_epi8(b2, _rtmp);\
        b3 = _mm256_shuffle_epi8(b3, _rtmp);\
        _b1_lo = _mm256_shuffle_epi8(_b1_lo, _rtmp);\
        _b2_lo = _mm256_shuffle_epi8(_b2_lo, _rtmp);\
        _b3_lo = _mm256_shuffle_epi8(_b3_lo, _rtmp);\
        b1 = _mm256_blendv_epi8(_b1_lo, b1, _roll_mask);\
        b2 = _mm256_blendv_epi8(_b2_lo, b2, _roll_mask);\
        b3 = _mm256_blendv_epi8(_b3_lo, b3, _roll_mask);\
    } while (0);\
    \
    do {\
        /* Sort the a and b vectors across each other. */\
        register __m256i _a1_tmp, _a2_tmp, _a3_tmp;\
        _a1_tmp = _mm256_blendv_epi8(a1, b1, _a_blend);\
        _a2_tmp = _mm256_blendv_epi8(a2, b2, _a_blend);\
        _a3_tmp = _mm256_blendv_epi8(a3, b3, _a_blend);\
        b1 = _mm256_blendv_epi8(a1, b1, _b_blend);\
        b2 = _mm256_blendv_epi8(a2, b2, _b_blend);\
        b3 = _mm256_blendv_epi8(a3, b3, _b_blend);\
        a1 = _a1_tmp;\
        a2 = _a2_tmp;\
        a3 = _a3_tmp;\
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
    register __m256i tmp = _mm256_load_si256((__m256i*) cmp_adjust);
    pivots = _mm256_add_epi8(pivots, tmp);

    // Load and sort the first chunk. This is done here to avoid repeat work.
    size_t lo = 0, hi = size - 1;
    size_t next = hi;
    uint64_t ac;
    register __m256i a1, a2, a3;
    a1 = _mm256_load_si256(&ch1[lo]);
    a2 = _mm256_load_si256(&ch2[lo]);
    a3 = _mm256_load_si256(&ch3[lo]);
    ARGMSORT1X32(pivots, a1, a2, a3, ac);

    // Perform the partition for all except the last step.
    while (hi > lo) {
        // Load in and sort new chunk.
        uint64_t bc = 0;
        register __m256i b1, b2, b3;
        b1 = _mm256_load_si256(&ch1[next]);
        b2 = _mm256_load_si256(&ch2[next]);
        b3 = _mm256_load_si256(&ch3[next]);
        ARGMSORT1X32(pivots, b1, b2, b3, bc);

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
    uint8_t pivot
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
        bound = AlignPartition(
            (__m256i*) &ch1[pre_align],
            (__m256i*) &ch2[pre_align],
            (__m256i*) &ch3[pre_align],
            align_size,
            pivot
        );

        bound = (bound * 32) + pre_align;
        while ((bound < size) && (ch1[bound] <= pivot)) bound++;
    }

    /* Partition the unaligned parts. */

    for (size_t lo = 0; (lo < pre_align) && (lo < bound);) {
        if (ch1[lo] > pivot) {
            bound--;
            SWAP(ch1[lo], ch1[bound]);
            SWAP(ch2[lo], ch2[bound]);
            SWAP(ch3[lo], ch3[bound]);
        } else {
            lo++;
        }
    }

    for (size_t hi = size - 1; (hi >= (size - post_align)) && (hi >= bound);) {
        if (ch1[hi] <= pivot) {
            SWAP(ch1[hi], ch1[bound]);
            SWAP(ch2[hi], ch2[bound]);
            SWAP(ch3[hi], ch3[bound]);
            bound++;
        } else {
            hi--;
        }
    }

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
    size_t k
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

    while ((size > 1) && (min_pivot < max_pivot)) {
        // Get our pivot. Random is "good enough" for O(n) in most cases.
        size_t pivot_idx = ((size_t) (unsigned int) rand()) % size;
        uint8_t pivot = ch1[pivot_idx];
        pivot = MIN(pivot, max_pivot);
        pivot = MAX(pivot, min_pivot);

        // Partition across our pivot.
        size_t mid = Partition(ch1, ch2, ch3, size, pivot);
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
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size
) {
    // Use the quickselect algorithm to find the median.
    size_t mid = size >> 1;
    uint8_t median = QSelect(ch1, ch2, ch3, size, mid);

    // Partition across the median.
    size_t lo_size = Partition(ch1, ch2, ch3, size, median);
    assert(lo_size > 0);

    // Partition again across (median - 1) to force all median values to the
    // middle of the array.
    if (median > 0) {
        Partition(ch1, ch2, ch3, lo_size, median - 1);
    }

    return mid;
}
