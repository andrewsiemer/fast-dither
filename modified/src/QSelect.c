/**
 * @file QSelect.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Exposes a QSelect and Partition function for unsigned arrays.
 * @bug No known bugs.
 */

#include <QSelect.h>

#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include <UtilMacro.h>

/*** Special look-up table for bitonic sort. Only used in this file. ***/
#include <sort_lut.h>

/// @brief An AVX vector containing all ones.
__attribute__((aligned (32))) const uint8_t oops_all_ones[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

/// @brief An AVX vector used to manually compute movemask.
__attribute__((aligned (32))) const uint8_t move_mask[32] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
};

/// @brief The identitiy shuffle for epi8.
__attribute__((aligned (32))) const uint8_t shuffle8id[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

/**
 * @brief Moves the high half of the 32-byte vector over by the given number
 *        of bytes, inserting it between the parts of the low half.
 */
#define MHHR1X32(v, b)\
do {\
    uint8_t _b = (b) & 0xF;\
    register __m256i _roll_mask = _mm256_cmpeq_epi8(v, v);\
    _roll_mask = _mm256_bslli_epi128(_roll_mask, 16 - ((b) & 0xF));\
    register __m128i _roll_lo = _mm_setzero_si128();\
    _roll_lo = _mm_cmpeq_epi8(_roll_lo, _roll_lo);\
    _roll_mask = _mm256_xor_si256(_roll_mask, _mm256_zextsi128_si256(_roll_lo));\
    \
    register __m256i _v_hi, _v_lo, _tmp;\
    _v_hi = _mm256_permute2x128_si256(v, v, 0x11);\
    v = _mm256_permute2x128_si256(v, v, 0x00);\
    _tmp = _mm256_bslli_epi128(_v_hi, 16 - _b);\
    _v_hi = _mm256_bsrli_epi128(_v_hi, _b);\
    _v_hi = _mm256_or_si256(_v_hi, _tmp);\
    \
    (v) = _mm256_blendv_epi8(_v_lo, _v_hi, _roll_mask);
} while (0)

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
    uint64_t _hi_counts[4];\
    \
    /* Use the given mask to create the scattered count and offset vectors. */\
    do {\
        register __m256i _ones, _mvmask;\
        _ones = _mm256_load_si256((const __m256i*) oops_all_ones);\
        _mvmask = _mm256_load_si256((const __m256i*) move_mask);\
        _hi_count = _mm256_and_si256(mask, _ones);\
        _sort_offset = _mm256_and_si256(mask, _mvmask);\
    } while (0);\
    \
    /* Reduce the high-value count and the offset into the 8-byte sort. */\
    do {\
        register __m256i _ctmp, _otmp, _hi_count, _sort_offset, _lsb_mask;\
        _lsb_mask = _mm256_srli_epi64(_mm256_cmpeq_epi8(a1, a1), 56);\
        _ctmp = _mm256_bsrli_epi128(_hi_count, 4);\
        _otmp = _mm256_bsrli_epi128(_sort_offset, 4);\
        _hi_count = _mm256_add_epi8(_ctmp, _hi_count);\
        _sort_offset = _mm256_or_si256(_otmp, _sort_offset);\
        \
        _ctmp = _mm256_bsrli_epi128(_hi_count, 2);\
        _otmp = _mm256_bsrli_epi128(_sort_offset, 2);\
        _hi_count = _mm256_add_epi8(_ctmp, _hi_count);\
        _sort_offset = _mm256_or_si256(_otmp, _sort_offset);\
        \
        _ctmp = _mm256_bsrli_epi128(_hi_count, 1);\
        _otmp = _mm256_bsrli_epi128(_sort_offset, 1);\
        _hi_count = _mm256_add_epi8(_ctmp, _hi_count);\
        _sort_offset = _mm256_or_si256(_otmp, _sort_offset);\
        \
        _hi_count = _mm256_and_si256(_hi_count, _lsb_mask);\
        _sort_offset = _mm256_and_si256(_sort_offset, _lsb_mask);\
        \
        _sort_mask = _mm256_i64gather_epi64(sort_imm, _sort_offset, 8);\
        _mm256_store_si256(&_hi_counts, _hi_count);\
        loc = _hi_counts[0] + _hi_counts[1];\
        hic = _hi_counts[2] + _hi_counts[3];\
    } while (0);\
    \
    /* Generate and apply a 16-byte sort shuffle to the 8-byte sort shuffle */\
    /* to create a "true" 16-byte sort shuffle vector. */\
    do {\
        /* FIXME: This can be done without a lookup table */\
        register __m128i _sort16_lo, _sort16_hi;\
        _sort16_lo = _mm_load_si128(&sort1b_2x16[_hi_counts[0]][_hi_counts[1]]);\
        _sort16_hi = _mm_load_si128(&sort1b_2x16[_hi_counts[2]][_hi_counts[3]]);\
        register __m256i _sort16 = _mm256_set_m128i(_sort16_hi, _sort16_lo);\
        _sort_mask = _mm256_shuffle_epi8(_sort_mask, _sort16);\
    } while (0);\
    \
    a1 = _mm256_shuffle_epi8(a1, _sort_mask);\
    a2 = _mm256_shuffle_epi8(a2, _sort_mask);\
    a3 = _mm256_shuffle_epi8(a3, _sort_mask);\
} while (0)

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 * @param mask The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 */
#define ARGMSORT1X32(mask, a1, a2, a3, count)\
do {\
    uint64_t _loc, _hic;\
    ARGMSORT2X16(mask, a1, a2, a3, _loc, _hic);\
    \
    register __m256i _move_mask, _a1_hi, _a2_hi, _a3_hi;\
    \
    /* Create a mask to use to blend the lo and hi vectors. */\
    do {\
        register __m128i _tmp_lo;\
        register __m256i _tmp;\
        _tmp_lo = _mm_setzero_si128();\
        _tmp_lo = _mm_cmpeq_epi8(_tmp_lo, _tmp_lo);\
        _tmp = _mm256_zextsi128_si256(_tmp_lo);\
        _move_mask = _mm256_cmpeq_epi8(a1, a1);\
        _move_mask = _mm256_bsrli_epi128(_move_mask, _hic);\
        _move_mask = _mm256_xor_si256(_move_mask, _tmp);\
    } while (0);\
    \
    /* Roll the high vectors by the high count value, then create a */\
    /* high-high and low-low vector for each vector */\
    do {\
        register _tmp, _roll_shuffle;\
        _roll_shuffle = _mm256_load_si256(shuffle8id);\
        _tmp = _mm256_bslli_epi128(_roll_shuffle, 16 - _hic);\
        _roll_shuffle = _mm256_bsrli_epi128(_roll_shuffle, _hic);\
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
 * @param amask The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 * @param bmask The mask array to use for sorting.
 * @param b1 An array to apply the masks sort to.
 * @param b2 An array to apply the masks sort to.
 * @param b3 An array to apply the masks sort to.
 */
#define ARGMSORT2X32(amask, a1, a2, a3, bmask, b1, b2, b3)\
do {\
    register __mm256i tmp2, masklo, maskhi;\
    /* Use a control block to hint to the compiler when these regs are done. */\
    do {\
        /* Set up masks for selecting our regs. */\
        register __mm256i notbmask, revamask, revbmask, tmp1;\
        tmp1 = _mm256_load_si256((const __m256i*) not_zero);\
        notbmask = _mm256_xor_si256(bmask, tmp1);\
        tmp2 = _mm256_load_si256((const __m256i*) reverse_2x16_vec);\
        revamask = _mm256_shuffle_epi8(amask, tmp2);\
        revbmask = _mm256_shuffle_epi8(bmask, tmp2);\
        revamask = _mm256_permute2x128_si256(revamask, revamask, 0x01);\
        revbmask = _mm256_permute2x128_si256(revbmask, revbmask, 0x01);\
        masklo = _mm256_or_si256(amask, revbmask);\
        maskhi = _mm256_or_si256(notbmask, revamask);\
        revbmask = _mm256_xor_si256(revbmask, tmp1);\
        \
        /* Sort the a and b masks. */\
        amask = _mm256_blendv_epi8(amask, revbmask, masklo);\
        bmask = _mm256_blendv_epi8(bmask, revamask, maskhi);\
    } while (0);\
    \
    /* Sort each {ai, bi} vector. */\
    register __m256i reva1, revb1, reva2, revb2, reva3, revb3;\
    reva1 = _mm256_shuffle_epi8(a1, tmp2);\
    reva2 = _mm256_shuffle_epi8(a2, tmp2);\
    reva3 = _mm256_shuffle_epi8(a3, tmp2);\
    revb1 = _mm256_shuffle_epi8(b1, tmp2);\
    revb2 = _mm256_shuffle_epi8(b2, tmp2);\
    revb3 = tmp2;\
    revb3 = _mm256_shuffle_epi8(b3, revb3);\
    reva1 = _mm256_permute2x128_si256(reva1, reva1, 0x01);\
    reva2 = _mm256_permute2x128_si256(reva2, reva2, 0x01);\
    reva3 = _mm256_permute2x128_si256(reva3, reva3, 0x01);\
    revb1 = _mm256_permute2x128_si256(revb1, revb1, 0x01);\
    revb2 = _mm256_permute2x128_si256(revb2, revb2, 0x01);\
    revb3 = _mm256_permute2x128_si256(revb3, revb3, 0x01);\
    a1 = _mm256_blendv_epi8(a1, revb1, masklo);\
    a2 = _mm256_blendv_epi8(a2, revb2, masklo);\
    a3 = _mm256_blendv_epi8(a3, revb3, masklo);\
    b1 = _mm256_blendv_epi8(b1, reva1, maskhi);\
    b2 = _mm256_blendv_epi8(b2, reva2, maskhi);\
    b3 = _mm256_blendv_epi8(b3, reva3, maskhi);\
    \
    ARGMSORT1X32(amask, a1, a2, a3);\
    ARGMSORT1X32(bmask, b1, b2, b3);\
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
    assert(size > 1);

    // Set up the pivot vector.
    __attribute__((aligned (32))) uint8_t pivot_arr[32];
    for (size_t i = 0; i < sizeof(pivot_arr); i++) { pivot_arr[i] = pivot; }
    const __m256i *pivots = (const __m256i *) pivot_arr;

    // Load and sort the first chunk. This is done here to avoid repeat work.
    size_t lo = 0, hi = size - 1;
    size_t next = hi;
    uint64_t ac;
    register __m256i a1, a2, a3, amask;
    a1 = _mm256_load_si256(&ch1[lo]);
    a2 = _mm256_load_si256(&ch2[lo]);
    a3 = _mm256_load_si256(&ch3[lo]);
    amask = _mm256_cmpgt_epi8(a1, _mm256_load_si256(pivots));
    ARGMSORT1X32(amask, a1, a2, a3, ac);

    // Perform the partition for all except the last step.
    while (hi > lo) {
        // Load in and sort new chunk.
        uint64_t bc;
        register __m256i b1, b2, b3, bmask;
        b1 = _mm256_load_si256(&ch1[next]);
        b2 = _mm256_load_si256(&ch2[next]);
        b3 = _mm256_load_si256(&ch3[next]);
        bmask = _mm256_cmpgt_epi8(b1, _mm256_load_si256(pivots));
        ARGMSORT1X32(bmask, b1, b2, b3, bc);

        // Sort across both chunks.
        ARGMSORT2X32(ac, a1, a2, a3, bc, b1, b2, b3);

        // Determine which side is full.
        if (_mm256_testc_si256(_mm256_setzero_si256(), amask)) {
            // amask is all zero, so the lo regs are full. Store them and then
            // move the b regs into the lo regs.
            _mm256_store_si256(&ch1[lo], a1);
            _mm256_store_si256(&ch2[lo], a2);
            _mm256_store_si256(&ch3[lo], a3);
            a1 = b1;
            a2 = b2;
            a3 = b3;
            amask = bmask;
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
    // Align each channel to the bounds for AlignPartition().
    // TODO

    // Perform the aligned partition.
    // TODO
}

void
MedianPartition(
    uint32_t *buf,
    size_t size,
    uint32_t pivot,
    size_t *plo,
    size_t *phi
) {
    assert(buf);
    assert(size > 0);
    assert(plo);
    assert(phi);

    // Note that mid_hi may underflow, due to it representing the first valid
    // element in the list of values equal to the midpoint. As such, we
    // compare lo to mid_lo and hi to mid_hi by their distance, which will
    // always be some value <= size/2, due to our start selection.
    size_t lo = 0, hi = size - 1;
    size_t mid_lo = hi >> 1, mid_hi = (hi >> 1) - 1;

    // Partition lower half.
    while ((mid_lo - lo) > 0) {
        if (buf[lo] < pivot) {
            lo++;
        } else if (buf[lo] > pivot) {
            SWAP(buf[lo], buf[hi]);
            if (hi-- == mid_hi) {
                // If the mid-point region is empty, we need to move both.
                if (((ptrdiff_t)((mid_hi + 1) - mid_lo)) == 0) { mid_lo--; }
                mid_hi--;
            }
        } else {
            // Exact match, move middle.
            mid_lo--;
            SWAP(buf[lo], buf[mid_lo]);
        }
    }

    // Partition whatever remains of the top half.
    while ((hi - mid_hi) > 0) {
        if (buf[hi] > pivot) {
            hi--;
        } else if (buf[hi] < pivot) {
            SWAP(buf[lo], buf[hi]);

            // If we're here, lo == mid_lo, so we need to move mid_lo up.
            // This means the middle loses an element, but we'll fix
            // it on the next iteration.
            lo++;
            if (((ptrdiff_t)((mid_hi + 1) - mid_lo)) == 0) { mid_hi++; }
            mid_lo++;
        } else {
            mid_hi++;
            SWAP(buf[hi], buf[mid_hi]);
        }
    }

    // If we didn't find the pivot, yell.
    assert(mid_hi >= mid_lo);
    *plo = mid_lo;
    *phi = mid_hi;
}

uint8_t
QSelect(
    uint8_t *ch1,
    uint8_t *ch2,
    uint8_t *ch3,
    size_t size,
    size_t k
) {
    assert(buf);
    assert(size > 0);
    assert(k < size);

    while (size > 1) {
        // Get our pivot. Random is "good enough" for O(n) in most cases.
        size_t pivot_idx = ((size_t) (unsigned int) rand()) % size;
        uint8_t pivot = ch1[pivot_idx];

        // Partition across our pivot.
        size_t mid = Partition(ch1, ch2, ch3, size, pivot);

        if (k <= mid) {
            size = mid;
        } else if (k > mid) {
            ch1 = &ch1[mid];
            ch2 = &ch2[mid];
            ch3 = &ch3[mid];
            size -= mid;
            k -= mid;
        }
    }

    return ch1[0];
}
