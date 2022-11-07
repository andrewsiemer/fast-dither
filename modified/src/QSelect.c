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

/// @brief An AVX vector used to reverse 2 16-byte chunks of a vec.
__attribute__((aligned (32))) const uint8_t reverse_2x16_vec[32] = {
    15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0,
    15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0
};

/// @brief An AVX vector used to invert 8-byte lanes for argmsort.
__attribute__((aligned (32))) const uint8_t argmsort_2x16_invert[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

/// @brief An AVX vector used to invert 16-byte lanes for argmsort.
__attribute__((aligned (32))) const uint8_t argmsort_1x32_invert[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

/// @brief An AVX vector containing all ones.
__attribute__((aligned (32))) const uint8_t oops_all_ones[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

/**
 * @brief Reverses 16-byte chunks of a vector.
 */
#define REVERSE2X16(vec)\
    _mm256_shuffle_epi8(\
        vec,\
        _mm256_load_si256((const __m256i*) reverse_2x16_vec)\
    )

/**
 * @brief Reverses a 32-byte vector.
 */
#define REVERSE1X32(vec) REVERSE2X16(_mm256_permute2x128_si256(vec, vec, 0x01))

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 * @param arg The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 */
#define ARGMSORT4X8(arg, a1, a2, a3)\
do {\
    const uint64_t *sort_imm = (uint64_t*) sort1b_4x8; /* OK; little endian. */\
    \
    /* Get the sorting vector. */\
    uint32_t mask = (uint32_t) (unsigned int) (int) _mm256_movemask_epi8(arg);\
    __attribute__((aligned (32)))\
    const uint64_t sort_vec[4] = {\
        sort_imm[mask & 0xFF],\
        sort_imm[(mask >> 8) & 0xFF],/* FIXME: Need to add 8 to each elt */\
        sort_imm[(mask >> 16) & 0xFF],\
        sort_imm[(mask >> 24) & 0xFF]/* FIXME */\
    };\
    \
    register __m256i tmp = _mm256_load_si256((const __m256i*) sort_vec);\
    (arg) = _mm256_shuffle_epi8(arg, tmp);\
    (a1) = _mm256_shuffle_epi8(a1, tmp);\
    (a2) = _mm256_shuffle_epi8(a2, tmp);\
    (a3) = _mm256_shuffle_epi8(a3, tmp);\
} while (0)

/**
 * @brief Sorts 16-bytes chunks of a vector based on the mask in arg.
 * @param arg The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 */
#define ARGMSORT2X16(arg, a1, a2, a3)\
do {\
    ARGMSORT4X8(arg, a1, a2, a3);\
    \
    register __m256i tmp1, tmp2, mask, revarg, reva1, reva2, reva3;\
    tmp1 = _mm256_load_si256((const __m256i*) argmsort_2x16_invert);\
    tmp2 = _mm256_xor_si256(arg, tmp1);\
    tmp1 = _mm256_load_si256((const __m256i*) reverse_2x16_vec);\
    revarg = _mm256_shuffle_epi8(arg, tmp1);\
    reva1 = _mm256_shuffle_epi8(a1, tmp1);\
    reva2 = _mm256_shuffle_epi8(a2, tmp1);\
    reva3 = _mm256_shuffle_epi8(a3, tmp1);\
    tmp1 = _mm256_shuffle_epi8(tmp2, tmp1);\
    mask = _mm256_or_si256(tmp1, tmp2);\
    arg = _mm256_blendv_epi8(arg, revarg, mask);\
    a1 = _mm256_blendv_epi8(a1, reva1, mask);\
    a2 = _mm256_blendv_epi8(a2, reva2, mask);\
    a3 = _mm256_blendv_epi8(a3, reva3, mask);\
    \
    ARGMSORT4X8(arg, a1, a2, a3);\
} while (0)

/**
 * @brief Sorts 8-bytes chunks of a vector based on the mask in arg.
 * @param arg The mask array to use for sorting.
 * @param a1 An array to apply the masks sort to.
 * @param a2 An array to apply the masks sort to.
 * @param a3 An array to apply the masks sort to.
 */
#define ARGMSORT1X32(arg, a1, a2, a3)\
do {\
    ARGMSORT2X16(arg, a1, a2, a3);\
    \
    register __m256i tmp1, tmp2, mask, revarg, reva1, reva2, reva3;\
    tmp1 = _mm256_load_si256((const __m256i*) argmsort_1x32_invert);\
    tmp2 = _mm256_xor_si256(arg, tmp1);\
    \
    tmp1 = _mm256_load_si256((const __m256i*) reverse_2x16_vec);\
    revarg = _mm256_shuffle_epi8(arg, tmp1);\
    reva1  = _mm256_shuffle_epi8(a1, tmp1);\
    reva2  = _mm256_shuffle_epi8(a2, tmp1);\
    reva3  = _mm256_shuffle_epi8(a3, tmp1);\
    tmp1   = _mm256_shuffle_epi8(tmp2, tmp1);\
    revarg = _mm256_permute2x128_si256(arg, arg, 0x01);\
    reva1  = _mm256_permute2x128_si256(a1, a1, 0x01);\
    reva2  = _mm256_permute2x128_si256(a2, a2, 0x01);\
    reva3  = _mm256_permute2x128_si256(a3, a3, 0x01);\
    tmp1   = _mm256_permute2x128_si256(tmp1, tmp1, 0x01);\
    \
    mask = _mm256_or_si256(tmp1, tmp2);\
    arg = _mm256_blendv_epi8(arg, revarg, mask);\
    a1  = _mm256_blendv_epi8(a1, reva1, mask);\
    a2  = _mm256_blendv_epi8(a2, reva2, mask);\
    a3  = _mm256_blendv_epi8(a3, reva3, mask);\
    \
    ARGMSORT2X16(arg, a1, a2, a3);\
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
        tmp1 = _mm256_load_si256((const __m256i*) oops_all_ones);\
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
    register __m256i a1, a2, a3, amask;
    a1 = _mm256_load_si256(&ch1[lo]);
    a2 = _mm256_load_si256(&ch2[lo]);
    a3 = _mm256_load_si256(&ch3[lo]);
    amask = _mm256_cmpgt_epi8(a1, _mm256_load_si256(pivots));
    ARGMSORT1X32(amask, a1, a2, a3);

    // Perform the partition for all except the last step.
    while (hi > lo) {
        // Load in and sort new chunk.
        register b1, b2, b3, bmask;
        b1 = _mm256_load_si256(&ch1[next]);
        b2 = _mm256_load_si256(&ch2[next]);
        b3 = _mm256_load_si256(&ch3[next]);
        bmask = _mm256_cmpgt_epi8(b1, _mm256_load_si256(pivots));
        ARGMSORT1X32(bmask, b1, b2, b3);

        // Sort across both chunks.
        ARGMSORT2X32(amask, a1, a2, a3, bmask, b1, b2, b3);

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
        uint32_t pivot = buf[pivot_idx];

        // Partition across our pivot.
        size_t mid = Partition(buf, size, pivot);

        if (k <= mid) {
            size = mid;
        } else if (k > mid) {
            buf = &buf[mid];
            size -= mid;
            k -= mid;
        }
    }

    return buf[0];
}
