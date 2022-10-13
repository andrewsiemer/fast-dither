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

void
Partition(
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
    while (lo <= hi) {
        if ((mid_lo - lo) > 0) {
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
        } else if ((hi - mid_hi) > 0) {
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
        } else {
            // Center has been filled with pivot.
            break;
        }
    }

    // If we didn't find the pivot, yell.
    assert(mid_hi >= mid_lo);

    *plo = mid_lo;
    *phi = mid_hi;
}

uint32_t
QSelect(
    uint32_t *buf,
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
        size_t plo, phi;
        Partition(buf, size, pivot, &plo, &phi);

        if (k < plo) {
            size = plo;
        } else if (k > phi) {
            phi++;
            buf = &buf[phi];
            size -= phi;
            k -= phi;
        } else {
            // We got lucky :)
            return buf[k];
        }
    }

    return buf[0];
}
