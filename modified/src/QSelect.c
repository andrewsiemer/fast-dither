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

#include <UtilMacro.h>

void
Partition(
    uint32_t *buf,
    size_t size,
    uint32_t pivot,
    size_t *lo_end,
    size_t *hi_start
) {
    assert(buf);
    assert(size > 0);
    assert(lo_end);
    assert(hi_start);

    size_t lo = 0, hi = size - 1;
    size_t mid_lo = hi >> 1, mid_hi = hi >> 1;
    while (lo < hi) {
        if (lo < mid_lo) {
            if (buf[lo] < pivot) {
                lo++;
            } else if (buf[lo] > pivot) {
                SWAP(buf[lo], buf[hi]);
                if (hi-- < mid_hi) {
                    // Fix mid_hi.
                    // This means that buf[lo] == pivot at this point.
                    mid_hi--;
                    assert(hi == (mid_hi - 1));
                }
            } else {
                // Exact match, move middle.
                mid_lo--;
                SWAP(buf[lo], buf[mid_lo]);
            }
        } else if (hi >= mid_hi) {
            if (buf[hi] > pivot) {
                hi--;
            } else if (buf[hi] < pivot) {
                SWAP(buf[lo], buf[hi]);
                lo++;
                // If we're here, lo == mid_lo, so we need to move mid_lo up.
                // This means the middle loses an element, but we'll fix
                // it on the next iteration.
                mid_lo++;
            } else {
                SWAP(buf[hi], buf[mid_hi]);
                mid_hi++;
            }
        } else {
            // Center has been filled with pivot.
            break;
        }
    }

    assert(lo == mid_lo);
    assert(hi == (mid_hi - 1));
    assert(mid_hi >= mid_lo);
    *lo_end = mid_lo - 1;
    *hi_start = mid_hi;
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

    // Base case
    if (size == 1) {
        return buf[0];
    }

    // Get our pivot. Random is "good enough" for O(n) in most cases.
    size_t pivot_idx = ((size_t) rand()) % size;
    uint32_t pivot = buf[pivot_idx];

    // Partition across our pivot.
    size_t lo_end, hi_start;
    Partition(buf, size, pivot, &lo_end, &hi_start);

    if (k <= lo_end) {
        return QSelect(buf, lo_end + 1, k);
    } else if (k >= hi_start) {
        return QSelect(&buf[hi_start], size - hi_start, k - hi_start);
    } else {
        // We got lucky :)
        return buf[k];
    }
}
