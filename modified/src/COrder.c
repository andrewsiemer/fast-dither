/**
 * @file COrder.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Implementation of byte swapping functions.
 * @bug No known bugs.
 *
 * These functions are called during MCQuantization to speed up the sorting
 * of the array by allowing simple comparison instructions to be used and by
 * ensuring that any byte swapping occurs with the maximum throughput of the
 * system.
 */

#include <COrder.h>
#include <MCQuantization.h>
#include <stdint.h>
#include <stdlib.h>

//The number of color channels we support.
#define COLOR_CHANNELS 3

// The different types of byte swaps which may be necessary to
// fix an array.
typedef enum {
    SWAP_ILL = 0,
    SWAP_123,
    SWAP_132,
    SWAP_213,
    SWAP_321,
    SWAP_231,
    SWAP_312
} COrderSwap;

// Determines how a swap should be performed to reach the needed byte order.
static int8_t swapTable[CO_COUNT][CO_COUNT] = {
    [CO_RGB] = {
        [CO_RGB] = SWAP_123,
        [CO_RBG] = SWAP_132,
        [CO_GRB] = SWAP_213,
        [CO_GBR] = SWAP_231,
        [CO_BRG] = SWAP_312,
        [CO_BGR] = SWAP_321
    },
    [CO_RBG] = {
        [CO_RGB] = SWAP_132,
        [CO_RBG] = SWAP_123,
        [CO_GRB] = SWAP_312,
        [CO_GBR] = SWAP_321,
        [CO_BRG] = SWAP_213,
        [CO_BGR] = SWAP_231
    },
    [CO_GRB] = {
        [CO_RGB] = SWAP_213,
        [CO_RBG] = SWAP_231,
        [CO_GRB] = SWAP_123,
        [CO_GBR] = SWAP_132,
        [CO_BRG] = SWAP_321,
        [CO_BGR] = SWAP_312
    },
    [CO_GBR] = {
        [CO_RGB] = SWAP_312,
        [CO_RBG] = SWAP_321,
        [CO_GRB] = SWAP_132,
        [CO_GBR] = SWAP_123,
        [CO_BRG] = SWAP_231,
        [CO_BGR] = SWAP_213
    },
    [CO_BRG] = {
        [CO_RGB] = SWAP_231,
        [CO_RBG] = SWAP_213,
        [CO_GRB] = SWAP_321,
        [CO_GBR] = SWAP_312,
        [CO_BRG] = SWAP_123,
        [CO_BGR] = SWAP_132
    },
    [CO_BGR] = {
        [CO_RGB] = SWAP_321,
        [CO_RBG] = SWAP_312,
        [CO_GRB] = SWAP_231,
        [CO_GBR] = SWAP_213,
        [CO_BRG] = SWAP_132,
        [CO_BGR] = SWAP_123
    }
};

// Gets the index ordering needed to access a triplet in rgb order.
static uint8_t RGBIndex[CO_COUNT][COLOR_CHANNELS] = {
    [CO_RGB] = { 0, 1, 2 },
    [CO_RBG] = { 0, 2, 1 },
    [CO_GRB] = { 1, 0, 2 },
    [CO_GBR] = { 2, 0, 1 },
    [CO_BRG] = { 1, 2, 0 },
    [CO_BGR] = { 2, 1, 0 }
};

// Swaps two integral values.
#define SWAP(a, b) do { (a) ^= (b); (b) ^= (a); (a) ^= (b); } while (0)

// Uses the RGBIndex array to get a color channel in an MC triplet.
#define GET_R(o, mc) ((int8_t) (mc).value[RGBIndex[o][0]])
#define GET_G(o, mc) ((int8_t) (mc).value[RGBIndex[o][1]])
#define GET_B(o, mc) ((int8_t) (mc).value[RGBIndex[o][2]])

// Swaps the bytes in a pixel.
#define SWIZZLE(p, b1, b2, b3)\
    (((((p) >> (8 * (b1 - 1))) & 0xFF) << 0) |\
     ((((p) >> (8 * (b2 - 1))) & 0xFF) << 8) |\
     ((((p) >> (8 * (b3 - 1))) & 0xFF) << 16))

/* Helper functions */
static inline COrderSwap COLookupSwap(COrder from, COrder to);
static void COSwap132(COrderPixel *buf, size_t size);
static void COSwap213(COrderPixel *buf, size_t size);
static void COSwap321(COrderPixel *buf, size_t size);
static void COSwap231(COrderPixel *buf, size_t size);
static void COSwap312(COrderPixel *buf, size_t size);

COrder
COFindTarget(
    COrder order,
    MCTriplet diffs
) {
    // Argsort the array, starting with an index vector that is the argsort
    // of rgb into our current format.
    uint8_t idx[COLOR_CHANNELS];
    for (size_t i = 0; i < COLOR_CHANNELS; i++) idx[i] = RGBIndex[order][i];
    for (size_t i = 0; i < COLOR_CHANNELS; i++) {
        for (size_t j = i + 1; j < COLOR_CHANNELS; j++) {
            if (diffs.value[i] > diffs.value[j]) {
                SWAP(diffs.value[i], diffs.value[j]);
                SWAP(idx[i], idx[j]);
            }
        }
    }

    // Use the index vector to determine which enum we should use.
    int rGTg = (idx[0] == 1) || ((idx[0] > 0) && idx[1] > 0);
    int rGTb = (idx[0] == 2) || ((idx[0] == 1) && idx[1] == 2);
    int gGTb = (idx[0] == 1) || ((idx[0] == 0) && idx[1] == 1);

    return (COrder) ((rGTg << 2) | (rGTb << 1) | gGTb);
}

void
COSwapTo(
    COrder from,
    COrder to,
    COrderPixel *buf,
    size_t size
) {
    COrderSwap target = COLookupSwap(from, to);
    switch (target) {
        case SWAP_123: break;
        case SWAP_132: COSwap132(buf, size); break;
        case SWAP_213: COSwap213(buf, size); break;
        case SWAP_321: COSwap321(buf, size); break;
        case SWAP_231: COSwap231(buf, size); break;
        case SWAP_312: COSwap312(buf, size); break;
        case SWAP_ILL:
        default:
            abort();
    }
}

static inline COrderSwap
COLookupSwap(
    COrder from,
    COrder to
) {
    return (COrderSwap) swapTable[from][to];
}

static void
COSwap132(
    COrderPixel *buf,
    size_t size
) {
    while (size--) {
        *buf = SWIZZLE(*buf, 1, 3, 2);
        buf++;
    }
}

static void
COSwap213(
    COrderPixel *buf,
    size_t size
) {
    while (size--) {
        *buf = SWIZZLE(*buf, 2, 1, 3);
        buf++;
    }
}

static void
COSwap321(
    COrderPixel *buf,
    size_t size
) {
    while (size--) {
        *buf = SWIZZLE(*buf, 3, 2, 1);
        buf++;
    }
}

static void
COSwap231(
    COrderPixel *buf,
    size_t size
) {
    while (size--) {
        *buf = SWIZZLE(*buf, 2, 3, 1);
        buf++;
    }
}

static void
COSwap312(
    COrderPixel *buf,
    size_t size
) {
    while (size--) {
        *buf = SWIZZLE(*buf, 3, 1, 2);
        buf++;
    }
}

