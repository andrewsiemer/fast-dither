/**
 * @file COrder.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Defines helper method for changing the byte order of colors.
 * @bug No known bugs.
 */

#ifndef __COLOR_ORDER_H__
#define __COLOR_ORDER_H__

#include <MCQuantization.h>
#include <stdint.h>

/**
 * @brief Color ordering enumeration.
 *
 * The bits are encoded as { R > G, R > B, G > B }. This allows us to
 * determine which byte order we want from a value range of each color
 * in the bucket during MC quantization.
 */
typedef enum {
    CO_RGB = 0x07,
    CO_RBG = 0x06,
    CO_GRB = 0x03,
    CO_GBR = 0x01,
    CO_BRG = 0x04,
    CO_BGR = 0x00,
    CO_ILL1 = 0x02,
    CO_ILL2 = 0x05,
    CO_COUNT = 8
} COrder;

// We store our pixels in a single 32-bit integer.
typedef uint32_t COrderPixel;

/**
 * @brief Finds the target byte order based on a triplet of value ranges.
 * @param order The current byte ordering of the MCTriplet.
 * @param diffs An MCTriplet containing the value range for each channel.
 * @return The ordering which should be assumed based on the ranges.
 */
COrder COFindTarget(COrder order, MCTriplet diffs);

/**
 * @brief Swaps a pixel array from one byte ordering to another.
 * @param from The current byte ordering of the array.
 * @param to The new byte ordering of the array.
 * @param buf The array to be swapped.
 * @param size The size of the array.
 */
void COSwapTo(COrder from, COrder to, COrderPixel *buf, size_t size);

#endif /* __COLOR_ORDER_H__ */
