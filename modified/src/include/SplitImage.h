/**
 * @file SplitImage.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Splits an image packed at each pixel into separate r, g, and b images.
 * @bug No known bugs.
 */

#ifndef __SPLIT_IMAGE_H__
#define __SPLIT_IMAGE_H__

/**
 * @brief Holds an image which has been split by its colors.
 *
 * The structure super promisses that r, g, and b will be aligned to a 32-byte
 * bound.
 */
typedef struct {
    uint8_t *r;
    uint8_t *g;
    uint8_t *b;
    size_t w;
    size_t h;
} SplitImage;

// TODO: Functions for packing and unpacking images.

#endif /* __SPLIT_IMAGE_H__ */
