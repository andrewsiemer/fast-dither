/**
 * @file MedianPartition.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Exposes a function to partition an array group across its median.
 * @bug No known bugs.
 */

#ifndef __MEDIAN_PARTITION_H__
#define __MEDIAN_PARTITION_H__

#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <MCQuantization.h>

/** @brief Holds the popcounts from the movemask within MedianPartition. */
typedef struct {
    uint8_t llcnt;
    uint8_t hlcnt;
    uint8_t lcnt;
    uint8_t hcnt;
} popcount_t;

/** @brief Holds temporary data between the phases of MedianPartition. */
typedef struct {
    uint32_t *counts;
    __m256i *s1;
    __m256i *s2;
    __m256i *s3;
} mp_workspace_t;

/**
 * @brief Initializes the given MP workspace.
 * @param ws The workspace to be initialized.
 * @param size The maximum size of the arrays being partitioned.
 */
void MPWorkspaceInit(mp_workspace_t *ws, size_t size);

/**
 * @brief Destroys the given MP workspace.
 */
void MPWorkspaceDestroy(mp_workspace_t *ws);

/**
 * @brief Partitions the first channel across its median, then arg-partitions
 *        the remaining two channels.
 * @param ch1 The channel to be partitioned.
 * @param ch2 The first channel to be arg-partitioned with ch1.
 * @param ch3 The second channel to be arg-partitioned with ch1.
 * @param size The size of the channels.
 * @param time The time tracking structure used to track time spent in the
 *             partition kernel.
 */
size_t MedianPartition(mp_workspace_t *ws,
                       uint8_t *ch1, uint8_t *ch2, uint8_t *ch3, size_t size,
                       mc_time_t *time);

#endif /* __MEDIAN_PARTITION_H__ */
