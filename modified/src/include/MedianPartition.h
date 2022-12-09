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

#include <MCQuantization.h>

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
size_t MedianPartition(uint8_t *ch1, uint8_t *ch2, uint8_t *ch3, size_t size,
                       mc_time_t *time);

#endif /* __MEDIAN_PARTITION_H__ */
