/**
 * @file QSelect.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Exposes a QSelect and Partition function for unsigned arrays.
 * @bug No known bugs.
 */

#ifndef __QUICK_SELECT_H__
#define __QUICK_SELECT_H__

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Partitions the array across the given value.
 *
 * Note that this implementation ensures that all elements equal to the
 * given value will be at the boundary between the two partitions.
 *
 * @param buf The array to be partitioned.
 * @param size The size of the array.
 * @param pivot The value the be partitioned across.
 * @param plo The inclusive lower bound of the region equal to pivot.
 * @param phi The inclusive upper bound of the region equal to pivot.
 */
void MedianPartition(uint32_t *buf, size_t size,
                     uint32_t pivot, size_t *plo, size_t *phi);

/**
 * @brief Selects the kth sorted element in the given array.
 * @param buf The array to be searched for the kth element.
 * @param size The size of the array.
 * @param k The k index to search for.
 * @return The kth element in a sorted version of buf.
 */
uint32_t QSelect(uint32_t *buf, size_t size, size_t k);

#endif /* __QUICK_SELECT_H__ */
