/**
 * @file UtilMacro.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Utility macros shared across multiple files.
 * @bug No known bugs.
 */

#ifndef __UTIL_MACRO_H__
#define __UTIL_MACRO_H__

// Swaps two integral values.
#define SWAP(a, b)\
do {\
    __typeof__(a) _tmp = (a);\
    (a) = (b);\
    (b) = _tmp;\
} while (0)

#endif /* __UTIL_MACRO_H__ */
