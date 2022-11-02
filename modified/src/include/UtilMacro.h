/**
 * @file UtilMacro.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Utility macros shared across multiple files.
 * @bug No known bugs.
 */

#ifndef __UTIL_MACRO_H__
#define __UTIL_MACRO_H__

#include <stdio.h>

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4

// Swaps two integral values.
#define SWAP(a, b)\
do {\
    __typeof__(a) _tmp = (a);\
    (a) = (b);\
    (b) = _tmp;\
} while (0)

// Outputs a timestamp into the given unsigned long long.
#define TIMESTAMP(ts)\
do {\
    unsigned _hi, _lo;\
    __asm__ __volatile__ ("rdtsc" : "=a" (_lo), "=d" (_hi));\
    ts = ((unsigned long long)_lo) | (((unsigned long long)_hi) << 32);\
} while (0)

// Reports the time of a given test based on the given stamps and test name.
#define TIME_REPORT(s, ts1, ts2)\
    printf("%lf,\n",\
        (ts2 - ts1) * (MAX_FREQ/BASE_FREQ))

#endif /* __UTIL_MACRO_H__ */
