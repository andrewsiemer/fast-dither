/**
 * @file CompilerGoop.h
 * @author Andrew Spaulding (aspauldi)
 * @brief Compilerisms.
 * @bug No known bugs.
 */

#ifndef __COMPILER_GOOP_H__
#define __COMPILER_GOOP_H__

/// @brief Aligns a type to the given alignment.
#define align(x) __attribute__((aligned(x)))

/// @brief Generates a compiler barrier.
#define COMPILER_BARRIER __asm__ __volatile__("" : : : "memory")

#endif /* __COMPILER_GOOP_H__ */
