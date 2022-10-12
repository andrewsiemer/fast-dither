/*
 *  MCQuantization.h
 *  dither Utility
 *
 *  Quantization algorithm declarations.
 *
 */

#ifndef MC_QUANTIZATION
#define MC_QUANTIZATION

#include <stdint.h>
#include <stddef.h>

typedef uint8_t mc_byte_t;
typedef unsigned int mc_uint_t;

typedef struct {
    mc_byte_t value[3];
    mc_byte_t pad;
} __attribute__((aligned (sizeof(uint32_t)))) MCTriplet;

MCTriplet MCTripletMake(mc_byte_t r, mc_byte_t g, mc_byte_t b);
MCTriplet *MCQuantizeData(MCTriplet *data, size_t size, mc_byte_t level);

#endif
