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
} MCTriplet;

typedef struct mc_workspace_t MCWorkspace;
extern unsigned long long shrink_pixels, shrink_cycles;

MCTriplet MCTripletMake(mc_byte_t r, mc_byte_t g, mc_byte_t b);
MCWorkspace *MCWorkspaceMake(mc_byte_t level);
void MCWorkspaceDestroy(MCWorkspace *ws);
MCTriplet *MCQuantizeData(MCTriplet *data, size_t size, MCWorkspace *ws);

#endif
