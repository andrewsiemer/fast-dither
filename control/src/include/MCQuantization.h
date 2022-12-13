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

typedef struct {
    unsigned long long shrink_time;
    unsigned long long shrink_units;
    unsigned long long sort_time;
    unsigned long long sort_units;
    unsigned long long mc_time;
    unsigned long long mc_units;
} mc_time_t;

MCTriplet MCTripletMake(mc_byte_t r, mc_byte_t g, mc_byte_t b);
MCWorkspace *MCWorkspaceMake(mc_byte_t level);
void MCWorkspaceDestroy(MCWorkspace *ws);
MCTriplet *MCQuantizeData(MCTriplet *data, size_t size, MCWorkspace *ws, mc_time_t *time);

void MCTimeInit(mc_time_t *time);
void MCTimeReport(mc_time_t *time);

#endif
