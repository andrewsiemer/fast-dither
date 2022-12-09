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

#include <SplitImage.h>
#include <DTPalette.h>

typedef uint8_t mc_byte_t;
typedef unsigned int mc_uint_t;

typedef struct mc_workspace_t MCWorkspace;

typedef struct {
    unsigned long long shrink_time;
    unsigned long long part_time;
} mc_time_t;

MCWorkspace *MCWorkspaceMake(mc_byte_t level);
void MCWorkspaceDestroy(MCWorkspace *ws);
DTPalette *MCQuantizeData(SplitImage *img, MCWorkspace *ws, mc_time_t *time);

void MCTimeInit(mc_time_t *time);
void MCTimeReport(mc_time_t *time, unsigned long long runs);

#endif
