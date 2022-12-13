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
    unsigned long long shrink_units;
    unsigned long long part_time;
    unsigned long long part_units;
    unsigned long long mid_time;
    unsigned long long mid_units;
    unsigned long long mc_time;
    unsigned long long mc_units;
    unsigned long long align_time;
    unsigned long long align_units;
    unsigned long long dc_time;
    unsigned long long dc_units;
    unsigned long long sub_time;
    unsigned long long sub_units;
    unsigned long long full_time;
    unsigned long long full_units;
} mc_time_t;

MCWorkspace *MCWorkspaceMake(mc_byte_t level, size_t img_size);
void MCWorkspaceDestroy(MCWorkspace *ws);
DTPalette *MCQuantizeData(SplitImage *img, MCWorkspace *ws, mc_time_t *time);

void MCTimeInit(mc_time_t *time);
void MCTimeReport(mc_time_t *time);

#endif
