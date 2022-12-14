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
#include <UtilMacro.h>

typedef uint8_t mc_byte_t;
typedef unsigned int mc_uint_t;

typedef struct mc_workspace_t MCWorkspace;

typedef struct mc_time {
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
    unsigned long long sub_time;
    unsigned long long sub_units;
    unsigned long long split_time;
    unsigned long long split_units;
    unsigned long long full_time;
    unsigned long long full_units;
} mc_time_t;

/// @brief Times the given chunk of code to the given time structure.
#define MC_TIME(t, u, c)\
do {\
    unsigned long long ts1, ts2;\
    TIMESTAMP(ts1);\
    c\
    TIMESTAMP(ts2);\
    t##_time += (ts2 - ts1);\
    t##_units += u;\
} while (0)

MCWorkspace *MCWorkspaceMake(mc_byte_t level, size_t img_size);
void MCWorkspaceDestroy(MCWorkspace *ws);
DTPalette *MCQuantizeData(SplitImage *img, MCWorkspace *ws, mc_time_t *time);

void MCTimeInit(mc_time_t *time);
void MCTimeReport(mc_time_t *time);

#endif
