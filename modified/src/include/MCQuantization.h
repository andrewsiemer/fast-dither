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

MCWorkspace *MCWorkspaceMake(mc_byte_t level);
void MCWorkspaceDestroy(MCWorkspace *ws);
DTPalette *MCQuantizeData(SplitImage *img, MCWorkspace *ws);

#endif
