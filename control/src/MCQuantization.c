/*
 *  MCQuantization.c
 *  dither Utility
 *
 *  Quantization algorithm implementation.
 *
 */

#include <MCQuantization.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <XMalloc.h>
#include <UtilMacro.h>

#define NUM_DIM 3u

typedef struct {
    MCTriplet min;
    MCTriplet max;
    size_t size;
    MCTriplet *data;
} MCCube;

struct mc_workspace_t {
    mc_byte_t level;
    size_t p_size;
    MCCube *cubes;
    MCTriplet *palette;
};

/* dimension comparison priority least -> greatest */
static size_t dimOrder[NUM_DIM];

void MCShrinkCube(MCCube *cube);
MCTriplet MCCubeAverage(MCCube *cube);
void MCCalculateBiggestDimension(MCCube *cube);
int MCCompareTriplet(const void *t1, const void *t2);

MCTriplet
MCTripletMake(mc_byte_t r, mc_byte_t g, mc_byte_t b)
{
    MCTriplet triplet;
    triplet.value[0] = r;
    triplet.value[1] = g;
    triplet.value[2] = b;

    return triplet;
}

MCWorkspace *
MCWorkspaceMake(mc_byte_t level)
{
    MCWorkspace *ws = XMalloc(sizeof(MCWorkspace));
    ws->level = level;
    ws->p_size = 1 << level;
    ws->cubes = XMalloc(sizeof(MCCube) * ws->p_size);
    ws->palette = XMalloc(sizeof(MCTriplet) * ws->p_size);
    return ws;
}

void
MCWorkspaceDestroy(MCWorkspace *ws)
{
    free(ws->cubes);
    free(ws->palette);
    free(ws);
}

MCTriplet *
MCQuantizeData(MCTriplet *data, size_t size, MCWorkspace *ws)
{
    /* first cube */
    ws->cubes[0].data = data;
    ws->cubes[0].size = size;
    MCShrinkCube(ws->cubes);

    /* remaining cubes */
    size_t parentIndex = 0;
    int iLevel = 1; /* iteration level */
    size_t offset;
    MCCube *parentCube;
    while (iLevel <= ws->level)
    {
        parentCube = &ws->cubes[parentIndex];

        MCCalculateBiggestDimension(parentCube);
        qsort(parentCube->data, parentCube->size,
                sizeof(MCTriplet), MCCompareTriplet);

        /* Get median location */
        size_t mid = parentCube->size >> 1;
        offset = ws->p_size >> iLevel;

        /* split cubes */
        ws->cubes[parentIndex+offset] = *parentCube;

        /* newSize is now the index of the first element above the
         * median, thus it is also the count of elements below the median */
        ws->cubes[parentIndex        ].size = mid + 1;
        ws->cubes[parentIndex+offset].data += mid + 1;
        ws->cubes[parentIndex+offset].size -= mid + 1;

        /* shrink new cubes */
        MCShrinkCube(&ws->cubes[parentIndex]);
        MCShrinkCube(&ws->cubes[parentIndex+offset]);

        /* check if iLevel must be increased by analysing if the next
         * offset is within palette size boundary. If not, change level
         * and reset parent to 0. If it is, set next element as parent. */
        if (parentIndex + (offset * 2) < ws->p_size) {
            parentIndex = parentIndex + (offset * 2);
        } else {
            parentIndex = 0;
            iLevel++;
        }
    }

    /* find final cube averages */
    for (size_t i = 0; i < ws->p_size; i++)
        ws->palette[i] = MCCubeAverage(&ws->cubes[i]);

    MCTriplet *ret = ws->palette;
    ws->palette = NULL;
    return ret;
}

void
MCShrinkCube(MCCube *cube)
{
    mc_byte_t r, g, b;
    MCTriplet *data;

    data = cube->data;

    cube->min = MCTripletMake(0xFF, 0xFF, 0xFF);
    cube->max = MCTripletMake(0x00, 0x00, 0x00);

    for (size_t i = 0; i < cube->size; i++) {
        r = data[i].value[0];
        g = data[i].value[1];
        b = data[i].value[2];

        if (r < cube->min.value[0]) cube->min.value[0] = r;
        if (g < cube->min.value[1]) cube->min.value[1] = g;
        if (b < cube->min.value[2]) cube->min.value[2] = b;

        if (r > cube->max.value[0]) cube->max.value[0] = r;
        if (g > cube->max.value[1]) cube->max.value[1] = g;
        if (b > cube->max.value[2]) cube->max.value[2] = b;
    }
}

MCTriplet
MCCubeAverage(MCCube *cube)
{
    return MCTripletMake(
        (cube->max.value[0] + cube->min.value[0]) / 2,
        (cube->max.value[1] + cube->min.value[1]) / 2,
        (cube->max.value[2] + cube->min.value[2]) / 2
    );
}

void
MCCalculateBiggestDimension(MCCube *cube)
{
    MCTriplet diffs;

    for (size_t i = 0; i < NUM_DIM; i++) {
        diffs.value[i] = cube->max.value[i] - cube->min.value[i];
        dimOrder[i] = i;
    }

    for (size_t i = 0; i < NUM_DIM; i++) {
        for (size_t j = i + 1; j < NUM_DIM; j++) {
            if (diffs.value[i] > diffs.value[j]) {
                SWAP(diffs.value[i], diffs.value[j]);
                SWAP(dimOrder[i], dimOrder[j]);
            }
        }
    }
}

int
MCCompareTriplet(const void *a, const void *b)
{
    MCTriplet t1, t2;
    int lhs, rhs;

    t1 = * (MCTriplet *)a;
    t2 = * (MCTriplet *)b;
    lhs = t1.value[dimOrder[0]] | t1.value[dimOrder[1]] << 8u
        | t1.value[dimOrder[2]] << 16u;
    rhs = t2.value[dimOrder[0]] | t2.value[dimOrder[1]] << 8u
        | t2.value[dimOrder[2]] << 16u;

    return lhs - rhs;
}
