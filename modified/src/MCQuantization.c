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
#include <assert.h>
#include <math.h>

#include <COrder.h>
#include <QSelect.h>
#include <XMalloc.h>

#define NUM_DIM 3u

typedef struct {
    MCTriplet min;
    MCTriplet max;
    COrder order;
    size_t size;
    MCTriplet *data;
} MCCube;

typedef struct mc_workspace_t {
    mc_byte_t level;
    size_t p_size;
    MCCube *cubes;
    MCTriplet *palette;
};

static void MCShrinkCube(MCCube *cube);
static void MCCalculateBiggestDimension(MCCube *cube);
static void MCSplit(MCCube *lo, MCCube *hi);
static MCTriplet MCCubeAverage(MCCube *cube);

MCTriplet
MCTripletMake(
    mc_byte_t r,
    mc_byte_t g,
    mc_byte_t b
) {
    MCTriplet triplet;
    triplet.value[0] = r;
    triplet.value[1] = g;
    triplet.value[2] = b;
    triplet.pad = 0;

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
}

void
MCWorkspaceDestroy(MCWorkspace *ws)
{
    free(ws->cubes);
    free(ws->palette);
    free(ws);
}

MCTriplet *
MCQuantizeData(
    MCTriplet *data,
    size_t size,
    MCWorkspace *ws
) {
    /* first cube */
    ws->cubes[0].order = CO_RGB;
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

        // Change the byte ordering based on the dimension priority.
        MCCalculateBiggestDimension(parentCube);

        // Partition the cube across the median.
        offset = ws->p_size >> iLevel;
        MCSplit(parentCube, &ws->cubes[parentIndex + offset]);

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
    for (size_t i = 0; i < p_size; i++)
        ws->palette[i] = MCCubeAverage(&ws->cubes[i]);

    MCTriplet *ret = ws->palette;
    ws->palette = NULL;
    return ret;
}

static void
MCShrinkCube(
    MCCube *cube
) {
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

static void
MCCalculateBiggestDimension(
    MCCube *cube
) {
    MCTriplet diffs = { .pad = 0 };
    COrder new;

    for (size_t i = 0; i < NUM_DIM; i++) {
        diffs.value[i] = cube->max.value[i] - cube->min.value[i];
    }

    new = COFindTarget(cube->order, diffs);
    COSwapTo(cube->order, new, (COrderPixel*) cube->data, cube->size);
    cube->order = new;
}

/**
 * @brief Splits lo across its median into [lo, hi].
 * @param lo The cube to be split and the output lo half.
 * @param hi The output hi half.
 */
static void
MCSplit(
    MCCube *lo,
    MCCube *hi
) {
    assert(lo);
    assert(hi);

    // Find the median.
    size_t mid = lo->size >> 1;
    uint32_t median = QSelect((uint32_t*) lo->data, lo->size, mid);

    // Partition across the median.
    size_t plo, phi;
    Partition((uint32_t*) lo->data, lo->size, median, &plo, &phi);
    assert((plo <= mid) && (mid <= phi));

    // Split the cubes by size.
    *hi = *lo;
    lo->size = mid + 1;
    hi->data += lo->size;
    hi->size -= lo->size;

    // Shrink the value range of the cubes.
    MCShrinkCube(lo);
    MCShrinkCube(hi);
}

static MCTriplet
MCCubeAverage(
    MCCube *cube
) {
    COSwapTo(cube->order, CO_RGB, (COrderPixel*) &cube->min, 1);
    COSwapTo(cube->order, CO_RGB, (COrderPixel*) &cube->max, 1);

    return MCTripletMake(
        (cube->max.value[0] + cube->min.value[0]) / 2,
        (cube->max.value[1] + cube->min.value[1]) / 2,
        (cube->max.value[2] + cube->min.value[2]) / 2
    );
}
