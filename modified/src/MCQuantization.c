/*
 *  MCQuantization.c
 *  dither Utility
 *
 *  Quantization algorithm implementation.
 *
 */

#include <MCQuantization.h>
#include <COrder.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define NUM_DIM 3u

typedef struct {
    MCTriplet min;
    MCTriplet max;
    COrder order;
    size_t size;
    MCTriplet *data;
} MCCube;

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
    triplet.pad = 0;

    return triplet;
}

MCTriplet *
MCQuantizeData(MCTriplet *data, size_t size, mc_byte_t level)
{
    size_t p_size; /* generated palette size */
    MCCube *cubes;
    MCTriplet *palette;

    p_size  = 1 << level;
    cubes   = malloc(sizeof(MCCube) * p_size);
    palette = malloc(sizeof(MCTriplet) * p_size);

    /* first cube */
    cubes[0].order = CO_RGB;
    cubes[0].data = data;
    cubes[0].size = size;
    MCShrinkCube(cubes);

    /* remaining cubes */
    size_t parentIndex = 0;
    int iLevel = 1; /* iteration level */
    size_t offset;
    MCCube *parentCube;
    while (iLevel <= level)
    {
        parentCube = &cubes[parentIndex];

        MCCalculateBiggestDimension(parentCube);
        qsort(parentCube->data, parentCube->size,
                sizeof(MCTriplet), MCCompareTriplet);

        /* Get median location */
        size_t mid = parentCube->size >> 1;
        offset = p_size >> iLevel;

        /* split cubes */
        cubes[parentIndex+offset] = *parentCube;

        /* newSize is now the index of the first element above the
         * median, thus it is also the count of elements below the median */
        cubes[parentIndex].size = mid + 1;
        cubes[parentIndex+offset].data += mid + 1;
        cubes[parentIndex+offset].size -= mid + 1;

        /* shrink new cubes */
        MCShrinkCube(&cubes[parentIndex]);
        MCShrinkCube(&cubes[parentIndex+offset]);

        /* check if iLevel must be increased by analysing if the next
         * offset is within palette size boundary. If not, change level
         * and reset parent to 0. If it is, set next element as parent. */
        if (parentIndex + (offset * 2) < p_size) {
            parentIndex = parentIndex + (offset * 2);
        } else {
            parentIndex = 0;
            iLevel++;
        }
    }

    /* find final cube averages */
    for (size_t i = 0; i < p_size; i++)
        palette[i] = MCCubeAverage(&cubes[i]);

    free(cubes);

    printf("Palette:\n");
    for (size_t i = 0; i < p_size; i++)
        printf("%zu: (%u, %u, %u)\n", i, palette[i].value[0], palette[i].value[1], palette[i].value[2]);
    return palette;
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
    COSwapTo(cube->order, CO_RGB, (COrderPixel*) &cube->min, 1);
    COSwapTo(cube->order, CO_RGB, (COrderPixel*) &cube->max, 1);

    return MCTripletMake(
        (cube->max.value[0] + cube->min.value[0]) / 2,
        (cube->max.value[1] + cube->min.value[1]) / 2,
        (cube->max.value[2] + cube->min.value[2]) / 2
    );
}

void
MCCalculateBiggestDimension(MCCube *cube)
{
    MCTriplet diffs = { .pad = 0 };
    COrder new;

    for (size_t i = 0; i < NUM_DIM; i++) {
        diffs.value[i] = cube->max.value[i] - cube->min.value[i];
    }

    new = COFindTarget(cube->order, diffs);
    COSwapTo(cube->order, new, (COrderPixel*) cube->data, cube->size);
    cube->order = new;
}

int
MCCompareTriplet(const void *a, const void *b)
{
    return (*(int32_t*)a) - (*(int32_t*)b);
}
