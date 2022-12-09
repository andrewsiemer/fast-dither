/**
 * @file sort_lut.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Generates a look-up table for bitonic sort.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

#define DELIM(it, s)\
do {\
    if ((it) > 0) {\
        printf(s);\
    }\
} while (0)

static void print_sort_elem_x8(unsigned int b) {
    unsigned int arr[8] = {0};

    if ((b >= 3) && ((b-3) < 256)) {
        b -= 3;
        int lo = 0, hi = 7;
        for (unsigned int i = 0; i < 8; i++) {
            if ((b >> i) & 1) {
                arr[hi--] = i;
            } else {
                arr[lo++] = i;
            }
        }
    }

    printf("    { ");
    for (unsigned int i = 0; i < 8; i++) {
        DELIM(i, ", ");
        printf("%u", arr[i]);
    }
    printf(" }");
}

static void print_sort_x8(void) {
    printf("__attribute__((aligned(32))) static const uint8_t sort1b_4x8[262][8] = {\n");
    for (unsigned int i = 0; i < 262; i++) {
        DELIM(i, ",\n");
        print_sort_elem_x8(i);
    }
    printf("\n};\n");
}

static void print_srl_blend_elem(unsigned int shift) {
    unsigned int mask = 0xFFFF0000;
    mask = (mask >> shift) | (mask << (32 - shift));

    printf("    { ");
    for (unsigned int i = 0; i < 32; i++) {
        DELIM(i, ", ");
        bool hi = ((mask >> i) & 1) ^ (i < 16);
        if (hi) {
            printf("255");
        } else {
            printf("0");
        }
    }
    printf(" }");
}

static void print_srl_blend(void) {
    printf("__attribute__((aligned(32))) static const uint8_t srl_blend[33][32] = {\n");
    for (unsigned int i = 0; i <= 32; i++) {
        DELIM(i, ",\n");
        print_srl_blend_elem(i);
    }
    printf("\n};\n");
}

static void print_shifted_set_mask_elem(unsigned int shift) {
    int first = 1;

    printf("    { ");
    for (unsigned int i = 0; i < (32 - shift); i++) {
        if (!first) { printf(", "); }
        first = 0;
        printf("255");
    }
    for (unsigned int i = 0; i < shift; i++) {
        if (!first) { printf(", "); }
        first = 0;
        printf("0");
    }
    printf(" }");
}

static void print_shifted_set_mask(void) {
    printf("__attribute__((aligned(32))) static const uint8_t shifted_set_mask[33][32] = {\n");
    for (unsigned int i = 0; i <= 32; i++) {
        DELIM(i, ",\n");
        print_shifted_set_mask_elem(i);
    }
    printf("\n};\n");
}

int main(void) {
    printf("#include <stdint.h>\n");
    printf("\n");
    print_sort_x8();
    printf("\n");
    print_srl_blend();
    printf("\n");
    print_shifted_set_mask();
    return 0;
}
