/**
 * @file sort_lut.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Generates a look-up table for bitonic sort.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>
#include <stdio.h>

#define DELIM(it, s)\
do {\
    if ((it) > 0) {\
        printf(s);\
    }\
} while (0)

static void print_sort_elem_x8(unsigned int b) {
    unsigned int arr[8] = {0};
    int lo = 0, hi = 7;
    for (unsigned int i = 0; i < 8; i++) {
        if ((b >> i) & 1) {
            arr[hi--] = i;
        } else {
            arr[lo++] = i;
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
    printf("static const uint8_t sort1b_4x8[256][8] = {\n");
    for (unsigned int i = 0; i < 256; i++) {
        DELIM(i, ",\n");
        print_sort_elem_x8(i);
    }
    printf("\n};\n");
}

static void print_srl_blend_elem(unsigned int shift) {
    unsigned int pre = 16 - shift;
    int first = 1;

    printf("{");
    for (unsigned int i = 0; i < pre; i++) {
        if (!first) { printf(", "); }
        first = 0;
        printf("0x00");
    }
    for (unsigned int i = 0; i < 16; i++) {
        if (!first) { printf(", "); }
        first = 0;
        printf("0xFF");
    }
    for (unsigned int i = 0; i < shift; i++) {
        if (!first) { printf(", "); }
        first = 0;
        printf("0x00");
    }
    printf("}\n");
}

static void print_srl_blend(void) {
    printf("__attribute__((aligned(32))) static const uint8_t srl_blend[17][32] = {\n");
    for (unsigned int i = 0; i <= 16; i++) {
        DELIM(i, ",\n");
        print_srl_blend_elem(i);
    }
    printf("\n};\n");
}

int main(void) {
    printf("#include <stdint.h>\n");
    printf("\n");
    print_sort_x8();
    printf("\n");
    print_srl_blend();
    return 0;
}
