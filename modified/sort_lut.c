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
    printf("    { ");
    int lo = 0, hi = 7;
    for (unsigned int i = 0; i < 8; i++) {
        DELIM(i, ", ");

        if ((b >> i) & 1) {
            assert(hi >= lo);
            printf("%d", hi--);
        } else {
            assert(hi >= lo);
            printf("%d", lo++);
        }
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

int main(void) {
    printf("#include <stdint.h>\n");
    printf("\n");
    print_sort_x8();
    return 0;
}
