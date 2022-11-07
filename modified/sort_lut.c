/**
 * @file sort_lut.c
 * @author Andrew Spaulding (aspauldi)
 * @brief Generates a look-up table for bitonic sort.
 * @bug No known bugs.
 */

#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static void print_sort(unsigned int b) {
    printf("    { ");
    int lo = 0, hi = 7;
    for (unsigned int i = 0; i < 8; i++) {
        if (i > 0) {
            printf(", ");
        }

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

int main(void) {
    printf("#include <stdint.h>\n");
    printf("const uint8_t sort1b_4x8[256][8] = {\n");
    for (unsigned int i = 0; i < 256; i++) {
        if (i > 0) {
            printf(",\n");
        }
        print_sort(i);
    }
    printf("\n};\n");
}
