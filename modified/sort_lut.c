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

static void print_sort_x16(void) {
    printf("static const uint8_t sort1b_2x16[9][9][16] = {\n");
    for (unsigned int i = 0; i < 9; i++) {
        DELIM(i, ",\n");

        printf("    {\n");
        for (unsigned int j = 0; j < 9; j++) {
            DELIM(j, ",\n");

            printf("        { ");
            for (unsigned int k = 0; k < 16; k++) {
                DELIM(k, ", ");

                if (k < (8 - i)) {
                    printf("%u", k);
                } else if (k < ((8 - i) + (8 - j))) {
                    printf("%u", 8 + (k - (8 - i)));
                } else if (k < ((8 - j) + 8)) {
                    printf("%u", k - (8 - j));
                } else {
                    printf("%u", k);
                }
            }
            printf(" }");
        }
        printf("\n    }");
    }
    printf("\n};\n");
}

static void print_roll_left_x16(void) {
    printf("static const uint8_t roll_left_16x[16][16] = {\n");
    for (unsigned int i = 0; i < 16; i++) {
        DELIM(i, ",\n");
        printf("    { ");
        for (unsigned int j = 0; j < 16; j++) {
            DELIM(j, ", ");
            printf("%u", (j + i) & 0xF);
        }
        printf(" }");
    }
    printf("\n};\n");
}

static void print_roll_left_x32(void) {
    printf("static const uint8_t roll_left_32x[16][16] = {\n");
    for (unsigned int i = 0; i < 16; i++) {
        DELIM(i, ",\n");
        printf("    { ");
        for (unsigned int j = 0; j < 16; j++) {
            DELIM(j, ", ");
            printf("0x%02x", (j < (16 - i)) ? 0x00 : 0xff);
        }
        printf(" }");
    }
    printf("\n};\n");
}

int main(void) {
    printf("#include <stdint.h>\n");
    printf("\n");
    print_sort_x8();
    printf("\n");
    print_sort_x16();
    printf("\n");
    print_roll_left_x16();
    printf("\n");
    print_roll_left_x32();
    return 0;
}
