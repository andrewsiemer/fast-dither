/*
 *  DTDither.c
 *  dither Utility
 *
 *  Dithering algorithm Implementation.
 *
 */

#include <DTDither.h>
#include <UtilMacro.h>

#include <stdint.h>
#include <immintrin.h>

#include <stdio.h>
#include <string.h>

/*
void PRINT_M256_EPI16(__m256i input)
{
    int16_t *temp_array;
    posix_memalign((void**) &temp_array, 64, 16 * sizeof(int16_t));
    _mm256_store_si256((__m256i*) temp_array, input);
    for (size_t i = 0; i < 16; i++)
        printf("%d ", temp_array[i]);
    printf("\n");
    free(temp_array);
}
*/

inline int16_t fsdither_kernel_scalar(int16_t diff_3, int16_t diff_2, int16_t diff_1, int16_t diff_0,
                                   int16_t original)
{
    diff_3 = diff_3 / 16;
    diff_2 = diff_2 * 5 / 16;
    diff_1 = diff_1 * 3 / 16;
    diff_0 = diff_0 * 7 / 16;
    return original + (diff_3 + diff_2 + diff_1 + diff_0);
}

/**
 * 
 * 
 */
static void fsdither_kernel_simd(int16_t *input, // Array A
                          int16_t *palette, // Array B
                          int16_t *output, // Array B
                          int16_t *offset // Array B
                          )
{
    __m256i diff_cols[3];
    __m256i diff_offset;

    // Data loading scope
    {
        size_t i;
        for (i = 0; i < 3; i++)
        {
            __m256i original = _mm256_load_si256((__m256i*) &input[i*16]);
            __m256i searched = _mm256_load_si256((__m256i*) &palette[i*16]);
            diff_cols[i] = _mm256_sub_epi16(original, searched);
        }

        diff_offset = _mm256_load_si256((__m256i*) &palette[i*16]);
    }

    __m256i diff_output;
    
    // Multiplications and Divisions
    {
        __m256i scalar_7 = _mm256_set1_epi16(7);
        __m256i scalar_5 = _mm256_set1_epi16(5);
        __m256i scalar_3 = _mm256_set1_epi16(3);
        __m256i wake_diff;

        // Calculate the 7/16 term
        wake_diff = _mm256_mullo_epi16(diff_cols[2], scalar_7);
        wake_diff = _mm256_srai_epi16(wake_diff, 4);
        {
            // Pop first 16b int and add to offset
            __m256i temp = _mm256_setzero_si256();
            temp = _mm256_blend_epi16(temp, wake_diff, 0x01);
            // Eliminate other 128b
            diff_offset = _mm256_add_epi16(diff_offset, temp);

            // Shift wake-diff (7/16) bits
            // Create temp with swap lanes
            temp = _mm256_permute2x128_si256(wake_diff, wake_diff, 0x81); 
            // Left-shift temp all the way to the other side of the lane
            temp = _mm256_bslli_epi128(temp, 16-2); 
            // Pop top 16b from both lanes
            wake_diff = _mm256_bsrli_epi128(wake_diff, 2); 
            // Re-add inner lost 16b
            wake_diff = _mm256_blend_epi16(wake_diff, temp, 0x80); 
        }

        // Calculate the 3/16 term
        diff_cols[2] = _mm256_mullo_epi16(diff_cols[2], scalar_3);
        diff_cols[2] = _mm256_srai_epi16(diff_cols[2], 4);
        diff_output = _mm256_add_epi16(wake_diff, diff_cols[2]);

        // Calculate the 5/16 term
        diff_cols[1] = _mm256_mullo_epi16(diff_cols[1], scalar_5);
        diff_cols[1] = _mm256_srai_epi16(diff_cols[1], 4);
        diff_output = _mm256_add_epi16(diff_output, diff_cols[1]);

        // Calculate the 1/16 term
        diff_cols[0] = _mm256_srai_epi16(diff_cols[0], 4);
        diff_output = _mm256_add_epi16(diff_output, diff_cols[0]);
    }
    
    // Data storing scope
    {
        // Finish shift
        // Left-shift temp all the way to the other side of the lane
        __m256i offset_output = _mm256_bsrli_epi128(diff_output, 16-2); 
        // Create temp with swap lanes
        __m256i temp = _mm256_permute2x128_si256(offset_output, diff_offset, 0x02); 
        offset_output = _mm256_permute2x128_si256(offset_output, offset_output, 0x01);

        // Pop top 16b from both lanes
        diff_output = _mm256_bslli_epi128(diff_output, 2); 
        // Re-add inner lost 16b and offset 16b
        diff_output = _mm256_blend_epi16(diff_output, temp, 0x01);
        
        // Store output
        _mm256_store_si256((__m256i*) output, diff_output);

        // Store lone 16b int
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);
        _mm256_maskstore_epi32((int32_t*) offset, mask, offset_output);
    }
}

/**
 * 
 */
static void fsdither_runner(int16_t *shifted_input, int16_t *shifted_output, size_t width, size_t height,
                     DTPalettePacked *palette)
{
    unsigned long color_size = height * width * sizeof(int16_t);
    int16_t throwaway[16];
    
    for (size_t i = 0; i < height / 16; i++)
    {
        // Constant calculation
        const size_t offset = i*16*width;

        // Startup
        for (size_t j = 0; j < 3; j++) //TODO: min
        {
            DTPixelDiff input;
            DTPixel output;
            switch (j)
            {
                // TODO: Fix with input
                case 0:
                    for (size_t k = 0; k < 3; k++)
                        shifted_output[offset+color_size*k] = shifted_input[offset+color_size*k];
                    input.r = (byte) MAX(MIN(shifted_output[offset+color_size*0], 255), 0);
                    input.g = (byte) MAX(MIN(shifted_output[offset+color_size*1], 255), 0);
                    input.b = (byte) MAX(MIN(shifted_output[offset+color_size*2], 255), 0);
                    output = FindClosestColorFromPalette(input, palette);
                    shifted_output[offset+color_size*0] = output.r;
                    shifted_output[offset+color_size*1] = output.g;
                    shifted_output[offset+color_size*2] = output.b;
                    break;
                case 1:
                    for (size_t k = 0; k < 3; k++)
                        shifted_output[offset+16+color_size*k] = fsdither_kernel_scalar(0, 0, 0, 
                            shifted_output[offset+color_size*k]-shifted_input[offset+color_size*k], 
                            shifted_input[offset+16+color_size*k]);
                    input.r = (byte) MAX(MIN(shifted_output[offset+16+color_size*0], 255), 0);
                    input.g = (byte) MAX(MIN(shifted_output[offset+16+color_size*1], 255), 0);
                    input.b = (byte) MAX(MIN(shifted_output[offset+16+color_size*2], 255), 0);
                    output = FindClosestColorFromPalette(input, palette);
                    shifted_output[offset+16+color_size*0] = output.r;
                    shifted_output[offset+16+color_size*1] = output.g;
                    shifted_output[offset+16+color_size*2] = output.b;
                    break;
                case 2:
                default:
                    for (size_t k = 0; k < 3; k++) {
                        shifted_output[offset+32] = fsdither_kernel_scalar(0, 0, 0, 
                            shifted_output[offset+16]-shifted_input[offset+16], shifted_input[offset+32]);
                        shifted_output[offset+33] = fsdither_kernel_scalar(shifted_output[offset+16]-shifted_input[offset+16],
                            0, 0, 0, shifted_input[offset+33]);
                    }
                    for (size_t k = 0; k < 1; k++) {
                        input.r = (byte) MAX(MIN(shifted_output[offset+32+k+color_size*0], 255), 0);
                        input.g = (byte) MAX(MIN(shifted_output[offset+32+k+color_size*1], 255), 0);
                        input.b = (byte) MAX(MIN(shifted_output[offset+32+k+color_size*2], 255), 0);
                        output = FindClosestColorFromPalette(input, palette);
                        shifted_output[offset+32+k+color_size*0] = output.r;
                        shifted_output[offset+32+k+color_size*1] = output.g;
                        shifted_output[offset+32+k+color_size*2] = output.b;
                    }
                    break;
            }   
        }

        // Steady-State
        for (size_t j = 3; j < width; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                int16_t* offset_output = i >= (height/16-1) ? &throwaway[0] : &shifted_output[(i+1)*16*width+k*color_size+j*16];
                fsdither_kernel_simd(&shifted_input[(j-3)*16+offset+k*color_size],
                                     &shifted_output[(j-3)*16+offset+k*color_size],
                                     &shifted_output[j*16+offset+k*color_size], offset_output);
            }

            for (size_t k = 0; k < 16; k++)
            {
                DTPixel input;
                input.r = (byte) MAX(MIN(shifted_output[j*16+offset+0*color_size], 255), 0);
                input.g = (byte) MAX(MIN(shifted_output[j*16+offset+1*color_size], 255), 0);
                input.b = (byte) MAX(MIN(shifted_output[j*16+offset+2*color_size], 255), 0);
                DTPixel output = FindClosestColorFromPalette(input, palette);
                shifted_output[j*16+offset+k+color_size*0] = output.r;
                shifted_output[j*16+offset+k+color_size*1] = output.g;
                shifted_output[j*16+offset+k+color_size*2] = output.b;
            }
        }
    }
}


/**
 * 
 * 
 */
static int16_t* shift_memory(uint32_t *interleaved, size_t width, size_t height,
                        size_t *new_width, size_t *new_height)
{
    size_t padded_width = (height <= 16) ? width + 2*(height-1) : width + 2*(16-1);
    size_t padded_height = (height / 16) * 16 + (height % 16 > 0) * 16; 
    unsigned long color_size = padded_height * padded_width;
    unsigned long memory_size = 3 * color_size *sizeof(int16_t);

    int16_t* shifted_memory;
    posix_memalign((void**) &shifted_memory, 64, memory_size);

    memset(shifted_memory, 0, memory_size);

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint32_t val = interleaved[i*width+j];
            size_t shift_index = (i/16)*16*padded_width + (j+(i%16)*2)*16 + (i%16); // TODO: Fix

            //printf("%ld\n", shift_index);
            shifted_memory[shift_index + 0] = (val >> 16) & 0xFF;
            shifted_memory[shift_index + 1*color_size] = (val >> 8) & 0xFF;
            shifted_memory[shift_index + 2*color_size] =  (val >> 0) & 0xFF;
        }
    }

    *new_width = padded_width;
    *new_height = padded_height;
    return shifted_memory;
}


/**
 * 
 */
static void deshift_memory(uint32_t* interleaved, int16_t* shifted, size_t width, size_t height,
                        size_t padded_width, size_t padded_height)
{
    unsigned long color_size = padded_height * padded_width;

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            size_t shift_index = (i/16)*16*padded_width +(j+(i%16)*2)*16 + (i%16);
            printf("%ld\n", shift_index);
            interleaved[i*width+j] = (shifted[shift_index + 0] << 16 | 
                                        shifted[shift_index + 1*color_size] << 8 |
                                        shifted[shift_index + 2*color_size]);
        }
    }
}

/*
int main()
{
    int16_t *original;
    int16_t *palette;
    int16_t *offset;

    posix_memalign((void**) &original, 64, 3 * 16 * sizeof(int16_t));
    posix_memalign((void**) &palette, 64, 4 * 16 * sizeof(int16_t));
    posix_memalign((void**) &offset, 64, 1 * 16 * sizeof(int16_t));

    for (size_t i = 0; i < 16*3; i++)
        original[i] = i;
    for (size_t i = 0; i < 16*4; i++)
        palette[i] = 0;
    for (size_t i = 0; i < 16; i++)
        offset[i] = 0;

    unsigned long long ts1, ts2;
    unsigned long total = 0;
    for (int runs = 0; runs != 100; runs++)
    {
        TIMESTAMP(ts1);
        for (int i = 0; i != 1; i++)
            fsdither_kernel_simd(&original[16*i], &palette[16*i], &palette[16*(3+i)], &offset[0]);
        TIMESTAMP(ts2);
        total += (ts2-ts1);
    }
    TIME_REPORT("Dither", 0, total);

    for (size_t i = 0; i < 16; i++)
        printf("%d ", palette[16*3+i]);
    printf("\n");
    for (size_t i = 0; i < 16; i++)
        printf("%d ", offset[i]);
    printf("\n");
}
*/

void
ApplyFloydSteinbergDither(DTImage *image, DTPalettePacked *palette)
{
    size_t shifted_width;
    size_t shifted_height;
    int16_t *shifted_memory = shift_memory((uint32_t*) image->pixels, image->width, image->height,
                                            &shifted_width, &shifted_height);
    
    int16_t *shifted_output;
    posix_memalign((void**) &shifted_output, 64, 3*shifted_width*shifted_height*sizeof(int16_t));
    memset(shifted_output, 0, 3*shifted_width*shifted_height*sizeof(int16_t));

    fsdither_runner(shifted_memory, shifted_output, shifted_width, shifted_height, palette);
    deshift_memory((uint32_t*)image->pixels, shifted_output, image->width, image->height,
                                             shifted_width, shifted_height);
    free(shifted_memory);
    free(shifted_output);
}
