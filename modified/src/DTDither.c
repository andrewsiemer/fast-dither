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
#include <assert.h>

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

/**
 * 
*/
static int16_t fsdither_kernel_scalar(int16_t diff_3, int16_t diff_2, int16_t diff_1, int16_t diff_0,
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
                                int16_t *output, // Array A
                                int16_t *offset // Array A
                                )
{
    __m256i diff_cols[3];
    __m256i diff_offset;
    __m256i original;

    // Data loading scope
    {
        size_t i;
        for (i = 0; i < 3; i++)
        {
            __m256i original = _mm256_load_si256((__m256i*) &input[i*16]);
            __m256i searched = _mm256_load_si256((__m256i*) &palette[i*16]);
            diff_cols[i] = _mm256_sub_epi16(original, searched);
        }

        original = _mm256_load_si256((__m256i*) &input[i*16]);
        //diff_output = _mm256_load_si256((__m256i*) &palette[i*16]);
    }
    
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
            diff_offset = temp;
            //diff_offset = _mm256_add_epi16(diff_offset, temp);

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
        //diff_output = _mm256_add_epi16(diff_output, wake_diff);

        // Calculate the 3/16 term
        diff_cols[2] = _mm256_mullo_epi16(diff_cols[2], scalar_3);
        diff_cols[2] = _mm256_srai_epi16(diff_cols[2], 4);
        diff_cols[2] = _mm256_add_epi16(diff_cols[2], wake_diff);

        // Calculate the 5/16 term
        diff_cols[1] = _mm256_mullo_epi16(diff_cols[1], scalar_5);
        diff_cols[1] = _mm256_srai_epi16(diff_cols[1], 4);
        diff_cols[1] = _mm256_add_epi16(diff_cols[2], diff_cols[1]);

        // Calculate the 1/16 term
        diff_cols[0] = _mm256_srai_epi16(diff_cols[0], 4);
        diff_cols[0] = _mm256_add_epi16(diff_cols[1], diff_cols[0]);
    }
    
    // Data storing scope
    {
        // Finish shift
        // Left-shift temp all the way to the other side of the lane
        __m256i offset_output = _mm256_bsrli_epi128(diff_cols[0], 16-2); 
        // Create temp with swap lanes
        __m256i temp = _mm256_permute2x128_si256(offset_output, diff_offset, 0x02); 
        offset_output = _mm256_permute2x128_si256(offset_output, offset_output, 0x01);

        // Pop top 16b from both lanes
        diff_cols[0] = _mm256_bslli_epi128(diff_cols[0], 2); 
        // Re-add inner lost 16b and offset 16b
        diff_cols[0] = _mm256_blend_epi16(diff_cols[0], temp, 0x01);

        // Min and max
        __m256i const_255 = _mm256_set1_epi16(255);
        diff_cols[0] = _mm256_add_epi16(diff_cols[0], original);
        diff_cols[0] = _mm256_min_epi16(diff_cols[0], const_255);
        diff_cols[0] = _mm256_max_epi16(diff_cols[0], _mm256_setzero_si256());
        
        // Store output
        //PRINT_M256_EPI16(diff_output);
        _mm256_store_si256((__m256i*) output, diff_cols[0]);

        // Store lone 16b int
        __m256i original_offset = _mm256_load_si256((__m256i*) offset);
        __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF);
        offset_output = _mm256_and_si256(mask, offset_output);        
        offset_output = _mm256_add_epi16(offset_output, original_offset);
        _mm256_store_si256((__m256i*) offset, offset_output);
    }
}

/**
 * 
 */
static void fsdither_runner(int16_t *shifted_input, int16_t *shifted_output, size_t width, size_t height,
                     DTPalettePacked *palette, dt_time_t* time)
{
    unsigned long color_size = height * width;
    __attribute__((aligned(32))) int16_t throwaway[16];
    
    for (size_t i = 0; i < height / 16; i++)
    {
        // Constant calculation
        const size_t offset = i*16*width;
        const size_t next_offset = (i+1)*16*width;

        // Startup
        for (size_t j = 0; j < MIN(3, width); j++)
        {
            DTPixelDiff input;
            DTPixel output;
            switch (j)
            {
                // TODO: Fix with input
                case 0:
                    for (size_t k = 0; k < 3; k++)
                        shifted_input[offset+color_size*k] = shifted_input[offset+color_size*k];
                    input.r = shifted_input[offset+color_size*0];
                    input.g = shifted_input[offset+color_size*1];
                    input.b = shifted_input[offset+color_size*2];
                    output = FindClosestColorFromPaletteDiff(input, palette);
                    shifted_output[offset+color_size*0] = output.r;
                    shifted_output[offset+color_size*1] = output.g;
                    shifted_output[offset+color_size*2] = output.b;
                    break;
                case 1:
                    for (size_t k = 0; k < 3; k++)
                        shifted_input[offset+16+color_size*k] = fsdither_kernel_scalar(0, 0, 0, 
                            shifted_input[offset+color_size*k],
                            shifted_input[offset+16+color_size*k]);
                    input.r = shifted_input[offset+16+color_size*0];
                    input.g = shifted_input[offset+16+color_size*1];
                    input.b = shifted_input[offset+16+color_size*2];
                    output = FindClosestColorFromPaletteDiff(input, palette);
                    shifted_output[offset+16+color_size*0] = output.r;
                    shifted_output[offset+16+color_size*1] = output.g;
                    shifted_output[offset+16+color_size*2] = output.b;
                    break;
                case 2:
                default:
                    for (size_t k = 0; k < 3; k++) {
                        shifted_input[offset+32+color_size*k] = fsdither_kernel_scalar(0, 0, 0, 
                            shifted_input[offset+16+color_size*k],
                            shifted_input[offset+32+color_size*k]);
                        shifted_input[offset+33+color_size*k] = fsdither_kernel_scalar(shifted_input[offset+16+color_size*k],
                            0, 0, 0, shifted_input[offset+33+color_size*k]);
                    }
                    for (size_t k = 0; k < 2; k++) {
                        input.r = shifted_input[offset+32+k+color_size*0];
                        input.g = shifted_input[offset+32+k+color_size*1];
                        input.b = shifted_input[offset+32+k+color_size*2];
                        output = FindClosestColorFromPaletteDiff(input, palette);
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
                unsigned long long ts1, ts2;
                int16_t* offset_output = (i >= (height/16-1)) || (j < 32) ? &throwaway[0] : &shifted_input[next_offset+k*color_size+(j-32)*16];
                TIMESTAMP(ts1);
                fsdither_kernel_simd(&shifted_input[(j-3)*16+offset+k*color_size],
                                     &shifted_output[(j-3)*16+offset+k*color_size],
                                     &shifted_input[j*16+offset+k*color_size], offset_output);
                TIMESTAMP(ts2);
                time->dither_time += (ts2 - ts1);
            }
            time->dither_units += 16;

            for (size_t k = 0; k < 16; k++)
            {
                shifted_input[j*16+offset+0*color_size+k] = MAX(MIN(shifted_input[j*16+offset+0*color_size+k], 255), 0);
                shifted_input[j*16+offset+1*color_size+k] = MAX(MIN(shifted_input[j*16+offset+1*color_size+k], 255), 0);
                shifted_input[j*16+offset+2*color_size+k] = MAX(MIN(shifted_input[j*16+offset+2*color_size+k], 255), 0);
            }

            for (size_t k = 0; k < 16; k++)
            {
                DTPixelDiff input;
                input.r = shifted_input[j*16+offset+0*color_size+k];
                input.g = shifted_input[j*16+offset+1*color_size+k];
                input.b = shifted_input[j*16+offset+2*color_size+k];
                DTPixel output = FindClosestColorFromPaletteDiff(input, palette);
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
static int16_t* shift_memory(DTPixel *pixels, size_t width, size_t height,
                        size_t *new_width, size_t *new_height, dt_time_t *time)
{
    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);

    size_t padded_width = (height <= 16) ? width + 2*(height-1) : width + 2*(16-1);
    size_t padded_height = (height / 16) * 16 + (height % 16 > 0) * 16; 
    unsigned long color_size = padded_height * padded_width;
    unsigned long memory_size = 3 * color_size * sizeof(int16_t);

    int16_t* shifted_memory;
    posix_memalign((void**) &shifted_memory, 64, memory_size);

    memset(shifted_memory, 0, memory_size);

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            DTPixel val = pixels[i*width+j];
            size_t shift_index = (i/16)*16*padded_width + (j+(i%16)*2)*16 + (i%16);

            shifted_memory[shift_index + 0] = val.r;
            shifted_memory[shift_index + 1*color_size] = val.g;
            shifted_memory[shift_index + 2*color_size] =  val.b;
        }
    }

    *new_width = padded_width;
    *new_height = padded_height;

    TIMESTAMP(ts2);
    time->shift_time += (ts2 - ts1);
    time->shift_units += color_size;

    return shifted_memory;
}

/**
 * 
 */
static void deshift_memory(DTPixel *pixels, int16_t* shifted, size_t width, size_t height,
                        size_t padded_width, size_t padded_height, dt_time_t *time)
{
    unsigned long long ts1, ts2;
    TIMESTAMP(ts1);

    unsigned long color_size = padded_height * padded_width;

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            size_t shift_index = (i/16)*16*padded_width +(j+(i%16)*2)*16 + (i%16);
            pixels[i*width+j].r = shifted[shift_index + 0];
            pixels[i*width+j].g = shifted[shift_index + 1*color_size];
            pixels[i*width+j].b = shifted[shift_index + 2*color_size];
        }
    }

    TIMESTAMP(ts2);
    time->deshift_time += (ts2 - ts1);
    time->deshift_units += color_size;
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

/*
static void
DTTimeAdd(
    dt_time_t *dst,
    dt_time_t *src
) {
    dst->shift_time += src->shift_time;
    dst->shift_units += src->shift_units;
    dst->dither_time += src->dither_time;
    dst->dither_units += src->dither_units;
    dst->deshift_time += src->deshift_time;
    dst->deshift_units += src->deshift_units;
}
*/

void DTTimeInit(dt_time_t *time)
{
    assert(time);
    *time = (dt_time_t) {0};
}

void DTTimeReport(dt_time_t *time)
{
    const double shift_theoretical = (2*16.0/3.0);
    const double dither_theoretical = (2.0/3.0);
    const double deshift_theoretical = (2*16.0/3.0);

    double shift_time = TIME_NORM(0, time->shift_time);
    double shift_pix = ((double)time->shift_units) / shift_time;
    double shift_peak = (shift_pix / shift_theoretical) * 100;

    double dither_time = TIME_NORM(0, time->dither_time);
    double dither_pix = ((double)time->dither_units) / dither_time;
    double dither_peak = (dither_pix / dither_theoretical) * 100;

    double deshift_time = TIME_NORM(0, time->deshift_time);
    double deshift_pix = ((double)time->deshift_units) / deshift_time;
    double deshift_peak = (deshift_pix / deshift_theoretical) * 100;

    printf("Shift%20s%-20.6lf%-20.6lf%.2lf%%\n", "", shift_time, shift_pix, shift_peak);
    printf("Dither%19s%-20.6lf%-20.6lf%.2lf%%\n", "", dither_time, dither_pix, dither_peak);
    printf("Deshift%18s%-20.6lf%-20.6lf%.2lf%%\n", "", deshift_time, deshift_pix, deshift_peak);
}

void
ApplyFloydSteinbergDither(DTImage *image, DTPalettePacked *palette)
{
    dt_time_t t;
    DTTimeInit(&t);

    size_t shifted_width;
    size_t shifted_height;
    int16_t *shifted_memory = shift_memory(image->pixels, image->width, image->height,
                                            &shifted_width, &shifted_height, &t);
    
    int16_t *shifted_output;
    posix_memalign((void**) &shifted_output, 64, 3*shifted_width*shifted_height*sizeof(int16_t));
    memset(shifted_output, 0, 3*shifted_width*shifted_height*sizeof(int16_t));

    fsdither_runner(shifted_memory, shifted_output, shifted_width, shifted_height, palette, &t);
    deshift_memory(image->pixels, shifted_output, image->width, image->height,
                                             shifted_width, shifted_height, &t);
    free(shifted_memory);
    free(shifted_output);

    DTTimeReport(&t);
}


