/* Hey, Emacs, this file contains -*- c -*- code. */

/*
 * Include Clang's/POCL's OpenCL header (if available) and
 * make a few defines to improve editing experience in IDEs.
 */
#ifdef HAVE_OPENCL_C_H
#include <stdbool.h>
#include "opencl-c.h"
#define constant
#define global
#define local
#define private
#define kernel
#endif


/*
 * Define some types, so we can use the same types as
 * in the host code.
 */
typedef ulong uint64_t;
typedef uint  uint32_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef uchar uint8_t;
typedef char int8_t;

/*
 * We access individual bytes, so we need the
 * byte addressable storage extension
 */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

/*
 * Return the id of the current thread within the
 * thread's macro block, i.e., each thread gets a number
 * between 0 and 63 (x coordinate plus 8 times y coordinate
 * within the macro block).
 */
inline size_t SELF64() {
    return get_local_id(0)%8 + 8*(get_local_id(1)%8);
}

/*
 * Coefficients of the matrix A to be used in DCT computation
 */
__attribute__((aligned(sizeof(float4))))
constant float dct_coeffs[64] = {
    0.35355338f ,  0.35355338f ,  0.35355338f ,  0.35355338f  ,
                   0.35355338f ,  0.35355338f ,  0.35355338f  ,  0.35355338f,
    0.49039263f ,  0.4157348f  ,  0.2777851f  ,  9.754512e-2f ,
                  -9.754516e-2f, -0.27778518f , -0.41573483f  , -0.49039266f,
    0.46193978f ,  0.19134171f , -0.19134176f , -0.46193978f  ,
                  -0.46193978f , -0.19134156f ,  0.1913418f   ,  0.46193978f,
    0.4157348f  , -9.754516e-2f, -0.49039266f , -0.277785f    ,
                   0.2777852f  ,  0.49039263f ,  9.7545035e-2f, -0.4157349f,
    0.35355338f , -0.35355338f , -0.35355332f ,  0.3535535f   ,
                   0.35355338f , -0.35355362f , -0.35355327f  ,  0.3535534f,
    0.2777851f  , -0.49039266f ,  9.754521e-2f,  0.41573468f  ,
                  -0.4157349f  , -9.754511e-2f,  0.49039266f  , -0.27778542f,
    0.19134171f , -0.46193978f ,  0.46193978f , -0.19134195f  ,
                  -0.19134149f ,  0.46193966f , -0.46193987f  ,  0.19134195f,
    9.754512e-2f, -0.277785f   ,  0.41573468f , -0.4903926f   ,
                   0.4903927f  , -0.4157348f  ,  0.27778557f  , -9.754577e-2f
};

/*
 * Coefficients of the matrix A^tr to be used in DCT computation
 */
__attribute__((aligned(sizeof(float4))))
constant float dct_coeffs_tr[64] = {
    0.35355338f,  0.49039263f ,  0.46193978f ,  0.4157348f   ,
                  0.35355338f ,  0.2777851f  ,  0.19134171f  , 9.754512e-2f,
    0.35355338f,  0.4157348f  ,  0.19134171f , -9.754516e-2f ,
                 -0.35355338f , -0.49039266f , -0.46193978f  , -0.277785f,
    0.35355338f,  0.2777851f  , -0.19134176f , -0.49039266f  ,
                 -0.35355332f ,  9.754521e-2f, 0.46193978f   , 0.41573468f,
    0.35355338f,  9.754512e-2f, -0.46193978f , -0.277785f    ,
                  0.3535535f  ,  0.41573468f , -0.19134195f  , -0.4903926f,
    0.35355338f, -9.754516e-2f, -0.46193978f ,  0.2777852f   ,
                  0.35355338f , -0.4157349f  , -0.19134149f  , 0.4903927f,
    0.35355338f, -0.27778518f , -0.19134156f ,  0.49039263f  ,
                 -0.35355362f , -9.754511e-2f,  0.46193966f  , -0.4157348f,
    0.35355338f, -0.41573483f ,  0.1913418f  ,  9.7545035e-2f,
                 -0.35355327f ,  0.49039266f , -0.46193987f  , 0.27778557f,
    0.35355338f, -0.49039266f ,  0.46193978f , -0.4157349f   ,
                  0.3535534f ,  -0.27778542f ,  0.19134195f  , -9.754577e-2f
};

/*
 * Permutation of 64 values in a macro block.
 * Used in qdct_block() and iqdct_block().
 */
__attribute__((aligned(sizeof(int4))))
constant int permut[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

/*
 * Quantization factors for the results of the DCT
 * of a macro block. Used in qdct_block() and iqdct_block().
 */
__attribute__((aligned(sizeof(int4))))
constant int quantization_factors[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};


static uint8_t first_nibble(int8_t val) {
    if (val == 1)
        return 0x8;
    else if (val == 2)
        return 0x9;
    else if (val == -1)
        return 0xA;
    else if (val == -2)
        return 0xB;
    else if (val >= 19)
        return 0xE;
    else if (val <= -19)
        return 0xF;
    else if (val <= -3)
        return 0xD;
    else
        return 0xC;
}


// will return the length & compress the data
//'n_values' values are read from 'input'
// zeros array will hold the number of zeros one another, if the value is nonzero it will equal to 0 and if the value is unknown yet it will be equal to -1.

int compress_data(__local int8_t *macroblock, __local int8_t *zeros, __local uint8_t *nibbles, __local int8_t *numOfNibbles, __local int8_t *prefixsum, __local uint8_t *codes, __local int8_t *totalNibblesNum) {

    int row_index = get_local_id(1)%8;
    int col_index = get_local_id(0)%8;

    uint8_t nibble1;
    uint8_t nibble2;
    uint8_t nibble3;

    int pos = 0;
    int8_t val;                                 // the value for each thread
    //values[8*row_index + col_index] = macroblock[8*row_index + col_index];
    val = macroblock[8*row_index + col_index];

    int8_t newValue =0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (val == 0) {
        zeros[8*row_index + col_index] = 1;
    }
    else {
        zeros[8*row_index + col_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i=1; i < 64; i= 2*i) { //binary tree logic -> 6 iterations

        if (8*row_index + col_index >= i && zeros[8*row_index + col_index]!= 0) {

            newValue = zeros[8*row_index + col_index] + zeros[8*row_index + col_index-i];
            
            if (newValue == i+1) { // +1 is for the value itself

                zeros[8*row_index + col_index] = newValue;  // no non-zero in between
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("%d", zeros[37]);

    if (val==0) {

        if (8*row_index + col_index != 63 && zeros[8*row_index + col_index]%7 == 0) {

            numOfNibbles[8*row_index + col_index] = 1;
            nibble1 = 7;
            (*totalNibblesNum)++;
        }
        if (8*row_index + col_index != 63 && zeros[8*row_index + col_index+1] == 0){

            numOfNibbles[8*row_index + col_index] = 1;
            nibble1 = zeros[(8*row_index + col_index)]%7;
            (*totalNibblesNum)++;
        }
        if (8*row_index + col_index == 63) { //son value ise ve sıfırsa

            numOfNibbles[8*row_index + col_index] = 1;
            nibble1 = 0;
            (*totalNibblesNum)++;
        }
    }
    else {  //when we meet a non-zero value

        uint8_t absval = val < 0 ? -val : val;

        if (absval >= 19) {
            numOfNibbles[8*row_index + col_index] = 3;
            nibble1 = first_nibble(val);
            uint8_t code = absval - 19;
            nibble2 = code >> 4;
            nibble3 = code & 0xF;
            (*totalNibblesNum)+=3;
        }
        else if (absval >= 3) {
            numOfNibbles[8*row_index + col_index] = 2;
            nibble1 = first_nibble(val);
            nibble2 = absval - 3;
            (*totalNibblesNum)+=2;
        }
        else {
            numOfNibbles[8*row_index + col_index] = 1;
            nibble1 = first_nibble(val);
            (*totalNibblesNum)++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //printf("%d", numOfNibbles[59]);
    //printf("%d", *totalNibblesNum);

    // for nibbles and prefixsum array

    prefixsum[8*row_index + col_index] = numOfNibbles[8*row_index + col_index];

    for (int i=1; i < 64; i= 2*i) { //binary tree logic -> 6 iterations

        if (8*row _index + col_index >= i) {

            prefixsum[8*row_index + col_index] = prefixsum[8*row_index + col_index] + prefixsum[8*row_index + col_index-i];
        }
    }

    //prefixsum means the length of the nibbles array after I put the value at that position

    barrier(CLK_LOCAL_MEM_FENCE);

    //printf("%d", prefixsum[20]);

    // nibbles array

    if (8*row_index + col_index != 0) {

        nibbles[prefixsum[8*row_index + col_index-1]] = nibble1;

        if (nibble2) {
            nibbles[prefixsum[8*row_index + col_index-1]+1] = nibble2;
        }
        if (nibble3) {
            nibbles[prefixsum[8*row_index + col_index-1]+2] = nibble3;
        }
    }
    else {

        nibbles[0] = nibble1;

        if (nibble2) {
            nibbles[1] = nibble2;
        }
        if (nibble3) {
            nibbles[2] = nibble3;
        }
    }
    if ((*totalNibblesNum)%2 != 0) {
        nibbles[prefixsum[63]] = 0;    //this
        (*totalNibblesNum)++;
    }

    //printf("%d", *totalNibblesNum);

    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("%d", nibbles[prefixsum[23]+2]);

    //lets put the nibbles into byte array codes
    
    int nibbleNum = numOfNibbles[8*row_index + col_index];  // number of nibbles of my thread's value

    if (nibbleNum != 0) {

        int nn;

        if (8*row_index + col_index == 0) {
            nn = 0;
        }
        else {
            nn = prefixsum[8*row_index + col_index-1];    //my position on nibbles array (where it starts)*****
        }
        
        for (int i=0; i<nibbleNum; i++) {

            if (nn & 1) {   //if my position is odd
                codes[nn/2] |= nibbles[nn];
            }
            else {  // even
                codes[nn/2] |= nibbles[nn] << 4;
            }
            nn++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // return length of codes array
    
    return *totalNibblesNum;

}


/*
 * Return the offset (in bytes) where the output data for our
 * block (which has 'len' bytes) with block number 'block_nr'
 * has to be stored. 'compr' indicates whether compression is
 * enabled. 'size' and 'offsets_and_sizes' correspond to the
 * kernel arguments of the same names.
 */
void get_result_offset(size_t len, uint block_nr, bool compr,
                       global uint *size, global uint *offsets_and_sizes,
                       local size_t *off) {
    const size_t self64 = SELF64();
    if (self64 == 0) {
        /* Atomically increment 'size' by 'len'. The return value
         * is the old value of 'size'.
         */
        size_t old_size = atomic_add(size, len);

        /* When compression is not enabled, our data has to be put
         * at 64*block_nr. With compression, we store the output data
         * contiguously, but we cannot guarantee the order (the CPU
         * will reorder the data later).
         */
        *off = compr ? old_size : 64*block_nr;

        /* Write out the offset and the length of our data so
         * the CPU knows to find the data for our block.
     * We do it in the strange way (using atomic_xchg)
     * instead of simple assignments solely for
     * the purpose of avoiding the (spurious)
     * "kernel has register spilling. Lower performance is expected."
     * warning a certain vendor's OpenCL implementation
     * generates.
         */
    // offsets_and_sizes[2*block_nr  ] = off;
    // offsets_and_sizes[2*block_nr+1] = len;
    atomic_xchg(&offsets_and_sizes[2*block_nr  ], *off);
    atomic_xchg(&offsets_and_sizes[2*block_nr+1], len);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * Encode 4 macro blocks with 8x8 pixels in each work group.
 * Each thread computes its macro block's x and y coordinate
 * in blockX and blockY. self64 is the (linearized row-wise)
 * index of each thread within its macro block (in the range 0..63).
 * block_nr is the global number of the macro block (numbering the
 * macro blocks row-wise from left to right).
 * 'image', 'rows' and 'columns' describe the input image.
 * 'format' tell whether to reoder only, apply DCT or
 * compute DCT+compression.
 * The result is stored in 'frame'.
 * 'size' and 'offsets_and_sizes' are used when compression is enabled
 * to tell the host where to find the data for each macro block
 * (see below).
 */
__attribute__((reqd_work_group_size(16, 16, 1)))
kernel void encode_frame(global uint8_t *image, int rows, int columns,
                         int format, global uint *size,
                         global uint *offsets_and_sizes,
                         global uint8_t *frame) {
    const uint self64 = SELF64();
    const uint blockX = get_global_id(0)/8;
    const uint blockY = get_global_id(1)/8;
    const uint block_nr = blockY*(columns/8) + blockX;

    // lb is the local block number, i.e., which of the 4
    // macro blocks in a work group this thread is in
    // (0 = upper left, 1 = upper right, 2 = lower left, 3 = lower right).
    const uint lb = get_local_id(0)/8 + 2*(get_local_id(1)/8);

    // For each macro block, the result is stored in 'result'.
    // Each of the 4 macro blocks in a work group has its own
    // result array. Use 'sresult' to store signed values,
    // 'result' to store unsigned values.
    local uint8_t result_[4][96] __attribute__((aligned(16)));
    local uint8_t *result = result_[lb];
    local int8_t *sresult = (local int8_t *)result;
    size_t len;

    const bool compr  =  format == 2;  // Is compression (-c) requested?

    // 'current' points to the upper left corner of the macro block
    // assigned to this thread.
    global uint8_t *current = image + 8*blockY*columns + 8*blockX;


    switch(format) {
    case 0: {  // Exercise (a)

        break;
    }
    case 1: {  // Exercise (b)

        break;
    }
    case 2: {  // Exercise (c)

        uint8_t row_index, col_index;

        row_index = self64 / 8;
        col_index = self64 % 8;

        local int8_t theB[4][64];
        local int8_t *B = theB[lb];

        B[row_index*8 + col_index] = (int8_t)((int)current[row_index*columns + col_index] - 128);

        local float the_ABtr_tr[4][64];
        local float *ABtr_tr = the_ABtr_tr[lb];

        // converting the values into float
        local float _myMatrix[4][64];
        _myMatrix[lb][self64] = B[self64];
        local float *myMatrix=_myMatrix[lb];

        barrier(CLK_LOCAL_MEM_FENCE);

        //Compute A*B_tr into ABtr_tr
        ABtr_tr[self64] = 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i=0; i<8; i++) {

            ABtr_tr[col_index + 8*row_index] += dct_coeffs_tr[8*row_index + i] * myMatrix[8*col_index + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        //Compute (A*B_tr)*A_tr in C
        local float the_C[4][64];
        local float *C = the_C[lb];

        C[self64]=0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i=0; i<8; i++) {

            C[8*row_index + col_index] += ABtr_tr[8*col_index + i] * dct_coeffs_tr[8*row_index + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // quantization_factors AND permute stuff

        sresult[ permut[self64] ] = convert_int_rte(C[self64] / quantization_factors[self64]);

        barrier(CLK_LOCAL_MEM_FENCE);

        local int8_t zeros_[4][64];
        local int8_t *zeros = zeros_[lb];

        local uint8_t nibbles_[4][192];
        local uint8_t *nibbles = nibbles_[lb];

        local int8_t numOfNibbles_[4][64];
        local int8_t *numOfNibbles = numOfNibbles_[lb];

        local int8_t prefixsum_[4][64];
        local int8_t *prefixsum = prefixsum_[lb];

        local uint8_t codes_[4][96];
        local uint8_t *codes = codes_[lb];
        codes[8*row_index + col_index] = 0;
        if (8*row_index + col_index < 32) {
            codes[8*row_index + col_index + 32] = 0;
        }


        local int8_t totalNibblesNum_[4];
        local int8_t *totalNibblesNum = totalNibblesNum_[lb];
        *totalNibblesNum = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        len = compress_data(sresult[lb], zeros, nibbles, numOfNibbles, prefixsum, codes, totalNibblesNum);
        //printf("%d", len);

        barrier(CLK_GLOBAL_MEM_FENCE);

        //printf("%d", codes[3]);

        break;
    }
    default:
        len = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute at which offset to put our data.
    // When compression is not enabled, the position of the output
    // data is simply 64*block_nr. When compression is enabled,
    // this is more difficult as we do not know how many bytes the
    // preceding macro blocks occupy (and we cannot synchronize
    // accross work groups). Therefore, the data for each macro block
    // is appended to the output as soon as the block is ready.
    // get_result_offset atomically increments *size by len
    // and stores the old value of *size together with len
    // in offsets_and_sizes, so the host knows later where
    // we stored our data. The host reorders the results to
    // be in the correct order. The value of *size before
    // the increment by len is stored in off[lb], so we can
    // put our data at frame[off[lb]..off[lb]+len-1].
    local size_t off[4];
    get_result_offset(len, block_nr, compr,
                      size, offsets_and_sizes, &off[lb]);
    for (size_t i=self64; i<len; i+=64)
        frame[off[lb]+i] = result[i];
    barrier(CLK_GLOBAL_MEM_FENCE);
}
