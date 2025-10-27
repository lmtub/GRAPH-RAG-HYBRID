static void ff_h264_idct_add8_sse2(uint8_t **dest, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=16; i<16+8; i+=2){

        if(nnzc[ scan8[i+0] ]|nnzc[ scan8[i+1] ])

            ff_x264_add8x4_idct_sse2 (dest[(i&4)>>2] + block_offset[i], block + i*16, stride);

        else if(block[i*16]|block[i*16+16])

            ff_h264_idct_dc_add8_mmx2(dest[(i&4)>>2] + block_offset[i], block + i*16, stride);

    }

}
