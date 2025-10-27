static void ff_h264_idct_add8_mmx2(uint8_t **dest, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=16; i<16+8; i++){

        if(nnzc[ scan8[i] ])

            ff_h264_idct_add_mmx    (dest[(i&4)>>2] + block_offset[i], block + i*16, stride);

        else if(block[i*16])

            ff_h264_idct_dc_add_mmx2(dest[(i&4)>>2] + block_offset[i], block + i*16, stride);

    }

}
