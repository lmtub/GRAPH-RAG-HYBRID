static void ff_h264_idct_add16intra_mmx2(uint8_t *dst, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=0; i<16; i++){

        if(nnzc[ scan8[i] ]) ff_h264_idct_add_mmx    (dst + block_offset[i], block + i*16, stride);

        else if(block[i*16]) ff_h264_idct_dc_add_mmx2(dst + block_offset[i], block + i*16, stride);

    }

}
