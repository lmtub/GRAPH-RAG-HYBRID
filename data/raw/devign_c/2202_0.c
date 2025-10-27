static void ff_h264_idct8_add4_mmx2(uint8_t *dst, const int *block_offset, DCTELEM *block, int stride, const uint8_t nnzc[6*8]){

    int i;

    for(i=0; i<16; i+=4){

        int nnz = nnzc[ scan8[i] ];

        if(nnz){

            if(nnz==1 && block[i*16]) ff_h264_idct8_dc_add_mmx2(dst + block_offset[i], block + i*16, stride);

            else                      ff_h264_idct8_add_mmx    (dst + block_offset[i], block + i*16, stride);

        }

    }

}
