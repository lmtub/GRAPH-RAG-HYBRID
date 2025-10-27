void ff_vp3_idct_put_c(uint8_t *dest/*align 8*/, int line_size, DCTELEM *block/*align 16*/){

    idct(dest, line_size, block, 1);

}
