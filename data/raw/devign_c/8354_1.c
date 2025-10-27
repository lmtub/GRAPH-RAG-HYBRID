static void vp8_idct_dc_add4y_c(uint8_t *dst, int16_t block[4][16],

                                ptrdiff_t stride)

{

    vp8_idct_dc_add_c(dst +  0, block[0], stride);

    vp8_idct_dc_add_c(dst +  4, block[1], stride);

    vp8_idct_dc_add_c(dst +  8, block[2], stride);

    vp8_idct_dc_add_c(dst + 12, block[3], stride);

}
