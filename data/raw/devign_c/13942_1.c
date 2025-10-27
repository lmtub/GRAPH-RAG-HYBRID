static void ff_wmv2_idct_add_c(uint8_t *dest, int line_size, DCTELEM *block)

{

    ff_wmv2_idct_c(block);

    add_pixels_clamped_c(block, dest, line_size);

}
