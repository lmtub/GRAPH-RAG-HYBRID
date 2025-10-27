void ff_jref_idct_put(uint8_t *dest, ptrdiff_t line_size, int16_t *block)

{

    ff_j_rev_dct(block);

    ff_put_pixels_clamped(block, dest, line_size);

}
