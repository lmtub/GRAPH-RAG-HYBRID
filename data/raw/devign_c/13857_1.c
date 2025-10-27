void ff_jref_idct_add(uint8_t *dest, ptrdiff_t line_size, int16_t *block)

{

    ff_j_rev_dct(block);

    ff_add_pixels_clamped(block, dest, line_size);

}
