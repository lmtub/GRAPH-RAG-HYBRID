static void arm_idct_add(UINT8 *dest, int line_size, DCTELEM *block)

{

    j_rev_dct_ARM (block);

    add_pixels_clamped(block, dest, line_size);

}
