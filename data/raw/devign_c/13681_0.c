static void fill_block(uint16_t *pdest, uint16_t color, int block_size, int pitch)

{

    int x, y;



    pitch -= block_size;

    for (y = 0; y != block_size; y++, pdest += pitch)

        for (x = 0; x != block_size; x++)

            *pdest++ = color;

}
