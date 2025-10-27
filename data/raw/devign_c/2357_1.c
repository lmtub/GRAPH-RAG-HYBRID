static void rv34_idct_dc_add_c(uint8_t *dst, int stride, int dc)

{

    const uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    int i, j;



    cm += (13*13*dc + 0x200) >> 10;



    for (i = 0; i < 4; i++)

    {

        for (j = 0; j < 4; j++)

            dst[j] = cm[ dst[j] ];



        dst += stride;

    }

}
