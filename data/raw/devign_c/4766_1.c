void ff_vp3_idct_dc_add_c(uint8_t *dest/*align 8*/, int line_size, const DCTELEM *block/*align 16*/){

    int i, dc = (block[0] + 15) >> 5;

    const uint8_t *cm = ff_cropTbl + MAX_NEG_CROP + dc;



    for(i = 0; i < 8; i++){

        dest[0] = cm[dest[0]];

        dest[1] = cm[dest[1]];

        dest[2] = cm[dest[2]];

        dest[3] = cm[dest[3]];

        dest[4] = cm[dest[4]];

        dest[5] = cm[dest[5]];

        dest[6] = cm[dest[6]];

        dest[7] = cm[dest[7]];

        dest += line_size;

    }

}
