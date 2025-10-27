void ff_put_pixels_clamped_c(const DCTELEM *block, uint8_t *restrict pixels,

                             int line_size)

{

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    /* read the pixels */

    for(i=0;i<8;i++) {

        pixels[0] = cm[block[0]];

        pixels[1] = cm[block[1]];

        pixels[2] = cm[block[2]];

        pixels[3] = cm[block[3]];

        pixels[4] = cm[block[4]];

        pixels[5] = cm[block[5]];

        pixels[6] = cm[block[6]];

        pixels[7] = cm[block[7]];



        pixels += line_size;

        block += 8;

    }

}
