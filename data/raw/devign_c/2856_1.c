static void put_pixels_clamped4_c(const DCTELEM *block, uint8_t *restrict pixels,

                                 int line_size)

{

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    /* read the pixels */

    for(i=0;i<4;i++) {

        pixels[0] = cm[block[0]];

        pixels[1] = cm[block[1]];

        pixels[2] = cm[block[2]];

        pixels[3] = cm[block[3]];



        pixels += line_size;

        block += 8;

    }

}
