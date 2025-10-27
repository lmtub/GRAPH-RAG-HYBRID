static void add_pixels_clamped2_c(const DCTELEM *block, uint8_t *restrict pixels,

                          int line_size)

{

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    /* read the pixels */

    for(i=0;i<2;i++) {

        pixels[0] = cm[pixels[0] + block[0]];

        pixels[1] = cm[pixels[1] + block[1]];

        pixels += line_size;

        block += 8;

    }

}
