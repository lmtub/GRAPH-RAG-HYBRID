static void add_pixels_clamped4_c(const DCTELEM *block, uint8_t *restrict pixels,

                          int line_size)

{

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    /* read the pixels */

    for(i=0;i<4;i++) {

        pixels[0] = cm[pixels[0] + block[0]];

        pixels[1] = cm[pixels[1] + block[1]];

        pixels[2] = cm[pixels[2] + block[2]];

        pixels[3] = cm[pixels[3] + block[3]];

        pixels += line_size;

        block += 8;

    }

}
