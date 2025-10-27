static void add_pixels_clamped_c(const int16_t *block, uint8_t *av_restrict pixels,

                                 ptrdiff_t line_size)

{

    int i;



    /* read the pixels */

    for (i = 0; i < 8; i++) {

        pixels[0] = av_clip_uint8(pixels[0] + block[0]);

        pixels[1] = av_clip_uint8(pixels[1] + block[1]);

        pixels[2] = av_clip_uint8(pixels[2] + block[2]);

        pixels[3] = av_clip_uint8(pixels[3] + block[3]);

        pixels[4] = av_clip_uint8(pixels[4] + block[4]);

        pixels[5] = av_clip_uint8(pixels[5] + block[5]);

        pixels[6] = av_clip_uint8(pixels[6] + block[6]);

        pixels[7] = av_clip_uint8(pixels[7] + block[7]);

        pixels   += line_size;

        block    += 8;

    }

}
