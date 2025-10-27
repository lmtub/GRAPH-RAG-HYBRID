static void FUNC(sao_band_filter)(uint8_t *_dst, uint8_t *_src,

                                  ptrdiff_t stride, SAOParams *sao,

                                  int *borders, int width, int height,

                                  int c_idx, int class)

{

    pixel *dst = (pixel *)_dst;

    pixel *src = (pixel *)_src;

    int offset_table[32] = { 0 };

    int k, y, x;

    int chroma = !!c_idx;

    int shift  = BIT_DEPTH - 5;

    int *sao_offset_val = sao->offset_val[c_idx];

    int sao_left_class  = sao->band_position[c_idx];

    int init_y = 0, init_x = 0;



    stride /= sizeof(pixel);



    switch (class) {

    case 0:

        if (!borders[2])

            width -= (8 >> chroma) + 2;

        if (!borders[3])

            height -= (4 >> chroma) + 2;

        break;

    case 1:

        init_y = -(4 >> chroma) - 2;

        if (!borders[2])

            width -= (8 >> chroma) + 2;

        height = (4 >> chroma) + 2;

        break;

    case 2:

        init_x = -(8 >> chroma) - 2;

        width  =  (8 >> chroma) + 2;

        if (!borders[3])

            height -= (4 >> chroma) + 2;

        break;

    case 3:

        init_y = -(4 >> chroma) - 2;

        init_x = -(8 >> chroma) - 2;

        width  =  (8 >> chroma) + 2;

        height =  (4 >> chroma) + 2;

        break;

    }



    dst = dst + (init_y * stride + init_x);

    src = src + (init_y * stride + init_x);

    for (k = 0; k < 4; k++)

        offset_table[(k + sao_left_class) & 31] = sao_offset_val[k + 1];

    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x++)

            dst[x] = av_clip_pixel(src[x] + offset_table[av_clip_pixel(src[x] >> shift)]);

        dst += stride;

        src += stride;

    }

}
