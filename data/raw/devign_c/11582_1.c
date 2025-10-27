static void FUNC(put_hevc_epel_bi_w_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,

                                       int16_t *src2,

                                       int height, int denom, int wx0, int wx1,

                                       int ox0, int ox1, intptr_t mx, intptr_t my, int width)

{

    int x, y;

    pixel *src = (pixel *)_src;

    ptrdiff_t srcstride  = _srcstride / sizeof(pixel);

    const int8_t *filter = ff_hevc_epel_filters[my - 1];

    pixel *dst          = (pixel *)_dst;

    ptrdiff_t dststride = _dststride / sizeof(pixel);

    int shift = 14 + 1 - BIT_DEPTH;

    int log2Wd = denom + shift - 1;



    ox0     = ox0 * (1 << (BIT_DEPTH - 8));

    ox1     = ox1 * (1 << (BIT_DEPTH - 8));

    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x++)

            dst[x] = av_clip_pixel(((EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) * wx1 + src2[x] * wx0 +

                                    ((ox0 + ox1 + 1) << log2Wd)) >> (log2Wd + 1));

        src  += srcstride;

        dst  += dststride;

        src2 += MAX_PB_SIZE;

    }

}
