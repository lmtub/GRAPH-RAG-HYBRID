static void FUNC(put_hevc_qpel_bi_w_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,

                                        int16_t *src2,

                                        int height, int denom, int wx0, int wx1,

                                        int ox0, int ox1, intptr_t mx, intptr_t my, int width)

{

    int x, y;

    const int8_t *filter;

    pixel *src = (pixel*)_src;

    ptrdiff_t srcstride = _srcstride / sizeof(pixel);

    pixel *dst          = (pixel *)_dst;

    ptrdiff_t dststride = _dststride / sizeof(pixel);

    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];

    int16_t *tmp = tmp_array;

    int shift = 14 + 1 - BIT_DEPTH;

    int log2Wd = denom + shift - 1;



    src   -= QPEL_EXTRA_BEFORE * srcstride;

    filter = ff_hevc_qpel_filters[mx - 1];

    for (y = 0; y < height + QPEL_EXTRA; y++) {

        for (x = 0; x < width; x++)

            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);

        src += srcstride;

        tmp += MAX_PB_SIZE;

    }



    tmp    = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;

    filter = ff_hevc_qpel_filters[my - 1];



    ox0     = ox0 * (1 << (BIT_DEPTH - 8));

    ox1     = ox1 * (1 << (BIT_DEPTH - 8));

    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x++)

            dst[x] = av_clip_pixel(((QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) * wx1 + src2[x] * wx0 +

                                    ((ox0 + ox1 + 1) << log2Wd)) >> (log2Wd + 1));

        tmp  += MAX_PB_SIZE;

        dst  += dststride;

        src2 += MAX_PB_SIZE;

    }

}
