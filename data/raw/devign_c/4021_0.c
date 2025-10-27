static av_always_inline int get_dst_color_err(PaletteUseContext *s,

                                              uint32_t c, int *er, int *eg, int *eb,

                                              const enum color_search_method search_method)

{

    const uint8_t a = c >> 24 & 0xff;

    const uint8_t r = c >> 16 & 0xff;

    const uint8_t g = c >>  8 & 0xff;

    const uint8_t b = c       & 0xff;

    const int dstx = color_get(s, c, a, r, g, b, search_method);

    const uint32_t dstc = s->palette[dstx];

    *er = r - (dstc >> 16 & 0xff);

    *eg = g - (dstc >>  8 & 0xff);

    *eb = b - (dstc       & 0xff);

    return dstx;

}
