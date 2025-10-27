static int config_props(AVFilterLink *inlink)

{

    FadeContext *s = inlink->dst->priv;

    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(inlink->format);



    s->hsub = pixdesc->log2_chroma_w;

    s->vsub = pixdesc->log2_chroma_h;



    s->bpp = av_get_bits_per_pixel(pixdesc) >> 3;

    s->alpha &= !!(pixdesc->flags & AV_PIX_FMT_FLAG_ALPHA);

    s->is_packed_rgb = ff_fill_rgba_map(s->rgba_map, inlink->format) >= 0;



    /* use CCIR601/709 black level for studio-level pixel non-alpha components */

    s->black_level =

            ff_fmt_is_in(inlink->format, studio_level_pix_fmts) && !s->alpha ? 16 : 0;

    /* 32768 = 1 << 15, it is an integer representation

     * of 0.5 and is for rounding. */

    s->black_level_scaled = (s->black_level << 16) + 32768;

    return 0;

}
