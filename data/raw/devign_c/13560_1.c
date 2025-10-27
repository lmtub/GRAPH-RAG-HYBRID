static void smptebars_fill_picture(AVFilterContext *ctx, AVFrame *picref)

{

    TestSourceContext *test = ctx->priv;

    FFDrawColor color;

    int r_w, r_h, w_h, p_w, p_h, i, tmp, x = 0;

    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(picref->format);



    r_w = FFALIGN((test->w + 6) / 7, 1 << pixdesc->log2_chroma_w);

    r_h = FFALIGN(test->h * 2 / 3, 1 << pixdesc->log2_chroma_h);

    w_h = FFALIGN(test->h * 3 / 4 - r_h,  1 << pixdesc->log2_chroma_h);

    p_w = FFALIGN(r_w * 5 / 4, 1 << pixdesc->log2_chroma_w);

    p_h = test->h - w_h - r_h;



#define DRAW_COLOR(rgba, x, y, w, h)                                    \

    ff_draw_color(&test->draw, &color, rgba);                           \

    ff_fill_rectangle(&test->draw, &color,                              \

                      picref->data, picref->linesize, x, y, w, h)       \



    for (i = 0; i < 7; i++) {

        DRAW_COLOR(rainbow[i], x, 0,   FFMIN(r_w, test->w - x), r_h);

        DRAW_COLOR(wobnair[i], x, r_h, FFMIN(r_w, test->w - x), w_h);

        x += r_w;

    }

    x = 0;

    DRAW_COLOR(i_pixel, x, r_h + w_h, p_w, p_h);

    x += p_w;

    DRAW_COLOR(white, x, r_h + w_h, p_w, p_h);

    x += p_w;

    DRAW_COLOR(q_pixel, x, r_h + w_h, p_w, p_h);

    x += p_w;

    tmp = FFALIGN(5 * r_w - x,  1 << pixdesc->log2_chroma_w);

    DRAW_COLOR(black, x, r_h + w_h, tmp, p_h);

    x += tmp;

    tmp = FFALIGN(r_w / 3,  1 << pixdesc->log2_chroma_w);

    DRAW_COLOR(neg4ire, x, r_h + w_h, tmp, p_h);

    x += tmp;

    DRAW_COLOR(black, x, r_h + w_h, tmp, p_h);

    x += tmp;

    DRAW_COLOR(pos4ire, x, r_h + w_h, tmp, p_h);

    x += tmp;

    DRAW_COLOR(black, x, r_h + w_h, test->w - x, p_h);

}
