static int filter_frame(AVFilterLink *link, AVFrame *picref)

{

    AVFilterContext *ctx = link->dst;

    IDETContext *idet = ctx->priv;



    if (idet->prev)

        av_frame_free(&idet->prev);

    idet->prev = idet->cur;

    idet->cur  = idet->next;

    idet->next = picref;



    if (!idet->cur)

        return 0;



    if (!idet->prev)

        idet->prev = av_frame_clone(idet->cur);



    if (!idet->csp)

        idet->csp = av_pix_fmt_desc_get(link->format);

    if (idet->csp->comp[0].depth_minus1 / 8 == 1){

        idet->filter_line = (ff_idet_filter_func)ff_idet_filter_line_c_16bit;

        if (ARCH_X86)

            ff_idet_init_x86(idet, 1);

    }



    filter(ctx);



    return ff_filter_frame(ctx->outputs[0], av_frame_clone(idet->cur));

}
