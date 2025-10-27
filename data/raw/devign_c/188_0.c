static int config_props(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    AVFilterLink *inlink = outlink->src->inputs[0];

    ScaleContext *scale = ctx->priv;

    int64_t w, h;



    if (!(w = scale->w))

        w = inlink->w;

    if (!(h = scale->h))

        h = inlink->h;

    if (w == -1)

        w = av_rescale(h, inlink->w, inlink->h);

    if (h == -1)

        h = av_rescale(w, inlink->h, inlink->w);



    if (w > INT_MAX || h > INT_MAX ||

        (h * inlink->w) > INT_MAX  ||

        (w * inlink->h) > INT_MAX)

        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");



    outlink->w = w;

    outlink->h = h;



    /* TODO: make algorithm configurable */

    scale->sws = sws_getContext(inlink ->w, inlink ->h, inlink ->format,

                                outlink->w, outlink->h, outlink->format,

                                SWS_BILINEAR, NULL, NULL, NULL);



    av_log(ctx, AV_LOG_INFO, "w:%d h:%d fmt:%s\n",

           outlink->w, outlink->h, av_pix_fmt_descriptors[outlink->format].name);



    scale->input_is_pal = inlink->format == PIX_FMT_PAL8      ||

                          inlink->format == PIX_FMT_BGR4_BYTE ||

                          inlink->format == PIX_FMT_RGB4_BYTE ||

                          inlink->format == PIX_FMT_BGR8      ||

                          inlink->format == PIX_FMT_RGB8;



    return !scale->sws;

}
