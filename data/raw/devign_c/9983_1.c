static int config_props(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    AVFilterLink *inlink = outlink->src->inputs[0];

    ScaleContext *scale = ctx->priv;

    int64_t w, h;

    double var_values[VARS_NB], res;

    char *expr;

    int ret;



    var_values[VAR_PI]    = M_PI;

    var_values[VAR_PHI]   = M_PHI;

    var_values[VAR_E]     = M_E;

    var_values[VAR_IN_W]  = var_values[VAR_IW] = inlink->w;

    var_values[VAR_IN_H]  = var_values[VAR_IH] = inlink->h;

    var_values[VAR_OUT_W] = var_values[VAR_OW] = NAN;

    var_values[VAR_OUT_H] = var_values[VAR_OH] = NAN;

    var_values[VAR_DAR]   = var_values[VAR_A]  = (float) inlink->w / inlink->h;

    var_values[VAR_SAR]   = inlink->sample_aspect_ratio.num ?

        (float) inlink->sample_aspect_ratio.num / inlink->sample_aspect_ratio.den : 1;

    var_values[VAR_HSUB]  = 1<<av_pix_fmt_descriptors[inlink->format].log2_chroma_w;

    var_values[VAR_VSUB]  = 1<<av_pix_fmt_descriptors[inlink->format].log2_chroma_h;



    /* evaluate width and height */

    av_expr_parse_and_eval(&res, (expr = scale->w_expr),

                           var_names, var_values,

                           NULL, NULL, NULL, NULL, NULL, 0, ctx);

    scale->w = var_values[VAR_OUT_W] = var_values[VAR_OW] = res;

    if ((ret = av_expr_parse_and_eval(&res, (expr = scale->h_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail;

    scale->h = var_values[VAR_OUT_H] = var_values[VAR_OH] = res;

    /* evaluate again the width, as it may depend on the output height */

    if ((ret = av_expr_parse_and_eval(&res, (expr = scale->w_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail;

    scale->w = res;



    w = scale->w;

    h = scale->h;



    /* sanity check params */

    if (w <  -1 || h <  -1) {

        av_log(ctx, AV_LOG_ERROR, "Size values less than -1 are not acceptable.\n");

        return AVERROR(EINVAL);

    }

    if (w == -1 && h == -1)

        scale->w = scale->h = 0;



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

    av_log(ctx, AV_LOG_INFO, "w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s flags:0x%0x\n",

           inlink ->w, inlink ->h, av_pix_fmt_descriptors[ inlink->format].name,

           outlink->w, outlink->h, av_pix_fmt_descriptors[outlink->format].name,

           scale->flags);



    scale->input_is_pal = av_pix_fmt_descriptors[inlink->format].flags & PIX_FMT_PAL;



    if (scale->sws)

        sws_freeContext(scale->sws);

    scale->sws = sws_getContext(inlink ->w, inlink ->h, inlink ->format,

                                outlink->w, outlink->h, outlink->format,

                                scale->flags, NULL, NULL, NULL);

    if (!scale->sws)

        return AVERROR(EINVAL);





    if (inlink->sample_aspect_ratio.num)

        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h*inlink->w,

                                                             outlink->w*inlink->h},

                                                inlink->sample_aspect_ratio);

    else

        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;



    return 0;



fail:

    av_log(NULL, AV_LOG_ERROR,

           "Error when evaluating the expression '%s'\n", expr);

    return ret;

}
