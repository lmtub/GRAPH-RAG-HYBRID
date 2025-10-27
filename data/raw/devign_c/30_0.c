static int config_props(AVFilterLink *inlink)

{

    AVFilterContext *ctx = inlink->dst;

    LutContext *lut = ctx->priv;

    const AVPixFmtDescriptor *desc = &av_pix_fmt_descriptors[inlink->format];

    int min[4], max[4];

    int val, comp, ret;



    lut->hsub = desc->log2_chroma_w;

    lut->vsub = desc->log2_chroma_h;



    lut->var_values[VAR_W] = inlink->w;

    lut->var_values[VAR_H] = inlink->h;



    switch (inlink->format) {

    case PIX_FMT_YUV410P:

    case PIX_FMT_YUV411P:

    case PIX_FMT_YUV420P:

    case PIX_FMT_YUV422P:

    case PIX_FMT_YUV440P:

    case PIX_FMT_YUV444P:

    case PIX_FMT_YUVA420P:

        min[Y] = min[U] = min[V] = 16;

        max[Y] = 235;

        max[U] = max[V] = 240;

        min[A] = 0; max[A] = 255;

        break;

    default:

        min[0] = min[1] = min[2] = min[3] = 0;

        max[0] = max[1] = max[2] = max[3] = 255;

    }



    lut->is_yuv = lut->is_rgb = 0;

    if      (ff_fmt_is_in(inlink->format, yuv_pix_fmts)) lut->is_yuv = 1;

    else if (ff_fmt_is_in(inlink->format, rgb_pix_fmts)) lut->is_rgb = 1;



    if (lut->is_rgb) {

        switch (inlink->format) {

        case PIX_FMT_ARGB:  lut->rgba_map[A] = 0; lut->rgba_map[R] = 1; lut->rgba_map[G] = 2; lut->rgba_map[B] = 3; break;

        case PIX_FMT_ABGR:  lut->rgba_map[A] = 0; lut->rgba_map[B] = 1; lut->rgba_map[G] = 2; lut->rgba_map[R] = 3; break;

        case PIX_FMT_RGBA:

        case PIX_FMT_RGB24: lut->rgba_map[R] = 0; lut->rgba_map[G] = 1; lut->rgba_map[B] = 2; lut->rgba_map[A] = 3; break;

        case PIX_FMT_BGRA:

        case PIX_FMT_BGR24: lut->rgba_map[B] = 0; lut->rgba_map[G] = 1; lut->rgba_map[R] = 2; lut->rgba_map[A] = 3; break;

        }

        lut->step = av_get_bits_per_pixel(desc) >> 3;

    }



    for (comp = 0; comp < desc->nb_components; comp++) {

        double res;



        /* create the parsed expression */

        ret = av_expr_parse(&lut->comp_expr[comp], lut->comp_expr_str[comp],

                            var_names, funcs1_names, funcs1, NULL, NULL, 0, ctx);

        if (ret < 0) {

            av_log(ctx, AV_LOG_ERROR,

                   "Error when parsing the expression '%s' for the component %d.\n",

                   lut->comp_expr_str[comp], comp);

            return AVERROR(EINVAL);

        }



        /* compute the lut */

        lut->var_values[VAR_MAXVAL] = max[comp];

        lut->var_values[VAR_MINVAL] = min[comp];



        for (val = 0; val < 256; val++) {

            lut->var_values[VAR_VAL] = val;

            lut->var_values[VAR_CLIPVAL] = av_clip(val, min[comp], max[comp]);

            lut->var_values[VAR_NEGVAL] =

                av_clip(min[comp] + max[comp] - lut->var_values[VAR_VAL],

                        min[comp], max[comp]);



            res = av_expr_eval(lut->comp_expr[comp], lut->var_values, lut);

            if (isnan(res)) {

                av_log(ctx, AV_LOG_ERROR,

                       "Error when evaluating the expression '%s' for the value %d for the component #%d.\n",

                       lut->comp_expr_str[comp], val, comp);

                return AVERROR(EINVAL);

            }

            lut->lut[comp][val] = av_clip((int)res, min[comp], max[comp]);

            av_log(ctx, AV_LOG_DEBUG, "val[%d][%d] = %d\n", comp, val, lut->lut[comp][val]);

        }

    }



    return 0;

}
