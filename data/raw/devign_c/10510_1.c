static int config_input(AVFilterLink *link)

{

    AVFilterContext *ctx = link->dst;

    CropContext *s = ctx->priv;

    const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(link->format);

    int ret;

    const char *expr;

    double res;



    s->var_values[VAR_E]     = M_E;

    s->var_values[VAR_PHI]   = M_PHI;

    s->var_values[VAR_PI]    = M_PI;

    s->var_values[VAR_IN_W]  = s->var_values[VAR_IW] = ctx->inputs[0]->w;

    s->var_values[VAR_IN_H]  = s->var_values[VAR_IH] = ctx->inputs[0]->h;

    s->var_values[VAR_X]     = NAN;

    s->var_values[VAR_Y]     = NAN;

    s->var_values[VAR_OUT_W] = s->var_values[VAR_OW] = NAN;

    s->var_values[VAR_OUT_H] = s->var_values[VAR_OH] = NAN;

    s->var_values[VAR_N]     = 0;

    s->var_values[VAR_T]     = NAN;



    av_image_fill_max_pixsteps(s->max_step, NULL, pix_desc);

    s->hsub = pix_desc->log2_chroma_w;

    s->vsub = pix_desc->log2_chroma_h;



    if ((ret = av_expr_parse_and_eval(&res, (expr = s->ow_expr),

                                      var_names, s->var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail_expr;

    s->var_values[VAR_OUT_W] = s->var_values[VAR_OW] = res;

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->oh_expr),

                                      var_names, s->var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail_expr;

    s->var_values[VAR_OUT_H] = s->var_values[VAR_OH] = res;

    /* evaluate again ow as it may depend on oh */

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->ow_expr),

                                      var_names, s->var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto fail_expr;



    s->var_values[VAR_OUT_W] = s->var_values[VAR_OW] = res;

    if (normalize_double(&s->w, s->var_values[VAR_OUT_W]) < 0 ||

        normalize_double(&s->h, s->var_values[VAR_OUT_H]) < 0) {

        av_log(ctx, AV_LOG_ERROR,

               "Too big value or invalid expression for out_w/ow or out_h/oh. "

               "Maybe the expression for out_w:'%s' or for out_h:'%s' is self-referencing.\n",

               s->ow_expr, s->oh_expr);

        return AVERROR(EINVAL);

    }

    s->w &= ~((1 << s->hsub) - 1);

    s->h &= ~((1 << s->vsub) - 1);






    if ((ret = av_expr_parse(&s->x_pexpr, s->x_expr, var_names,

                             NULL, NULL, NULL, NULL, 0, ctx)) < 0 ||

        (ret = av_expr_parse(&s->y_pexpr, s->y_expr, var_names,

                             NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        return AVERROR(EINVAL);



    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d -> w:%d h:%d\n",

           link->w, link->h, s->w, s->h);



    if (s->w <= 0 || s->h <= 0 ||

        s->w > link->w || s->h > link->h) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid too big or non positive size for width '%d' or height '%d'\n",

               s->w, s->h);

        return AVERROR(EINVAL);

    }



    /* set default, required in the case the first computed value for x/y is NAN */

    s->x = (link->w - s->w) / 2;

    s->y = (link->h - s->h) / 2;

    s->x &= ~((1 << s->hsub) - 1);

    s->y &= ~((1 << s->vsub) - 1);

    return 0;



fail_expr:

    av_log(NULL, AV_LOG_ERROR, "Error when evaluating the expression '%s'\n", expr);

    return ret;

}