static int config_input(AVFilterLink *inlink)

{

    AVFilterContext *ctx = inlink->dst;

    PadContext *s = ctx->priv;

    const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(inlink->format);

    uint8_t rgba_color[4];

    int ret, is_packed_rgba;

    double var_values[VARS_NB], res;

    char *expr;



    s->hsub = pix_desc->log2_chroma_w;

    s->vsub = pix_desc->log2_chroma_h;



    var_values[VAR_PI]    = M_PI;

    var_values[VAR_PHI]   = M_PHI;

    var_values[VAR_E]     = M_E;

    var_values[VAR_IN_W]  = var_values[VAR_IW] = inlink->w;

    var_values[VAR_IN_H]  = var_values[VAR_IH] = inlink->h;

    var_values[VAR_OUT_W] = var_values[VAR_OW] = NAN;

    var_values[VAR_OUT_H] = var_values[VAR_OH] = NAN;

    var_values[VAR_A]     = (double) inlink->w / inlink->h;

    var_values[VAR_HSUB]  = 1<<s->hsub;

    var_values[VAR_VSUB]  = 1<<s->vsub;



    /* evaluate width and height */

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->w_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->w = var_values[VAR_OUT_W] = var_values[VAR_OW] = res;

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->h_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->h = var_values[VAR_OUT_H] = var_values[VAR_OH] = res;

    /* evaluate the width again, as it may depend on the evaluated output height */

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->w_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->w = var_values[VAR_OUT_W] = var_values[VAR_OW] = res;



    /* evaluate x and y */

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->x_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->x = var_values[VAR_X] = res;

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->y_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->y = var_values[VAR_Y] = res;

    /* evaluate x again, as it may depend on the evaluated y value */

    if ((ret = av_expr_parse_and_eval(&res, (expr = s->x_expr),

                                      var_names, var_values,

                                      NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0)

        goto eval_fail;

    s->x = var_values[VAR_X] = res;



    /* sanity check params */

    if (s->w < 0 || s->h < 0 || s->x < 0 || s->y < 0) {

        av_log(ctx, AV_LOG_ERROR, "Negative values are not acceptable.\n");

        return AVERROR(EINVAL);

    }



    if (!s->w)

        s->w = inlink->w;

    if (!s->h)

        s->h = inlink->h;



    s->w &= ~((1 << s->hsub) - 1);

    s->h &= ~((1 << s->vsub) - 1);

    s->x &= ~((1 << s->hsub) - 1);

    s->y &= ~((1 << s->vsub) - 1);



    s->in_w = inlink->w & ~((1 << s->hsub) - 1);

    s->in_h = inlink->h & ~((1 << s->vsub) - 1);



    memcpy(rgba_color, s->color, sizeof(rgba_color));

    ff_fill_line_with_color(s->line, s->line_step, s->w, s->color,

                            inlink->format, rgba_color, &is_packed_rgba, NULL);



    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d -> w:%d h:%d x:%d y:%d color:0x%02X%02X%02X%02X[%s]\n",

           inlink->w, inlink->h, s->w, s->h, s->x, s->y,

           s->color[0], s->color[1], s->color[2], s->color[3],

           is_packed_rgba ? "rgba" : "yuva");



    if (s->x <  0 || s->y <  0                      ||

        s->w <= 0 || s->h <= 0                      ||

        (unsigned)s->x + (unsigned)inlink->w > s->w ||

        (unsigned)s->y + (unsigned)inlink->h > s->h) {

        av_log(ctx, AV_LOG_ERROR,

               "Input area %d:%d:%d:%d not within the padded area 0:0:%d:%d or zero-sized\n",

               s->x, s->y, s->x + inlink->w, s->y + inlink->h, s->w, s->h);

        return AVERROR(EINVAL);

    }



    return 0;



eval_fail:

    av_log(NULL, AV_LOG_ERROR,

           "Error when evaluating the expression '%s'\n", expr);

    return ret;



}
