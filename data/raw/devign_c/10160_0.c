static int process_command(AVFilterContext *ctx, const char *cmd, const char *args,

                           char *res, int res_len, int flags)

{

    OverlayContext *over = ctx->priv;

    int ret;



    if      (!strcmp(cmd, "x"))

        ret = set_expr(&over->x_pexpr, args, ctx);

    else if (!strcmp(cmd, "y"))

        ret = set_expr(&over->y_pexpr, args, ctx);

    else if (!strcmp(cmd, "enable"))

        ret = set_expr(&over->enable_pexpr, args, ctx);

    else

        ret = AVERROR(ENOSYS);



    if (ret < 0)

        return ret;



    if (over->eval_mode == EVAL_MODE_INIT) {

        eval_expr(ctx, EVAL_ALL);

        av_log(ctx, AV_LOG_VERBOSE, "x:%f xi:%d y:%f yi:%d enable:%f\n",

               over->var_values[VAR_X], over->x,

               over->var_values[VAR_Y], over->y,

               over->enable);

    }

    return ret;

}
