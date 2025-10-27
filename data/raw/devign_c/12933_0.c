static int set_expr(AVExpr **pexpr, const char *expr, void *log_ctx)

{

    int ret;

    AVExpr *old = NULL;



    if (*pexpr)

        old = *pexpr;

    ret = av_expr_parse(pexpr, expr, var_names,

                        NULL, NULL, NULL, NULL, 0, log_ctx);

    if (ret < 0) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Error when evaluating the expression '%s'\n", expr);

        *pexpr = old;

        return ret;

    }



    av_expr_free(old);

    return 0;

}
