static int init(AVFilterContext *ctx, const char *args)
{
    EvalContext *eval = ctx->priv;
    char *args1 = av_strdup(args);
    char *expr, *buf, *bufptr;
    int ret, i;
    eval->class = &aevalsrc_class;
    av_opt_set_defaults(eval);
    /* parse expressions */
    buf = args1;
    i = 0;
    while (expr = av_strtok(buf, ":", &bufptr)) {
        ret = av_expr_parse(&eval->expr[i], expr, var_names,
                            NULL, NULL, NULL, NULL, 0, ctx);
        if (ret < 0)
        i++;
        if (bufptr && *bufptr == ':') { /* found last expression */
            bufptr++;
            break;
        buf = NULL;
    eval->nb_channels = i;
    if (bufptr && (ret = av_set_options_string(eval, bufptr, "=", ":")) < 0)
    if (eval->chlayout_str) {
        int n;
        ret = ff_parse_channel_layout(&eval->chlayout, eval->chlayout_str, ctx);
        if (ret < 0)
        n = av_get_channel_layout_nb_channels(eval->chlayout);
        if (n != eval->nb_channels) {
            av_log(ctx, AV_LOG_ERROR,
                   "Mismatch between the specified number of channels '%d' "
                   "and the number of channels '%d' in the specified channel layout '%s'\n",
                   eval->nb_channels, n, eval->chlayout_str);
            ret = AVERROR(EINVAL);
    } else {
        /* guess channel layout from nb expressions/channels */
        eval->chlayout = av_get_default_channel_layout(eval->nb_channels);
        if (!eval->chlayout) {
            av_log(ctx, AV_LOG_ERROR, "Invalid number of channels '%d' provided\n",
                   eval->nb_channels);
            ret = AVERROR(EINVAL);
    if ((ret = ff_parse_sample_rate(&eval->sample_rate, eval->sample_rate_str, ctx)))
    eval->duration = -1;
    if (eval->duration_str) {
        int64_t us = -1;
        if ((ret = av_parse_time(&us, eval->duration_str, 1)) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Invalid duration: '%s'\n", eval->duration_str);
        eval->duration = (double)us / 1000000;
    eval->n = 0;
end:
    av_free(args1);
    return ret;