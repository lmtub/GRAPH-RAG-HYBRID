static int config_output(AVFilterLink *outlink) {

    static const char hdcd_baduse[] =

        "The HDCD filter is unlikely to produce a desirable result in this context.";

    AVFilterContext *ctx = outlink->src;

    HDCDContext *s = ctx->priv;

    AVFilterLink *lk = outlink;

    while(lk != NULL) {

        AVFilterContext *nextf = lk->dst;

        if (lk->type == AVMEDIA_TYPE_AUDIO) {

            if (lk->format == AV_SAMPLE_FMT_S16 || lk->format == AV_SAMPLE_FMT_U8) {

                av_log(ctx, AV_LOG_WARNING, "s24 output is being truncated to %s at %s.\n",

                    av_get_sample_fmt_name(lk->format),

                    (nextf->name) ? nextf->name : "<unknown>"

                    );

                s->bad_config = 1;

                break;

            }

        }

        lk = (nextf->outputs) ? nextf->outputs[0] : NULL;

    }

    if (s->bad_config)

        av_log(ctx, AV_LOG_WARNING, "%s\n", hdcd_baduse);

    return 0;

}
