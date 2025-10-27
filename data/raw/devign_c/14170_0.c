static int filter_query_formats(AVFilterContext *ctx)

{

    int ret, i;

    AVFilterFormats *formats;

    AVFilterChannelLayouts *chlayouts;

    AVFilterFormats *samplerates;

    enum AVMediaType type = ctx->inputs  && ctx->inputs [0] ? ctx->inputs [0]->type :

                            ctx->outputs && ctx->outputs[0] ? ctx->outputs[0]->type :

                            AVMEDIA_TYPE_VIDEO;



    if ((ret = ctx->filter->query_formats(ctx)) < 0) {

        av_log(ctx, AV_LOG_ERROR, "Query format failed for '%s': %s\n",

               ctx->name, av_err2str(ret));

        return ret;

    }



    for (i = 0; i < ctx->nb_inputs; i++)

        sanitize_channel_layouts(ctx, ctx->inputs[i]->out_channel_layouts);

    for (i = 0; i < ctx->nb_outputs; i++)

        sanitize_channel_layouts(ctx, ctx->outputs[i]->in_channel_layouts);



    formats = ff_all_formats(type);

    if (!formats)

        return AVERROR(ENOMEM);

    ff_set_common_formats(ctx, formats);

    if (type == AVMEDIA_TYPE_AUDIO) {

        samplerates = ff_all_samplerates();

        if (!samplerates)

            return AVERROR(ENOMEM);

        ff_set_common_samplerates(ctx, samplerates);

        chlayouts = ff_all_channel_layouts();

        if (!chlayouts)

            return AVERROR(ENOMEM);

        ff_set_common_channel_layouts(ctx, chlayouts);

    }

    return 0;

}
