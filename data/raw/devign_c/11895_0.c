static av_cold int asink_init(AVFilterContext *ctx, void *opaque)

{

    BufferSinkContext *buf = ctx->priv;

    AVABufferSinkParams *params = opaque;



    if (params && params->sample_fmts) {

        buf->sample_fmts = ff_copy_int_list(params->sample_fmts);

        if (!buf->sample_fmts)

            return AVERROR(ENOMEM);

    }

    if (params && params->sample_rates) {

        buf->sample_rates = ff_copy_int_list(params->sample_rates);

        if (!buf->sample_rates)

            return AVERROR(ENOMEM);

    }

    if (params && (params->channel_layouts || params->channel_counts)) {

        if (params->all_channel_counts) {

            av_log(ctx, AV_LOG_ERROR,

                   "Conflicting all_channel_counts and list in parameters\n");

            return AVERROR(EINVAL);

        }

        buf->channel_layouts = concat_channels_lists(params->channel_layouts,

                                                     params->channel_counts);

        if (!buf->channel_layouts)

            return AVERROR(ENOMEM);

    }

    if (params)

        buf->all_channel_counts = params->all_channel_counts;

    return common_init(ctx);

}
