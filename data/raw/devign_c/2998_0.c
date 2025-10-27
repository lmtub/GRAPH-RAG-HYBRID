static int asink_query_formats(AVFilterContext *ctx)

{

    BufferSinkContext *buf = ctx->priv;

    AVFilterFormats *formats = NULL;

    AVFilterChannelLayouts *layouts = NULL;

    unsigned i;

    int ret;



    if (buf->sample_fmts_size     % sizeof(*buf->sample_fmts)     ||

        buf->sample_rates_size    % sizeof(*buf->sample_rates)    ||

        buf->channel_layouts_size % sizeof(*buf->channel_layouts) ||

        buf->channel_counts_size  % sizeof(*buf->channel_counts)) {

        av_log(ctx, AV_LOG_ERROR, "Invalid size for format lists\n");

#define LOG_ERROR(field) \

        if (buf->field ## _size % sizeof(*buf->field)) \

            av_log(ctx, AV_LOG_ERROR, "  " #field " is %d, should be " \

                   "multiple of %d\n", \

                   buf->field ## _size, (int)sizeof(*buf->field));

        LOG_ERROR(sample_fmts);

        LOG_ERROR(sample_rates);

        LOG_ERROR(channel_layouts);

        LOG_ERROR(channel_counts);

#undef LOG_ERROR

        return AVERROR(EINVAL);

    }



    if (buf->sample_fmts_size) {

        for (i = 0; i < NB_ITEMS(buf->sample_fmts); i++)

            if ((ret = ff_add_format(&formats, buf->sample_fmts[i])) < 0)

                return ret;

        ff_set_common_formats(ctx, formats);

    }



    if (buf->channel_layouts_size || buf->channel_counts_size ||

        buf->all_channel_counts) {

        for (i = 0; i < NB_ITEMS(buf->channel_layouts); i++)

            if ((ret = ff_add_channel_layout(&layouts, buf->channel_layouts[i])) < 0)

                return ret;

        for (i = 0; i < NB_ITEMS(buf->channel_counts); i++)

            if ((ret = ff_add_channel_layout(&layouts, FF_COUNT2LAYOUT(buf->channel_counts[i]))) < 0)

                return ret;

        if (buf->all_channel_counts) {

            if (layouts)

                av_log(ctx, AV_LOG_WARNING,

                       "Conflicting all_channel_counts and list in options\n");

            else if (!(layouts = ff_all_channel_counts()))

                return AVERROR(ENOMEM);

        }

        ff_set_common_channel_layouts(ctx, layouts);

    }



    if (buf->sample_rates_size) {

        formats = NULL;

        for (i = 0; i < NB_ITEMS(buf->sample_rates); i++)

            if ((ret = ff_add_format(&formats, buf->sample_rates[i])) < 0)

                return ret;

        ff_set_common_samplerates(ctx, formats);

    }



    return 0;

}
