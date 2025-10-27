static int configure_output_audio_filter(FilterGraph *fg, OutputFilter *ofilter, AVFilterInOut *out)

{

    OutputStream *ost = ofilter->ost;

    AVCodecContext *codec  = ost->st->codec;

    AVFilterContext *last_filter = out->filter_ctx;

    int pad_idx = out->pad_idx;

    char *sample_fmts, *sample_rates, *channel_layouts;

    char name[255];

    int ret;





    snprintf(name, sizeof(name), "output stream %d:%d", ost->file_index, ost->index);

    ret = avfilter_graph_create_filter(&ofilter->filter,

                                       avfilter_get_by_name("ffabuffersink"),

                                       name, NULL, NULL, fg->graph);

    if (ret < 0)

        return ret;



#define AUTO_INSERT_FILTER(opt_name, filter_name, arg) do {                 \

    AVFilterContext *filt_ctx;                                              \

                                                                            \

    av_log(NULL, AV_LOG_INFO, opt_name " is forwarded to lavfi "            \

           "similarly to -af " filter_name "=%s.\n", arg);                  \

                                                                            \

    ret = avfilter_graph_create_filter(&filt_ctx,                           \

                                       avfilter_get_by_name(filter_name),   \

                                       filter_name, arg, NULL, fg->graph);  \

    if (ret < 0)                                                            \

        return ret;                                                         \

                                                                            \

    ret = avfilter_link(last_filter, pad_idx, filt_ctx, 0);                 \

    if (ret < 0)                                                            \

        return ret;                                                         \

                                                                            \

    last_filter = filt_ctx;                                                 \

    pad_idx = 0;                                                            \

} while (0)

    if (ost->audio_channels_mapped) {

        int i;

        AVBPrint pan_buf;

        av_bprint_init(&pan_buf, 256, 8192);

        av_bprintf(&pan_buf, "0x%"PRIx64,

                   av_get_default_channel_layout(ost->audio_channels_mapped));

        for (i = 0; i < ost->audio_channels_mapped; i++)

            if (ost->audio_channels_map[i] != -1)

                av_bprintf(&pan_buf, ":c%d=c%d", i, ost->audio_channels_map[i]);



        AUTO_INSERT_FILTER("-map_channel", "pan", pan_buf.str);

        av_bprint_finalize(&pan_buf, NULL);

    }



    if (codec->channels && !codec->channel_layout)

        codec->channel_layout = av_get_default_channel_layout(codec->channels);



    sample_fmts     = choose_sample_fmts(ost);

    sample_rates    = choose_sample_rates(ost);

    channel_layouts = choose_channel_layouts(ost);

    if (sample_fmts || sample_rates || channel_layouts) {

        AVFilterContext *format;

        char args[256];

        int len = 0;



        if (sample_fmts)

            len += snprintf(args + len, sizeof(args) - len, "sample_fmts=%s:",

                            sample_fmts);

        if (sample_rates)

            len += snprintf(args + len, sizeof(args) - len, "sample_rates=%s:",

                            sample_rates);

        if (channel_layouts)

            len += snprintf(args + len, sizeof(args) - len, "channel_layouts=%s:",

                            channel_layouts);

        args[len - 1] = 0;



        av_freep(&sample_fmts);

        av_freep(&sample_rates);

        av_freep(&channel_layouts);



        snprintf(name, sizeof(name), "audio format for output stream %d:%d",

                 ost->file_index, ost->index);

        ret = avfilter_graph_create_filter(&format,

                                           avfilter_get_by_name("aformat"),

                                           name, args, NULL, fg->graph);

        if (ret < 0)

            return ret;



        ret = avfilter_link(last_filter, pad_idx, format, 0);

        if (ret < 0)

            return ret;



        last_filter = format;

        pad_idx = 0;

    }



    if (audio_volume != 256 && 0) {

        char args[256];



        snprintf(args, sizeof(args), "%f", audio_volume / 256.);

        AUTO_INSERT_FILTER("-vol", "volume", args);

    }



    if ((ret = avfilter_link(last_filter, pad_idx, ofilter->filter, 0)) < 0)

        return ret;



    return 0;

}
