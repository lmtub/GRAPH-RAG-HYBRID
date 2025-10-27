static int configure_video_filters(AVFilterGraph *graph, VideoState *is, const char *vfilters)

{

    static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE };

    char sws_flags_str[128];

    char buffersrc_args[256];

    int ret;

    AVBufferSinkParams *buffersink_params = av_buffersink_params_alloc();

    AVFilterContext *filt_src = NULL, *filt_out = NULL, *filt_format, *filt_crop;

    AVCodecContext *codec = is->video_st->codec;



    snprintf(sws_flags_str, sizeof(sws_flags_str), "flags=%d", sws_flags);

    graph->scale_sws_opts = av_strdup(sws_flags_str);



    snprintf(buffersrc_args, sizeof(buffersrc_args),

             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",

             codec->width, codec->height, codec->pix_fmt,

             is->video_st->time_base.num, is->video_st->time_base.den,

             codec->sample_aspect_ratio.num, codec->sample_aspect_ratio.den);



    if ((ret = avfilter_graph_create_filter(&filt_src,

                                            avfilter_get_by_name("buffer"),

                                            "ffplay_buffer", buffersrc_args, NULL,

                                            graph)) < 0)

        return ret;



    buffersink_params->pixel_fmts = pix_fmts;

    ret = avfilter_graph_create_filter(&filt_out,

                                       avfilter_get_by_name("ffbuffersink"),

                                       "ffplay_buffersink", NULL, buffersink_params, graph);

    av_freep(&buffersink_params);

    if (ret < 0)

        return ret;



    /* SDL YUV code is not handling odd width/height for some driver

     * combinations, therefore we crop the picture to an even width/height. */

    if ((ret = avfilter_graph_create_filter(&filt_crop,

                                            avfilter_get_by_name("crop"),

                                            "ffplay_crop", "floor(in_w/2)*2:floor(in_h/2)*2", NULL, graph)) < 0)

        return ret;

    if ((ret = avfilter_graph_create_filter(&filt_format,

                                            avfilter_get_by_name("format"),

                                            "format", "yuv420p", NULL, graph)) < 0)

        return ret;

    if ((ret = avfilter_link(filt_crop, 0, filt_format, 0)) < 0)

        return ret;

    if ((ret = avfilter_link(filt_format, 0, filt_out, 0)) < 0)

        return ret;



    if ((ret = configure_filtergraph(graph, vfilters, filt_src, filt_crop)) < 0)

        return ret;



    is->in_video_filter  = filt_src;

    is->out_video_filter = filt_out;



    return ret;

}
