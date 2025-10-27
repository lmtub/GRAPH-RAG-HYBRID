static int configure_output_video_filter(FilterGraph *fg, OutputFilter *ofilter, AVFilterInOut *out)

{

    char *pix_fmts;

    OutputStream *ost = ofilter->ost;

    AVCodecContext *codec = ost->st->codec;

    AVFilterContext *last_filter = out->filter_ctx;

    int pad_idx = out->pad_idx;

    int ret;

    char name[255];



    snprintf(name, sizeof(name), "output stream %d:%d", ost->file_index, ost->index);

    ret = avfilter_graph_create_filter(&ofilter->filter,

                                       avfilter_get_by_name("buffersink"),

                                       name, NULL, pix_fmts, fg->graph);

    if (ret < 0)

        return ret;



    if (codec->width || codec->height) {

        char args[255];

        AVFilterContext *filter;



        snprintf(args, sizeof(args), "%d:%d:flags=0x%X",

                 codec->width,

                 codec->height,

                 (unsigned)ost->sws_flags);

        snprintf(name, sizeof(name), "scaler for output stream %d:%d",

                 ost->file_index, ost->index);

        if ((ret = avfilter_graph_create_filter(&filter, avfilter_get_by_name("scale"),

                                                name, args, NULL, fg->graph)) < 0)

            return ret;

        if ((ret = avfilter_link(last_filter, pad_idx, filter, 0)) < 0)

            return ret;



        last_filter = filter;

        pad_idx = 0;

    }



    if ((pix_fmts = choose_pix_fmts(ost))) {

        AVFilterContext *filter;

        snprintf(name, sizeof(name), "pixel format for output stream %d:%d",

                 ost->file_index, ost->index);

        if ((ret = avfilter_graph_create_filter(&filter,

                                                avfilter_get_by_name("format"),

                                                "format", pix_fmts, NULL,

                                                fg->graph)) < 0)

            return ret;

        if ((ret = avfilter_link(last_filter, pad_idx, filter, 0)) < 0)

            return ret;



        last_filter = filter;

        pad_idx     = 0;

        av_freep(&pix_fmts);

    }



    if (ost->frame_rate.num) {

        AVFilterContext *fps;

        char args[255];



        snprintf(args, sizeof(args), "fps=%d/%d", ost->frame_rate.num,

                 ost->frame_rate.den);

        snprintf(name, sizeof(name), "fps for output stream %d:%d",

                 ost->file_index, ost->index);

        ret = avfilter_graph_create_filter(&fps, avfilter_get_by_name("fps"),

                                           name, args, NULL, fg->graph);

        if (ret < 0)

            return ret;



        ret = avfilter_link(last_filter, pad_idx, fps, 0);

        if (ret < 0)

            return ret;

        last_filter = fps;

        pad_idx = 0;

    }



    if ((ret = avfilter_link(last_filter, pad_idx, ofilter->filter, 0)) < 0)

        return ret;



    return 0;

}
