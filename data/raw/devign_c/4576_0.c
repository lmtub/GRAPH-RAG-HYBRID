static int configure_input_video_filter(FilterGraph *fg, InputFilter *ifilter,

                                        AVFilterInOut *in)

{

    AVFilterContext *last_filter;

    const AVFilter *buffer_filt = avfilter_get_by_name("buffer");

    InputStream *ist = ifilter->ist;

    InputFile     *f = input_files[ist->file_index];

    AVRational tb = ist->framerate.num ? av_inv_q(ist->framerate) :

                                         ist->st->time_base;

    AVRational sar;

    char args[255], name[255];

    int ret, pad_idx = 0;



    sar = ist->st->sample_aspect_ratio.num ?

          ist->st->sample_aspect_ratio :

          ist->st->codec->sample_aspect_ratio;

    snprintf(args, sizeof(args), "%d:%d:%d:%d:%d:%d:%d", ist->st->codec->width,

             ist->st->codec->height, ist->st->codec->pix_fmt,

             tb.num, tb.den, sar.num, sar.den);

    snprintf(name, sizeof(name), "graph %d input from stream %d:%d", fg->index,

             ist->file_index, ist->st->index);



    if ((ret = avfilter_graph_create_filter(&ifilter->filter, buffer_filt, name,

                                            args, NULL, fg->graph)) < 0)

        return ret;

    last_filter = ifilter->filter;



    if (ist->framerate.num) {

        AVFilterContext *setpts;



        snprintf(name, sizeof(name), "force CFR for input from stream %d:%d",

                 ist->file_index, ist->st->index);

        if ((ret = avfilter_graph_create_filter(&setpts,

                                                avfilter_get_by_name("setpts"),

                                                name, "N", NULL,

                                                fg->graph)) < 0)

            return ret;



        if ((ret = avfilter_link(last_filter, 0, setpts, 0)) < 0)

            return ret;



        last_filter = setpts;

    }



    snprintf(name, sizeof(name), "trim for input stream %d:%d",

             ist->file_index, ist->st->index);

    ret = insert_trim(((f->start_time == AV_NOPTS_VALUE) || !f->accurate_seek) ?

                      AV_NOPTS_VALUE : 0, INT64_MAX, &last_filter, &pad_idx, name);

    if (ret < 0)

        return ret;



    if ((ret = avfilter_link(last_filter, 0, in->filter_ctx, in->pad_idx)) < 0)

        return ret;

    return 0;

}
