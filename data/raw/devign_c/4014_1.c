av_cold static int lavfi_read_header(AVFormatContext *avctx)

{

    LavfiContext *lavfi = avctx->priv_data;

    AVFilterInOut *input_links = NULL, *output_links = NULL, *inout;

    AVFilter *buffersink, *abuffersink;

    int *pix_fmts = create_all_formats(AV_PIX_FMT_NB);

    enum AVMediaType type;

    int ret = 0, i, n;



#define FAIL(ERR) { ret = ERR; goto end; }



    if (!pix_fmts)

        FAIL(AVERROR(ENOMEM));



    avfilter_register_all();



    buffersink = avfilter_get_by_name("ffbuffersink");

    abuffersink = avfilter_get_by_name("ffabuffersink");



    if (lavfi->graph_filename && lavfi->graph_str) {

        av_log(avctx, AV_LOG_ERROR,

               "Only one of the graph or graph_file options must be specified\n");

        return AVERROR(EINVAL);

    }



    if (lavfi->graph_filename) {

        uint8_t *file_buf, *graph_buf;

        size_t file_bufsize;

        ret = av_file_map(lavfi->graph_filename,

                          &file_buf, &file_bufsize, 0, avctx);

        if (ret < 0)

            return ret;



        /* create a 0-terminated string based on the read file */

        graph_buf = av_malloc(file_bufsize + 1);

        if (!graph_buf) {

            av_file_unmap(file_buf, file_bufsize);

            return AVERROR(ENOMEM);

        }

        memcpy(graph_buf, file_buf, file_bufsize);

        graph_buf[file_bufsize] = 0;

        av_file_unmap(file_buf, file_bufsize);

        lavfi->graph_str = graph_buf;

    }



    if (!lavfi->graph_str)

        lavfi->graph_str = av_strdup(avctx->filename);



    /* parse the graph, create a stream for each open output */

    if (!(lavfi->graph = avfilter_graph_alloc()))

        FAIL(AVERROR(ENOMEM));



    if ((ret = avfilter_graph_parse(lavfi->graph, lavfi->graph_str,

                                    &input_links, &output_links, avctx)) < 0)

        FAIL(ret);



    if (input_links) {

        av_log(avctx, AV_LOG_ERROR,

               "Open inputs in the filtergraph are not acceptable\n");

        FAIL(AVERROR(EINVAL));

    }



    /* count the outputs */

    for (n = 0, inout = output_links; inout; n++, inout = inout->next);



    if (!(lavfi->sink_stream_map = av_malloc(sizeof(int) * n)))

        FAIL(AVERROR(ENOMEM));

    if (!(lavfi->sink_eof = av_mallocz(sizeof(int) * n)))

        FAIL(AVERROR(ENOMEM));

    if (!(lavfi->stream_sink_map = av_malloc(sizeof(int) * n)))

        FAIL(AVERROR(ENOMEM));



    for (i = 0; i < n; i++)

        lavfi->stream_sink_map[i] = -1;



    /* parse the output link names - they need to be of the form out0, out1, ...

     * create a mapping between them and the streams */

    for (i = 0, inout = output_links; inout; i++, inout = inout->next) {

        int stream_idx;

        if (!strcmp(inout->name, "out"))

            stream_idx = 0;

        else if (sscanf(inout->name, "out%d\n", &stream_idx) != 1) {

            av_log(avctx,  AV_LOG_ERROR,

                   "Invalid outpad name '%s'\n", inout->name);

            FAIL(AVERROR(EINVAL));

        }



        if ((unsigned)stream_idx >= n) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid index was specified in output '%s', "

                   "must be a non-negative value < %d\n",

                   inout->name, n);

            FAIL(AVERROR(EINVAL));

        }



        /* is an audio or video output? */

        type = inout->filter_ctx->output_pads[inout->pad_idx].type;

        if (type != AVMEDIA_TYPE_VIDEO && type != AVMEDIA_TYPE_AUDIO) {

            av_log(avctx,  AV_LOG_ERROR,

                   "Output '%s' is not a video or audio output, not yet supported\n", inout->name);

            FAIL(AVERROR(EINVAL));

        }



        if (lavfi->stream_sink_map[stream_idx] != -1) {

            av_log(avctx,  AV_LOG_ERROR,

                   "An output with stream index %d was already specified\n",

                   stream_idx);

            FAIL(AVERROR(EINVAL));

        }

        lavfi->sink_stream_map[i] = stream_idx;

        lavfi->stream_sink_map[stream_idx] = i;

    }



    /* for each open output create a corresponding stream */

    for (i = 0, inout = output_links; inout; i++, inout = inout->next) {

        AVStream *st;

        if (!(st = avformat_new_stream(avctx, NULL)))

            FAIL(AVERROR(ENOMEM));

        st->id = i;

    }



    /* create a sink for each output and connect them to the graph */

    lavfi->sinks = av_malloc(sizeof(AVFilterContext *) * avctx->nb_streams);

    if (!lavfi->sinks)

        FAIL(AVERROR(ENOMEM));



    for (i = 0, inout = output_links; inout; i++, inout = inout->next) {

        AVFilterContext *sink;



        type = inout->filter_ctx->output_pads[inout->pad_idx].type;



        if (type == AVMEDIA_TYPE_VIDEO && ! buffersink ||

            type == AVMEDIA_TYPE_AUDIO && ! abuffersink) {

                av_log(avctx, AV_LOG_ERROR, "Missing required buffersink filter, aborting.\n");

                FAIL(AVERROR_FILTER_NOT_FOUND);

        }



        if (type == AVMEDIA_TYPE_VIDEO) {

            AVBufferSinkParams *buffersink_params = av_buffersink_params_alloc();



            buffersink_params->pixel_fmts = pix_fmts;

            ret = avfilter_graph_create_filter(&sink, buffersink,

                                               inout->name, NULL,

                                               buffersink_params, lavfi->graph);

            av_freep(&buffersink_params);



            if (ret < 0)

                goto end;

        } else if (type == AVMEDIA_TYPE_AUDIO) {

            enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_U8,

                                                  AV_SAMPLE_FMT_S16,

                                                  AV_SAMPLE_FMT_S32,

                                                  AV_SAMPLE_FMT_FLT,

                                                  AV_SAMPLE_FMT_DBL, -1 };

            AVABufferSinkParams *abuffersink_params = av_abuffersink_params_alloc();

            abuffersink_params->sample_fmts = sample_fmts;



            ret = avfilter_graph_create_filter(&sink, abuffersink,

                                               inout->name, NULL,

                                               abuffersink_params, lavfi->graph);

            av_free(abuffersink_params);

            if (ret < 0)

                goto end;

        }



        lavfi->sinks[i] = sink;

        if ((ret = avfilter_link(inout->filter_ctx, inout->pad_idx, sink, 0)) < 0)

            FAIL(ret);

    }



    /* configure the graph */

    if ((ret = avfilter_graph_config(lavfi->graph, avctx)) < 0)

        FAIL(ret);



    if (lavfi->dump_graph) {

        char *dump = avfilter_graph_dump(lavfi->graph, lavfi->dump_graph);

        fputs(dump, stderr);

        fflush(stderr);

        av_free(dump);

    }



    /* fill each stream with the information in the corresponding sink */

    for (i = 0; i < avctx->nb_streams; i++) {

        AVFilterLink *link = lavfi->sinks[lavfi->stream_sink_map[i]]->inputs[0];

        AVStream *st = avctx->streams[i];

        st->codec->codec_type = link->type;

        avpriv_set_pts_info(st, 64, link->time_base.num, link->time_base.den);

        if (link->type == AVMEDIA_TYPE_VIDEO) {

            st->codec->codec_id   = AV_CODEC_ID_RAWVIDEO;

            st->codec->pix_fmt    = link->format;

            st->codec->time_base  = link->time_base;

            st->codec->width      = link->w;

            st->codec->height     = link->h;

            st       ->sample_aspect_ratio =

            st->codec->sample_aspect_ratio = link->sample_aspect_ratio;

        } else if (link->type == AVMEDIA_TYPE_AUDIO) {

            st->codec->codec_id    = av_get_pcm_codec(link->format, -1);

            st->codec->channels    = av_get_channel_layout_nb_channels(link->channel_layout);

            st->codec->sample_fmt  = link->format;

            st->codec->sample_rate = link->sample_rate;

            st->codec->time_base   = link->time_base;

            st->codec->channel_layout = link->channel_layout;

            if (st->codec->codec_id == AV_CODEC_ID_NONE)

                av_log(avctx, AV_LOG_ERROR,

                       "Could not find PCM codec for sample format %s.\n",

                       av_get_sample_fmt_name(link->format));

        }

    }



end:

    av_free(pix_fmts);

    avfilter_inout_free(&input_links);

    avfilter_inout_free(&output_links);

    if (ret < 0)

        lavfi_read_close(avctx);

    return ret;

}
