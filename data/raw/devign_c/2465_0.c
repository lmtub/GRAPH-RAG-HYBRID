static int open_input_file(OptionsContext *o, const char *filename)

{

    InputFile *f;

    AVFormatContext *ic;

    AVInputFormat *file_iformat = NULL;

    int err, i, ret;

    int64_t timestamp;

    uint8_t buf[128];

    AVDictionary **opts;

    AVDictionary *unused_opts = NULL;

    AVDictionaryEntry *e = NULL;

    int orig_nb_streams;                     // number of streams before avformat_find_stream_info



    if (o->format) {

        if (!(file_iformat = av_find_input_format(o->format))) {

            av_log(NULL, AV_LOG_FATAL, "Unknown input format: '%s'\n", o->format);

            exit_program(1);

        }

    }



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    using_stdin |= !strncmp(filename, "pipe:", 5) ||

                    !strcmp(filename, "/dev/stdin");



    /* get default parameters from command line */

    ic = avformat_alloc_context();

    if (!ic) {

        print_error(filename, AVERROR(ENOMEM));

        exit_program(1);

    }

    if (o->nb_audio_sample_rate) {

        snprintf(buf, sizeof(buf), "%d", o->audio_sample_rate[o->nb_audio_sample_rate - 1].u.i);

        av_dict_set(&o->g->format_opts, "sample_rate", buf, 0);

    }

    if (o->nb_audio_channels) {

        /* because we set audio_channels based on both the "ac" and

         * "channel_layout" options, we need to check that the specified

         * demuxer actually has the "channels" option before setting it */

        if (file_iformat && file_iformat->priv_class &&

            av_opt_find(&file_iformat->priv_class, "channels", NULL, 0,

                        AV_OPT_SEARCH_FAKE_OBJ)) {

            snprintf(buf, sizeof(buf), "%d",

                     o->audio_channels[o->nb_audio_channels - 1].u.i);

            av_dict_set(&o->g->format_opts, "channels", buf, 0);

        }

    }

    if (o->nb_frame_rates) {

        /* set the format-level framerate option;

         * this is important for video grabbers, e.g. x11 */

        if (file_iformat && file_iformat->priv_class &&

            av_opt_find(&file_iformat->priv_class, "framerate", NULL, 0,

                        AV_OPT_SEARCH_FAKE_OBJ)) {

            av_dict_set(&o->g->format_opts, "framerate",

                        o->frame_rates[o->nb_frame_rates - 1].u.str, 0);

        }

    }

    if (o->nb_frame_sizes) {

        av_dict_set(&o->g->format_opts, "video_size", o->frame_sizes[o->nb_frame_sizes - 1].u.str, 0);

    }

    if (o->nb_frame_pix_fmts)

        av_dict_set(&o->g->format_opts, "pixel_format", o->frame_pix_fmts[o->nb_frame_pix_fmts - 1].u.str, 0);



    ic->flags |= AVFMT_FLAG_NONBLOCK;

    ic->interrupt_callback = int_cb;



    /* open the input file with generic libav function */

    err = avformat_open_input(&ic, filename, file_iformat, &o->g->format_opts);

    if (err < 0) {

        print_error(filename, err);

        exit_program(1);

    }

    assert_avoptions(o->g->format_opts);



    /* apply forced codec ids */

    for (i = 0; i < ic->nb_streams; i++)

        choose_decoder(o, ic, ic->streams[i]);



    /* Set AVCodecContext options for avformat_find_stream_info */

    opts = setup_find_stream_info_opts(ic, o->g->codec_opts);

    orig_nb_streams = ic->nb_streams;



    /* If not enough info to get the stream parameters, we decode the

       first frames to get it. (used in mpeg case for example) */

    ret = avformat_find_stream_info(ic, opts);

    if (ret < 0) {

        av_log(NULL, AV_LOG_FATAL, "%s: could not find codec parameters\n", filename);

        avformat_close_input(&ic);

        exit_program(1);

    }



    timestamp = o->start_time;

    /* add the stream start time */

    if (ic->start_time != AV_NOPTS_VALUE)

        timestamp += ic->start_time;



    /* if seeking requested, we execute it */

    if (o->start_time != 0) {

        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);

        if (ret < 0) {

            av_log(NULL, AV_LOG_WARNING, "%s: could not seek to position %0.3f\n",

                   filename, (double)timestamp / AV_TIME_BASE);

        }

    }



    /* update the current parameters so that they match the one of the input stream */

    add_input_streams(o, ic);



    /* dump the file content */

    av_dump_format(ic, nb_input_files, filename, 0);



    GROW_ARRAY(input_files, nb_input_files);

    f = av_mallocz(sizeof(*f));

    if (!f)

        exit_program(1);

    input_files[nb_input_files - 1] = f;



    f->ctx        = ic;

    f->ist_index  = nb_input_streams - ic->nb_streams;

    f->ts_offset  = o->input_ts_offset - (copy_ts ? 0 : timestamp);

    f->nb_streams = ic->nb_streams;

    f->rate_emu   = o->rate_emu;



    /* check if all codec options have been used */

    unused_opts = strip_specifiers(o->g->codec_opts);

    for (i = f->ist_index; i < nb_input_streams; i++) {

        e = NULL;

        while ((e = av_dict_get(input_streams[i]->opts, "", e,

                                AV_DICT_IGNORE_SUFFIX)))

            av_dict_set(&unused_opts, e->key, NULL, 0);

    }



    e = NULL;

    while ((e = av_dict_get(unused_opts, "", e, AV_DICT_IGNORE_SUFFIX))) {

        const AVClass *class = avcodec_get_class();

        const AVOption *option = av_opt_find(&class, e->key, NULL, 0,

                                             AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ);

        if (!option)

            continue;

        if (!(option->flags & AV_OPT_FLAG_DECODING_PARAM)) {

            av_log(NULL, AV_LOG_ERROR, "Codec AVOption %s (%s) specified for "

                   "input file #%d (%s) is not a decoding option.\n", e->key,

                   option->help ? option->help : "", nb_input_files - 1,

                   filename);

            exit_program(1);

        }



        av_log(NULL, AV_LOG_WARNING, "Codec AVOption %s (%s) specified for "

               "input file #%d (%s) has not been used for any stream. The most "

               "likely reason is either wrong type (e.g. a video option with "

               "no video streams) or that it is a private option of some decoder "

               "which was not actually used for any stream.\n", e->key,

               option->help ? option->help : "", nb_input_files - 1, filename);

    }

    av_dict_free(&unused_opts);



    for (i = 0; i < o->nb_dump_attachment; i++) {

        int j;



        for (j = 0; j < ic->nb_streams; j++) {

            AVStream *st = ic->streams[j];



            if (check_stream_specifier(ic, st, o->dump_attachment[i].specifier) == 1)

                dump_attachment(st, o->dump_attachment[i].u.str);

        }

    }



    for (i = 0; i < orig_nb_streams; i++)

        av_dict_free(&opts[i]);

    av_freep(&opts);



    return 0;

}
