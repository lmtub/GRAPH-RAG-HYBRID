static int opt_input_file(const char *opt, const char *filename)

{

    AVFormatContext *ic;

    AVInputFormat *file_iformat = NULL;

    int err, i, ret, rfps, rfps_base;

    int64_t timestamp;

    uint8_t buf[128];

    AVDictionary **opts;

    int orig_nb_streams;                     // number of streams before avformat_find_stream_info



    if (last_asked_format) {

        if (!(file_iformat = av_find_input_format(last_asked_format))) {

            fprintf(stderr, "Unknown input format: '%s'\n", last_asked_format);

            ffmpeg_exit(1);

        }

        last_asked_format = NULL;

    }



    if (!strcmp(filename, "-"))

        filename = "pipe:";



    using_stdin |= !strncmp(filename, "pipe:", 5) ||

                    !strcmp(filename, "/dev/stdin");



    /* get default parameters from command line */

    ic = avformat_alloc_context();

    if (!ic) {

        print_error(filename, AVERROR(ENOMEM));

        ffmpeg_exit(1);

    }

    if (audio_sample_rate) {

        snprintf(buf, sizeof(buf), "%d", audio_sample_rate);

        av_dict_set(&format_opts, "sample_rate", buf, 0);

    }

    if (audio_channels) {

        snprintf(buf, sizeof(buf), "%d", audio_channels);

        av_dict_set(&format_opts, "channels", buf, 0);

    }

    if (frame_rate.num) {

        snprintf(buf, sizeof(buf), "%d/%d", frame_rate.num, frame_rate.den);

        av_dict_set(&format_opts, "framerate", buf, 0);

    }

    if (frame_width && frame_height) {

        snprintf(buf, sizeof(buf), "%dx%d", frame_width, frame_height);

        av_dict_set(&format_opts, "video_size", buf, 0);

    }

    if (frame_pix_fmt != PIX_FMT_NONE)

        av_dict_set(&format_opts, "pixel_format", av_get_pix_fmt_name(frame_pix_fmt), 0);



    ic->video_codec_id   =

        find_codec_or_die(video_codec_name   , AVMEDIA_TYPE_VIDEO   , 0);

    ic->audio_codec_id   =

        find_codec_or_die(audio_codec_name   , AVMEDIA_TYPE_AUDIO   , 0);

    ic->subtitle_codec_id=

        find_codec_or_die(subtitle_codec_name, AVMEDIA_TYPE_SUBTITLE, 0);

    ic->flags |= AVFMT_FLAG_NONBLOCK;



    /* open the input file with generic libav function */

    err = avformat_open_input(&ic, filename, file_iformat, &format_opts);

    if (err < 0) {

        print_error(filename, err);

        ffmpeg_exit(1);

    }

    assert_avoptions(format_opts);



    if(opt_programid) {

        int i, j;

        int found=0;

        for(i=0; i<ic->nb_streams; i++){

            ic->streams[i]->discard= AVDISCARD_ALL;

        }

        for(i=0; i<ic->nb_programs; i++){

            AVProgram *p= ic->programs[i];

            if(p->id != opt_programid){

                p->discard = AVDISCARD_ALL;

            }else{

                found=1;

                for(j=0; j<p->nb_stream_indexes; j++){

                    ic->streams[p->stream_index[j]]->discard= AVDISCARD_DEFAULT;

                }

            }

        }

        if(!found){

            fprintf(stderr, "Specified program id not found\n");

            ffmpeg_exit(1);

        }

        opt_programid=0;

    }



    if (loop_input) {

        av_log(NULL, AV_LOG_WARNING, "-loop_input is deprecated, use -loop 1\n");

        ic->loop_input = loop_input;

    }



    /* Set AVCodecContext options for avformat_find_stream_info */

    opts = setup_find_stream_info_opts(ic, codec_opts);

    orig_nb_streams = ic->nb_streams;



    /* If not enough info to get the stream parameters, we decode the

       first frames to get it. (used in mpeg case for example) */

    ret = avformat_find_stream_info(ic, opts);

    if (ret < 0 && verbose >= 0) {

        fprintf(stderr, "%s: could not find codec parameters\n", filename);

        av_close_input_file(ic);

        ffmpeg_exit(1);

    }



    timestamp = start_time;

    /* add the stream start time */

    if (ic->start_time != AV_NOPTS_VALUE)

        timestamp += ic->start_time;



    /* if seeking requested, we execute it */

    if (start_time != 0) {

        ret = av_seek_frame(ic, -1, timestamp, AVSEEK_FLAG_BACKWARD);

        if (ret < 0) {

            fprintf(stderr, "%s: could not seek to position %0.3f\n",

                    filename, (double)timestamp / AV_TIME_BASE);

        }

        /* reset seek info */

        start_time = 0;

    }



    /* update the current parameters so that they match the one of the input stream */

    for(i=0;i<ic->nb_streams;i++) {

        AVStream *st = ic->streams[i];

        AVCodecContext *dec = st->codec;

        InputStream *ist;



        dec->thread_count = thread_count;



        input_streams = grow_array(input_streams, sizeof(*input_streams), &nb_input_streams, nb_input_streams + 1);

        ist = &input_streams[nb_input_streams - 1];

        ist->st = st;

        ist->file_index = nb_input_files;

        ist->discard = 1;

        ist->opts = filter_codec_opts(codec_opts, ist->st->codec->codec_id, 0);



        if (i < nb_ts_scale)

            ist->ts_scale = ts_scale[i];



        switch (dec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ist->dec = avcodec_find_decoder_by_name(audio_codec_name);

            if(!ist->dec)

                ist->dec = avcodec_find_decoder(dec->codec_id);

            if(audio_disable)

                st->discard= AVDISCARD_ALL;

            break;

        case AVMEDIA_TYPE_VIDEO:

            ist->dec= avcodec_find_decoder_by_name(video_codec_name);

            if(!ist->dec)

                ist->dec = avcodec_find_decoder(dec->codec_id);

            rfps      = ic->streams[i]->r_frame_rate.num;

            rfps_base = ic->streams[i]->r_frame_rate.den;

            if (dec->lowres) {

                dec->flags |= CODEC_FLAG_EMU_EDGE;

            }

            if(me_threshold)

                dec->debug |= FF_DEBUG_MV;



            if (dec->time_base.den != rfps*dec->ticks_per_frame || dec->time_base.num != rfps_base) {



                if (verbose >= 0)

                    fprintf(stderr,"\nSeems stream %d codec frame rate differs from container frame rate: %2.2f (%d/%d) -> %2.2f (%d/%d)\n",

                            i, (float)dec->time_base.den / dec->time_base.num, dec->time_base.den, dec->time_base.num,



                    (float)rfps / rfps_base, rfps, rfps_base);

            }



            if(video_disable)

                st->discard= AVDISCARD_ALL;

            else if(video_discard)

                st->discard= video_discard;

            break;

        case AVMEDIA_TYPE_DATA:

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            ist->dec = avcodec_find_decoder_by_name(subtitle_codec_name);

            if(!ist->dec)

                ist->dec = avcodec_find_decoder(dec->codec_id);

            if(subtitle_disable)

                st->discard = AVDISCARD_ALL;

            break;

        case AVMEDIA_TYPE_ATTACHMENT:

        case AVMEDIA_TYPE_UNKNOWN:

            break;

        default:

            abort();

        }

    }



    /* dump the file content */

    if (verbose >= 0)

        av_dump_format(ic, nb_input_files, filename, 0);



    input_files = grow_array(input_files, sizeof(*input_files), &nb_input_files, nb_input_files + 1);

    input_files[nb_input_files - 1].ctx        = ic;

    input_files[nb_input_files - 1].ist_index  = nb_input_streams - ic->nb_streams;

    input_files[nb_input_files - 1].ts_offset  = input_ts_offset - (copy_ts ? 0 : timestamp);



    top_field_first = -1;

    frame_rate    = (AVRational){0, 0};

    frame_pix_fmt = PIX_FMT_NONE;

    frame_height = 0;

    frame_width  = 0;

    audio_sample_rate = 0;

    audio_channels    = 0;

    audio_sample_fmt  = AV_SAMPLE_FMT_NONE;

    av_freep(&ts_scale);

    nb_ts_scale = 0;



    for (i = 0; i < orig_nb_streams; i++)

        av_dict_free(&opts[i]);

    av_freep(&opts);

    av_freep(&video_codec_name);

    av_freep(&audio_codec_name);

    av_freep(&subtitle_codec_name);

    uninit_opts();

    init_opts();

    return 0;

}
