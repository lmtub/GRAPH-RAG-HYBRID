static void opt_input_file(const char *filename)

{

    AVFormatContext *ic;

    AVFormatParameters params, *ap = &params;

    AVInputFormat *file_iformat = NULL;

    int err, i, ret, rfps, rfps_base;

    int64_t timestamp;



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



    memset(ap, 0, sizeof(*ap));

    ap->prealloced_context = 1;

    ap->sample_rate = audio_sample_rate;

    ap->channels = audio_channels;

    ap->time_base.den = frame_rate.num;

    ap->time_base.num = frame_rate.den;

    ap->width = frame_width;

    ap->height = frame_height;

    ap->pix_fmt = frame_pix_fmt;

   // ap->sample_fmt = audio_sample_fmt; //FIXME:not implemented in libavformat

    ap->channel = video_channel;

    ap->standard = video_standard;



    set_context_opts(ic, avformat_opts, AV_OPT_FLAG_DECODING_PARAM, NULL);



    ic->video_codec_id   =

        find_codec_or_die(video_codec_name   , AVMEDIA_TYPE_VIDEO   , 0,

                          avcodec_opts[AVMEDIA_TYPE_VIDEO   ]->strict_std_compliance);

    ic->audio_codec_id   =

        find_codec_or_die(audio_codec_name   , AVMEDIA_TYPE_AUDIO   , 0,

                          avcodec_opts[AVMEDIA_TYPE_AUDIO   ]->strict_std_compliance);

    ic->subtitle_codec_id=

        find_codec_or_die(subtitle_codec_name, AVMEDIA_TYPE_SUBTITLE, 0,

                          avcodec_opts[AVMEDIA_TYPE_SUBTITLE]->strict_std_compliance);

    ic->flags |= AVFMT_FLAG_NONBLOCK;



    if(pgmyuv_compatibility_hack)

        ic->video_codec_id= CODEC_ID_PGMYUV;



    /* open the input file with generic libav function */

    err = av_open_input_file(&ic, filename, file_iformat, 0, ap);

    if (err < 0) {

        print_error(filename, err);

        ffmpeg_exit(1);

    }

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



    ic->loop_input = loop_input;



    /* If not enough info to get the stream parameters, we decode the

       first frames to get it. (used in mpeg case for example) */

    ret = av_find_stream_info(ic);

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

        avcodec_thread_init(dec, thread_count);

        input_codecs = grow_array(input_codecs, sizeof(*input_codecs), &nb_input_codecs, nb_input_codecs + 1);

        switch (dec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            input_codecs[nb_input_codecs-1] = avcodec_find_decoder_by_name(audio_codec_name);

            set_context_opts(dec, avcodec_opts[AVMEDIA_TYPE_AUDIO], AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_DECODING_PARAM, input_codecs[nb_input_codecs-1]);

            //fprintf(stderr, "\nInput Audio channels: %d", dec->channels);

            channel_layout    = dec->channel_layout;

            audio_channels    = dec->channels;

            audio_sample_rate = dec->sample_rate;

            audio_sample_fmt  = dec->sample_fmt;

            if(audio_disable)

                st->discard= AVDISCARD_ALL;

            /* Note that av_find_stream_info can add more streams, and we

             * currently have no chance of setting up lowres decoding

             * early enough for them. */

            if (dec->lowres)

                audio_sample_rate >>= dec->lowres;

            break;

        case AVMEDIA_TYPE_VIDEO:

            input_codecs[nb_input_codecs-1] = avcodec_find_decoder_by_name(video_codec_name);

            set_context_opts(dec, avcodec_opts[AVMEDIA_TYPE_VIDEO], AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM, input_codecs[nb_input_codecs-1]);

            frame_height = dec->height;

            frame_width  = dec->width;

            if(ic->streams[i]->sample_aspect_ratio.num)

                frame_aspect_ratio=av_q2d(ic->streams[i]->sample_aspect_ratio);

            else

                frame_aspect_ratio=av_q2d(dec->sample_aspect_ratio);

            frame_aspect_ratio *= (float) dec->width / dec->height;

            frame_pix_fmt = dec->pix_fmt;

            rfps      = ic->streams[i]->r_frame_rate.num;

            rfps_base = ic->streams[i]->r_frame_rate.den;

            if (dec->lowres) {

                dec->flags |= CODEC_FLAG_EMU_EDGE;

                frame_height >>= dec->lowres;

                frame_width  >>= dec->lowres;

            }

            if(me_threshold)

                dec->debug |= FF_DEBUG_MV;



            if (dec->time_base.den != rfps*dec->ticks_per_frame || dec->time_base.num != rfps_base) {



                if (verbose >= 0)

                    fprintf(stderr,"\nSeems stream %d codec frame rate differs from container frame rate: %2.2f (%d/%d) -> %2.2f (%d/%d)\n",

                            i, (float)dec->time_base.den / dec->time_base.num, dec->time_base.den, dec->time_base.num,



                    (float)rfps / rfps_base, rfps, rfps_base);

            }

            /* update the current frame rate to match the stream frame rate */

            frame_rate.num = rfps;

            frame_rate.den = rfps_base;



            if(video_disable)

                st->discard= AVDISCARD_ALL;

            else if(video_discard)

                st->discard= video_discard;

            break;

        case AVMEDIA_TYPE_DATA:

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            input_codecs[nb_input_codecs-1] = avcodec_find_decoder_by_name(subtitle_codec_name);

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



    input_files[nb_input_files] = ic;

    input_files_ts_offset[nb_input_files] = input_ts_offset - (copy_ts ? 0 : timestamp);

    /* dump the file content */

    if (verbose >= 0)

        dump_format(ic, nb_input_files, filename, 0);



    nb_input_files++;



    video_channel = 0;



    av_freep(&video_codec_name);

    av_freep(&audio_codec_name);

    av_freep(&subtitle_codec_name);

}
