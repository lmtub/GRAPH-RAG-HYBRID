static int transcode_init(OutputFile *output_files,

                          int nb_output_files,

                          InputFile *input_files,

                          int nb_input_files)

{

    int ret = 0, i, j;

    AVFormatContext *os;

    AVCodecContext *codec, *icodec;

    OutputStream *ost;

    InputStream *ist;

    char error[1024];

    int want_sdp = 1;



    /* init framerate emulation */

    for (i = 0; i < nb_input_files; i++) {

        InputFile *ifile = &input_files[i];

        if (ifile->rate_emu)

            for (j = 0; j < ifile->nb_streams; j++)

                input_streams[j + ifile->ist_index].start = av_gettime();

    }



    /* output stream init */

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i].ctx;

        if (!os->nb_streams && !(os->oformat->flags & AVFMT_NOSTREAMS)) {

            av_dump_format(os, i, os->filename, 1);

            fprintf(stderr, "Output file #%d does not contain any stream\n", i);

            return AVERROR(EINVAL);

        }

    }



    /* for each output stream, we compute the right encoding parameters */

    for (i = 0; i < nb_output_streams; i++) {

        ost = &output_streams[i];

        os = output_files[ost->file_index].ctx;

        ist = &input_streams[ost->source_index];



        codec = ost->st->codec;

        icodec = ist->st->codec;



        ost->st->disposition = ist->st->disposition;

        codec->bits_per_raw_sample= icodec->bits_per_raw_sample;

        codec->chroma_sample_location = icodec->chroma_sample_location;



        if (ost->st->stream_copy) {

            uint64_t extra_size = (uint64_t)icodec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE;



            if (extra_size > INT_MAX) {

                return AVERROR(EINVAL);

            }



            /* if stream_copy is selected, no need to decode or encode */

            codec->codec_id = icodec->codec_id;

            codec->codec_type = icodec->codec_type;



            if(!codec->codec_tag){

                if(   !os->oformat->codec_tag

                   || av_codec_get_id (os->oformat->codec_tag, icodec->codec_tag) == codec->codec_id

                   || av_codec_get_tag(os->oformat->codec_tag, icodec->codec_id) <= 0)

                    codec->codec_tag = icodec->codec_tag;

            }



            codec->bit_rate = icodec->bit_rate;

            codec->rc_max_rate    = icodec->rc_max_rate;

            codec->rc_buffer_size = icodec->rc_buffer_size;

            codec->extradata= av_mallocz(extra_size);

            if (!codec->extradata) {

                return AVERROR(ENOMEM);

            }

            memcpy(codec->extradata, icodec->extradata, icodec->extradata_size);

            codec->extradata_size= icodec->extradata_size;

            if(!copy_tb && av_q2d(icodec->time_base)*icodec->ticks_per_frame > av_q2d(ist->st->time_base) && av_q2d(ist->st->time_base) < 1.0/500){

                codec->time_base = icodec->time_base;

                codec->time_base.num *= icodec->ticks_per_frame;

                av_reduce(&codec->time_base.num, &codec->time_base.den,

                          codec->time_base.num, codec->time_base.den, INT_MAX);

            }else

                codec->time_base = ist->st->time_base;

            switch(codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                if(audio_volume != 256) {

                    fprintf(stderr,"-acodec copy and -vol are incompatible (frames are not decoded)\n");

                    exit_program(1);

                }

                codec->channel_layout = icodec->channel_layout;

                codec->sample_rate = icodec->sample_rate;

                codec->channels = icodec->channels;

                codec->frame_size = icodec->frame_size;

                codec->audio_service_type = icodec->audio_service_type;

                codec->block_align= icodec->block_align;

                if(codec->block_align == 1 && codec->codec_id == CODEC_ID_MP3)

                    codec->block_align= 0;

                if(codec->codec_id == CODEC_ID_AC3)

                    codec->block_align= 0;

                break;

            case AVMEDIA_TYPE_VIDEO:

                codec->pix_fmt = icodec->pix_fmt;

                codec->width = icodec->width;

                codec->height = icodec->height;

                codec->has_b_frames = icodec->has_b_frames;

                if (!codec->sample_aspect_ratio.num) {

                    codec->sample_aspect_ratio =

                    ost->st->sample_aspect_ratio =

                        ist->st->sample_aspect_ratio.num ? ist->st->sample_aspect_ratio :

                        ist->st->codec->sample_aspect_ratio.num ?

                        ist->st->codec->sample_aspect_ratio : (AVRational){0, 1};

                }

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                codec->width = icodec->width;

                codec->height = icodec->height;

                break;

            case AVMEDIA_TYPE_DATA:

                break;

            default:

                abort();

            }

        } else {

            if (!ost->enc)

                ost->enc = avcodec_find_encoder(ost->st->codec->codec_id);

            switch(codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                ost->fifo= av_fifo_alloc(1024);

                if (!ost->fifo) {

                    return AVERROR(ENOMEM);

                }

                ost->reformat_pair = MAKE_SFMT_PAIR(AV_SAMPLE_FMT_NONE,AV_SAMPLE_FMT_NONE);

                if (!codec->sample_rate) {

                    codec->sample_rate = icodec->sample_rate;

                    if (icodec->lowres)

                        codec->sample_rate >>= icodec->lowres;

                }

                choose_sample_rate(ost->st, ost->enc);

                codec->time_base = (AVRational){1, codec->sample_rate};

                if (codec->sample_fmt == AV_SAMPLE_FMT_NONE)

                    codec->sample_fmt = icodec->sample_fmt;

                choose_sample_fmt(ost->st, ost->enc);

                if (!codec->channels)

                    codec->channels = icodec->channels;

                codec->channel_layout = icodec->channel_layout;

                if (av_get_channel_layout_nb_channels(codec->channel_layout) != codec->channels)

                    codec->channel_layout = 0;

                ost->audio_resample = codec->sample_rate != icodec->sample_rate || audio_sync_method > 1;

                icodec->request_channels = codec->channels;

                ist->decoding_needed = 1;

                ost->encoding_needed = 1;

                ost->resample_sample_fmt  = icodec->sample_fmt;

                ost->resample_sample_rate = icodec->sample_rate;

                ost->resample_channels    = icodec->channels;

                break;

            case AVMEDIA_TYPE_VIDEO:

                if (codec->pix_fmt == PIX_FMT_NONE)

                    codec->pix_fmt = icodec->pix_fmt;

                choose_pixel_fmt(ost->st, ost->enc);



                if (ost->st->codec->pix_fmt == PIX_FMT_NONE) {

                    fprintf(stderr, "Video pixel format is unknown, stream cannot be encoded\n");

                    exit_program(1);

                }



                if (!codec->width || !codec->height) {

                    codec->width  = icodec->width;

                    codec->height = icodec->height;

                }



                ost->video_resample = codec->width   != icodec->width  ||

                                      codec->height  != icodec->height ||

                                      codec->pix_fmt != icodec->pix_fmt;

                if (ost->video_resample) {

#if !CONFIG_AVFILTER

                    avcodec_get_frame_defaults(&ost->pict_tmp);

                    if(avpicture_alloc((AVPicture*)&ost->pict_tmp, codec->pix_fmt,

                                       codec->width, codec->height)) {

                        fprintf(stderr, "Cannot allocate temp picture, check pix fmt\n");

                        exit_program(1);

                    }

                    ost->img_resample_ctx = sws_getContext(

                        icodec->width,

                        icodec->height,

                        icodec->pix_fmt,

                        codec->width,

                        codec->height,

                        codec->pix_fmt,

                        ost->sws_flags, NULL, NULL, NULL);

                    if (ost->img_resample_ctx == NULL) {

                        fprintf(stderr, "Cannot get resampling context\n");

                        exit_program(1);

                    }

#endif

                    codec->bits_per_raw_sample= 0;

                }



                ost->resample_height = icodec->height;

                ost->resample_width  = icodec->width;

                ost->resample_pix_fmt= icodec->pix_fmt;

                ost->encoding_needed = 1;

                ist->decoding_needed = 1;



                if (!ost->frame_rate.num)

                    ost->frame_rate = ist->st->r_frame_rate.num ? ist->st->r_frame_rate : (AVRational){25,1};

                if (ost->enc && ost->enc->supported_framerates && !force_fps) {

                    int idx = av_find_nearest_q_idx(ost->frame_rate, ost->enc->supported_framerates);

                    ost->frame_rate = ost->enc->supported_framerates[idx];

                }

                codec->time_base = (AVRational){ost->frame_rate.den, ost->frame_rate.num};



#if CONFIG_AVFILTER

                if (configure_video_filters(ist, ost)) {

                    fprintf(stderr, "Error opening filters!\n");

                    exit(1);

                }

#endif

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                ost->encoding_needed = 1;

                ist->decoding_needed = 1;

                break;

            default:

                abort();

                break;

            }

            /* two pass mode */

            if (ost->encoding_needed &&

                (codec->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2))) {

                char logfilename[1024];

                FILE *f;



                snprintf(logfilename, sizeof(logfilename), "%s-%d.log",

                         pass_logfilename_prefix ? pass_logfilename_prefix : DEFAULT_PASS_LOGFILENAME_PREFIX,

                         i);

                if (codec->flags & CODEC_FLAG_PASS1) {

                    f = fopen(logfilename, "wb");

                    if (!f) {

                        fprintf(stderr, "Cannot write log file '%s' for pass-1 encoding: %s\n", logfilename, strerror(errno));

                        exit_program(1);

                    }

                    ost->logfile = f;

                } else {

                    char  *logbuffer;

                    size_t logbuffer_size;

                    if (read_file(logfilename, &logbuffer, &logbuffer_size) < 0) {

                        fprintf(stderr, "Error reading log file '%s' for pass-2 encoding\n", logfilename);

                        exit_program(1);

                    }

                    codec->stats_in = logbuffer;

                }

            }

        }

        if(codec->codec_type == AVMEDIA_TYPE_VIDEO){

            int size= codec->width * codec->height;

            bit_buffer_size= FFMAX(bit_buffer_size, 6*size + 200);

        }

    }



    if (!bit_buffer)

        bit_buffer = av_malloc(bit_buffer_size);

    if (!bit_buffer) {

        fprintf(stderr, "Cannot allocate %d bytes output buffer\n",

                bit_buffer_size);

        return AVERROR(ENOMEM);

    }



    /* open each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = &output_streams[i];

        if (ost->encoding_needed) {

            AVCodec *codec = ost->enc;

            AVCodecContext *dec = input_streams[ost->source_index].st->codec;

            if (!codec) {

                snprintf(error, sizeof(error), "Encoder (codec id %d) not found for output stream #%d.%d",

                         ost->st->codec->codec_id, ost->file_index, ost->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            if (dec->subtitle_header) {

                ost->st->codec->subtitle_header = av_malloc(dec->subtitle_header_size);

                if (!ost->st->codec->subtitle_header) {

                    ret = AVERROR(ENOMEM);

                    goto dump_format;

                }

                memcpy(ost->st->codec->subtitle_header, dec->subtitle_header, dec->subtitle_header_size);

                ost->st->codec->subtitle_header_size = dec->subtitle_header_size;

            }

            if (avcodec_open2(ost->st->codec, codec, &ost->opts) < 0) {

                snprintf(error, sizeof(error), "Error while opening encoder for output stream #%d.%d - maybe incorrect parameters such as bit_rate, rate, width or height",

                        ost->file_index, ost->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            assert_codec_experimental(ost->st->codec, 1);

            assert_avoptions(ost->opts);

            if (ost->st->codec->bit_rate && ost->st->codec->bit_rate < 1000)

                av_log(NULL, AV_LOG_WARNING, "The bitrate parameter is set too low."

                                             "It takes bits/s as argument, not kbits/s\n");

            extra_size += ost->st->codec->extradata_size;

        }

    }



    /* init input streams */

    for (i = 0; i < nb_input_streams; i++)

        if ((ret = init_input_stream(i, output_streams, nb_output_streams, error, sizeof(error))) < 0)

            goto dump_format;



    /* open files and write file headers */

    for (i = 0; i < nb_output_files; i++) {

        os = output_files[i].ctx;

        if (avformat_write_header(os, &output_files[i].opts) < 0) {

            snprintf(error, sizeof(error), "Could not write header for output file #%d (incorrect codec parameters ?)", i);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        assert_avoptions(output_files[i].opts);

        if (strcmp(os->oformat->name, "rtp")) {

            want_sdp = 0;

        }

    }



 dump_format:

    /* dump the file output parameters - cannot be done before in case

       of stream copy */

    for(i=0;i<nb_output_files;i++) {

        av_dump_format(output_files[i].ctx, i, output_files[i].ctx->filename, 1);

    }



    /* dump the stream mapping */

    if (verbose >= 0) {

        fprintf(stderr, "Stream mapping:\n");

        for (i = 0; i < nb_output_streams;i ++) {

            ost = &output_streams[i];

            fprintf(stderr, "  Stream #%d.%d -> #%d.%d",

                    input_streams[ost->source_index].file_index,

                    input_streams[ost->source_index].st->index,

                    ost->file_index,

                    ost->index);

            if (ost->sync_ist != &input_streams[ost->source_index])

                fprintf(stderr, " [sync #%d.%d]",

                        ost->sync_ist->file_index,

                        ost->sync_ist->st->index);

            if (ost->st->stream_copy)

                fprintf(stderr, " (copy)");

            else

                fprintf(stderr, " (%s -> %s)", input_streams[ost->source_index].dec ?

                        input_streams[ost->source_index].dec->name : "?",

                        ost->enc ? ost->enc->name : "?");

            fprintf(stderr, "\n");

        }

    }



    if (ret) {

        fprintf(stderr, "%s\n", error);

        return ret;

    }



    if (want_sdp) {

        print_sdp(output_files, nb_output_files);

    }



    return 0;

}
