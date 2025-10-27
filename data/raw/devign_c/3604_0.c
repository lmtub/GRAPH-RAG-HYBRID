static int transcode(OutputFile *output_files,

                     int nb_output_files,

                     InputFile *input_files,

                     int nb_input_files)

{

    int ret = 0, i;

    AVFormatContext *is, *os;

    AVCodecContext *codec, *icodec;

    OutputStream *ost;

    InputStream *ist;

    char error[1024];

    int key;

    int want_sdp = 1;

    uint8_t *no_packet;

    int no_packet_count=0;

    int64_t timer_start;



    if (!(no_packet = av_mallocz(nb_input_files)))

        exit_program(1);



    if (rate_emu)

        for (i = 0; i < nb_input_streams; i++)

            input_streams[i].start = av_gettime();



    /* output stream init */

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i].ctx;

        if (!os->nb_streams && !(os->oformat->flags & AVFMT_NOSTREAMS)) {

            av_dump_format(os, i, os->filename, 1);

            fprintf(stderr, "Output file #%d does not contain any stream\n", i);

            ret = AVERROR(EINVAL);

            goto fail;

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

                ret = AVERROR(EINVAL);

                goto fail;

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

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            memcpy(codec->extradata, icodec->extradata, icodec->extradata_size);

            codec->extradata_size= icodec->extradata_size;



            codec->time_base = ist->st->time_base;

            if(!strcmp(os->oformat->name, "avi")) {

                if(!copy_tb && av_q2d(icodec->time_base)*icodec->ticks_per_frame > 2*av_q2d(ist->st->time_base) && av_q2d(ist->st->time_base) < 1.0/500){

                    codec->time_base = icodec->time_base;

                    codec->time_base.num *= icodec->ticks_per_frame;

                    codec->time_base.den *= 2;

                }

            } else if(!(os->oformat->flags & AVFMT_VARIABLE_FPS)) {

                if(!copy_tb && av_q2d(icodec->time_base)*icodec->ticks_per_frame > av_q2d(ist->st->time_base) && av_q2d(ist->st->time_base) < 1.0/500){

                    codec->time_base = icodec->time_base;

                    codec->time_base.num *= icodec->ticks_per_frame;

                }

            }

            av_reduce(&codec->time_base.num, &codec->time_base.den,

                        codec->time_base.num, codec->time_base.den, INT_MAX);



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

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                ost->reformat_pair = MAKE_SFMT_PAIR(AV_SAMPLE_FMT_NONE,AV_SAMPLE_FMT_NONE);

                if (!codec->sample_rate) {

                    codec->sample_rate = icodec->sample_rate;

                }

                choose_sample_rate(ost->st, ost->enc);

                codec->time_base = (AVRational){1, codec->sample_rate};

                if (codec->sample_fmt == AV_SAMPLE_FMT_NONE)

                    codec->sample_fmt = icodec->sample_fmt;

                choose_sample_fmt(ost->st, ost->enc);

                if (!codec->channels) {

                    codec->channels = icodec->channels;

                    codec->channel_layout = icodec->channel_layout;

                }

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

                    codec->bits_per_raw_sample= frame_bits_per_raw_sample;

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

                if(   av_q2d(codec->time_base) < 0.001 && video_sync_method

                   && (video_sync_method==1 || (video_sync_method<0 && !(os->oformat->flags & AVFMT_VARIABLE_FPS)))){

                    av_log(os, AV_LOG_WARNING, "Frame rate very high for a muxer not effciciently supporting it.\n"

                                               "Please consider specifiying a lower framerate, a different muxer or -vsync 2\n");

                }



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

            if (ost->encoding_needed && codec->codec_id != CODEC_ID_H264 &&

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

            /* maximum video buffer size is 6-bytes per pixel, plus DPX header size */

            int size= codec->width * codec->height;

            bit_buffer_size= FFMAX(bit_buffer_size, 6*size + 1664);

        }

    }



    if (!bit_buffer)

        bit_buffer = av_malloc(bit_buffer_size);

    if (!bit_buffer) {

        fprintf(stderr, "Cannot allocate %d bytes output buffer\n",

                bit_buffer_size);

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    /* open each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = &output_streams[i];

        if (ost->encoding_needed) {

            AVCodec *codec = ost->enc;

            AVCodecContext *dec = input_streams[ost->source_index].st->codec;

            if (!codec) {

                snprintf(error, sizeof(error), "Encoder (codec %s) not found for output stream #%d.%d",

                         avcodec_get_name(ost->st->codec->codec_id), ost->file_index, ost->index);

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

        if ((ret = init_input_stream(i, output_streams, nb_output_streams, error, sizeof(error)) < 0))

            goto dump_format;



    /* open files and write file headers */

    for (i = 0; i < nb_output_files; i++) {

        os = output_files[i].ctx;

        if (avformat_write_header(os, &output_files[i].opts) < 0) {

            snprintf(error, sizeof(error), "Could not write header for output file #%d (incorrect codec parameters ?)", i);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

//        assert_avoptions(output_files[i].opts);

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

            if(ost->encoding_needed)

                fprintf(stderr, ": %s -> %s",

                    input_streams[ost->source_index].dec ?

                        input_streams[ost->source_index].dec->name : "?",

                    ost->enc ? ost->enc->name : "?");

            else

                fprintf(stderr, ": copy");

            fprintf(stderr, "\n");

        }

    }



    if (ret) {

        fprintf(stderr, "%s\n", error);

        goto fail;

    }



    if (want_sdp) {

        print_sdp(output_files, nb_output_files);

    }



    if (!using_stdin) {

        if(verbose >= 0)

            fprintf(stderr, "Press [q] to stop, [?] for help\n");

        avio_set_interrupt_cb(decode_interrupt_cb);

    }

    term_init();



    timer_start = av_gettime();



    for(; received_sigterm == 0;) {

        int file_index, ist_index;

        AVPacket pkt;

        int64_t ipts_min;

        double opts_min;



    redo:

        ipts_min= INT64_MAX;

        opts_min= 1e100;

        /* if 'q' pressed, exits */

        if (!using_stdin) {

            if (q_pressed)

                break;

            /* read_key() returns 0 on EOF */

            key = read_key();

            if (key == 'q')

                break;

            if (key == '+') verbose++;

            if (key == '-') verbose--;

            if (key == 's') qp_hist     ^= 1;

            if (key == 'h'){

                if (do_hex_dump){

                    do_hex_dump = do_pkt_dump = 0;

                } else if(do_pkt_dump){

                    do_hex_dump = 1;

                } else

                    do_pkt_dump = 1;

                av_log_set_level(AV_LOG_DEBUG);

            }

            if (key == 'c' || key == 'C'){

                char ret[4096], target[64], cmd[256], arg[256]={0};

                double ts;

                fprintf(stderr, "\nEnter command: <target> <time> <command>[ <argument>]\n");

                if(scanf("%4095[^\n\r]%*c", ret) == 1 && sscanf(ret, "%63[^ ] %lf %255[^ ] %255[^\n]", target, &ts, cmd, arg) >= 3){

                    for(i=0;i<nb_output_streams;i++) {

                        int r;

                        ost = &output_streams[i];

                        if(ost->graph){

                            if(ts<0){

                                r= avfilter_graph_send_command(ost->graph, target, cmd, arg, ret, sizeof(ret), key == 'c' ? AVFILTER_CMD_FLAG_ONE : 0);

                                fprintf(stderr, "Command reply for %d: %d, %s\n", i, r, ret);

                            }else{

                                r= avfilter_graph_queue_command(ost->graph, target, cmd, arg, 0, ts);

                            }

                        }

                    }

                }else{

                    fprintf(stderr, "Parse error\n");

                }

            }

            if (key == 'd' || key == 'D'){

                int debug=0;

                if(key == 'D') {

                    debug = input_streams[0].st->codec->debug<<1;

                    if(!debug) debug = 1;

                    while(debug & (FF_DEBUG_DCT_COEFF|FF_DEBUG_VIS_QP|FF_DEBUG_VIS_MB_TYPE)) //unsupported, would just crash

                        debug += debug;

                }else

                    scanf("%d", &debug);

                for(i=0;i<nb_input_streams;i++) {

                    input_streams[i].st->codec->debug = debug;

                }

                for(i=0;i<nb_output_streams;i++) {

                    ost = &output_streams[i];

                    ost->st->codec->debug = debug;

                }

                if(debug) av_log_set_level(AV_LOG_DEBUG);

                fprintf(stderr,"debug=%d\n", debug);

            }

            if (key == '?'){

                fprintf(stderr, "key    function\n"

                                "?      show this help\n"

                                "+      increase verbosity\n"

                                "-      decrease verbosity\n"

                                "c      Send command to filtergraph\n"

                                "D      cycle through available debug modes\n"

                                "h      dump packets/hex press to cycle through the 3 states\n"

                                "q      quit\n"

                                "s      Show QP histogram\n"

                );

            }

        }



        /* select the stream that we must read now by looking at the

           smallest output pts */

        file_index = -1;

        for (i = 0; i < nb_output_streams; i++) {

            OutputFile *of;

            int64_t ipts;

            double  opts;

            ost = &output_streams[i];

            of = &output_files[ost->file_index];

            os = output_files[ost->file_index].ctx;

            ist = &input_streams[ost->source_index];

            if (ost->is_past_recording_time || no_packet[ist->file_index] ||

                (os->pb && avio_tell(os->pb) >= of->limit_filesize))

                continue;

            opts = ost->st->pts.val * av_q2d(ost->st->time_base);

            ipts = ist->pts;

            if (!input_files[ist->file_index].eof_reached){

                if(ipts < ipts_min) {

                    ipts_min = ipts;

                    if(input_sync ) file_index = ist->file_index;

                }

                if(opts < opts_min) {

                    opts_min = opts;

                    if(!input_sync) file_index = ist->file_index;

                }

            }

            if(ost->frame_number >= max_frames[ost->st->codec->codec_type]){

                file_index= -1;

                break;

            }

        }

        /* if none, if is finished */

        if (file_index < 0) {

            if(no_packet_count){

                no_packet_count=0;

                memset(no_packet, 0, nb_input_files);

                usleep(10000);

                continue;

            }

            break;

        }



        /* read a frame from it and output it in the fifo */

        is = input_files[file_index].ctx;

        ret= av_read_frame(is, &pkt);

        if(ret == AVERROR(EAGAIN)){

            no_packet[file_index]=1;

            no_packet_count++;

            continue;

        }

        if (ret < 0) {

            input_files[file_index].eof_reached = 1;

            if (opt_shortest)

                break;

            else

                continue;

        }



        no_packet_count=0;

        memset(no_packet, 0, nb_input_files);



        if (do_pkt_dump) {

            av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                             is->streams[pkt.stream_index]);

        }

        /* the following test is needed in case new streams appear

           dynamically in stream : we ignore them */

        if (pkt.stream_index >= input_files[file_index].nb_streams)

            goto discard_packet;

        ist_index = input_files[file_index].ist_index + pkt.stream_index;

        ist = &input_streams[ist_index];

        if (ist->discard)

            goto discard_packet;



        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts += av_rescale_q(input_files[ist->file_index].ts_offset, AV_TIME_BASE_Q, ist->st->time_base);

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts += av_rescale_q(input_files[ist->file_index].ts_offset, AV_TIME_BASE_Q, ist->st->time_base);



        if (ist->ts_scale) {

            if(pkt.pts != AV_NOPTS_VALUE)

                pkt.pts *= ist->ts_scale;

            if(pkt.dts != AV_NOPTS_VALUE)

                pkt.dts *= ist->ts_scale;

        }



//        fprintf(stderr, "next:%"PRId64" dts:%"PRId64" off:%"PRId64" %d\n", ist->next_pts, pkt.dts, input_files[ist->file_index].ts_offset, ist->st->codec->codec_type);

        if (pkt.dts != AV_NOPTS_VALUE && ist->next_pts != AV_NOPTS_VALUE

            && (is->iformat->flags & AVFMT_TS_DISCONT)) {

            int64_t pkt_dts= av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

            int64_t delta= pkt_dts - ist->next_pts;

            if((delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||

                (delta > 1LL*dts_delta_threshold*AV_TIME_BASE &&

                 ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

                pkt_dts+1<ist->pts)&& !copy_ts){

                input_files[ist->file_index].ts_offset -= delta;

                if (verbose > 2)

                    fprintf(stderr, "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",

                            delta, input_files[ist->file_index].ts_offset);

                pkt.dts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

                if(pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            }

        }



        //fprintf(stderr,"read #%d.%d size=%d\n", ist->file_index, ist->st->index, pkt.size);

        if (output_packet(ist, ist_index, output_streams, nb_output_streams, &pkt) < 0) {



            if (verbose >= 0)

                fprintf(stderr, "Error while decoding stream #%d.%d\n",

                        ist->file_index, ist->st->index);

            if (exit_on_error)

                exit_program(1);

            av_free_packet(&pkt);

            goto redo;

        }



    discard_packet:

        av_free_packet(&pkt);



        /* dump report by using the output first video and audio streams */

        print_report(output_files, output_streams, nb_output_streams, 0, timer_start);

    }



    /* at the end of stream, we must flush the decoder buffers */

    for (i = 0; i < nb_input_streams; i++) {

        ist = &input_streams[i];

        if (ist->decoding_needed) {

            output_packet(ist, i, output_streams, nb_output_streams, NULL);

        }

    }

    flush_encoders(output_streams, nb_output_streams);



    term_exit();



    /* write the trailer if needed and close file */

    for(i=0;i<nb_output_files;i++) {

        os = output_files[i].ctx;

        av_write_trailer(os);

    }



    /* dump report by using the first video and audio streams */

    print_report(output_files, output_streams, nb_output_streams, 1, timer_start);



    /* close each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = &output_streams[i];

        if (ost->encoding_needed) {

            av_freep(&ost->st->codec->stats_in);

            avcodec_close(ost->st->codec);

        }

#if CONFIG_AVFILTER

        avfilter_graph_free(&ost->graph);

#endif

    }



    /* close each decoder */

    for (i = 0; i < nb_input_streams; i++) {

        ist = &input_streams[i];

        if (ist->decoding_needed) {

            avcodec_close(ist->st->codec);

        }

    }



    /* finished ! */

    ret = 0;



 fail:

    av_freep(&bit_buffer);

    av_freep(&no_packet);



    if (output_streams) {

        for (i = 0; i < nb_output_streams; i++) {

            ost = &output_streams[i];

            if (ost) {

                if (ost->st->stream_copy)

                    av_freep(&ost->st->codec->extradata);

                if (ost->logfile) {

                    fclose(ost->logfile);

                    ost->logfile = NULL;

                }

                av_fifo_free(ost->fifo); /* works even if fifo is not

                                             initialized but set to zero */

                av_freep(&ost->st->codec->subtitle_header);

                av_free(ost->resample_frame.data[0]);

                av_free(ost->forced_kf_pts);

                if (ost->video_resample)

                    sws_freeContext(ost->img_resample_ctx);

                if (ost->resample)

                    audio_resample_close(ost->resample);

                if (ost->reformat_ctx)

                    av_audio_convert_free(ost->reformat_ctx);

                av_dict_free(&ost->opts);

            }

        }

    }

    return ret;

}
