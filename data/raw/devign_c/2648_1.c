static int transcode(OutputFile *output_files,

                     int nb_output_files,

                     InputFile *input_files,

                     int nb_input_files)

{

    int ret, i;

    AVFormatContext *is, *os;

    OutputStream *ost;

    InputStream *ist;

    uint8_t *no_packet;

    int no_packet_count=0;

    int64_t timer_start;

    int key;



    if (!(no_packet = av_mallocz(nb_input_files)))

        exit_program(1);



    ret = transcode_init(output_files, nb_output_files, input_files, nb_input_files);

    if (ret < 0)

        goto fail;



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
