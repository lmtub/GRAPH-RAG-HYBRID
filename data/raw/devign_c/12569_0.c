static int transcode(void)

{

    int ret, i;

    AVFormatContext *is, *os;

    OutputStream *ost;

    InputStream *ist;

    uint8_t *no_packet;

    int no_packet_count = 0;

    int64_t timer_start;

    int key;



    if (!(no_packet = av_mallocz(nb_input_files)))

        exit_program(1);



    ret = transcode_init();

    if (ret < 0)

        goto fail;



    if (!using_stdin) {

        av_log(NULL, AV_LOG_INFO, "Press [q] to stop, [?] for help\n");

    }



    timer_start = av_gettime();



    for (; received_sigterm == 0;) {

        int file_index, ist_index;

        AVPacket pkt;

        int64_t cur_time= av_gettime();



        /* if 'q' pressed, exits */

        if (!using_stdin) {

            static int64_t last_time;

            if (received_nb_signals)

                break;

            /* read_key() returns 0 on EOF */

            if(cur_time - last_time >= 100000 && !run_as_daemon){

                key =  read_key();

                last_time = cur_time;

            }else

                key = -1;

            if (key == 'q')

                break;

            if (key == '+') av_log_set_level(av_log_get_level()+10);

            if (key == '-') av_log_set_level(av_log_get_level()-10);

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

                char buf[4096], target[64], command[256], arg[256] = {0};

                double time;

                int k, n = 0;

                fprintf(stderr, "\nEnter command: <target> <time> <command>[ <argument>]\n");

                i = 0;

                while ((k = read_key()) != '\n' && k != '\r' && i < sizeof(buf)-1)

                    if (k > 0)

                        buf[i++] = k;

                buf[i] = 0;

                if (k > 0 &&

                    (n = sscanf(buf, "%63[^ ] %lf %255[^ ] %255[^\n]", target, &time, command, arg)) >= 3) {

                    av_log(NULL, AV_LOG_DEBUG, "Processing command target:%s time:%f command:%s arg:%s",

                           target, time, command, arg);

                    for (i = 0; i < nb_filtergraphs; i++) {

                        FilterGraph *fg = filtergraphs[i];

                        if (fg->graph) {

                            if (time < 0) {

                                ret = avfilter_graph_send_command(fg->graph, target, command, arg, buf, sizeof(buf),

                                                                  key == 'c' ? AVFILTER_CMD_FLAG_ONE : 0);

                                fprintf(stderr, "Command reply for stream %d: ret:%d res:%s\n", i, ret, buf);

                            } else {

                                ret = avfilter_graph_queue_command(fg->graph, target, command, arg, 0, time);

                            }

                        }

                    }

                } else {

                    av_log(NULL, AV_LOG_ERROR,

                           "Parse error, at least 3 arguments were expected, "

                           "only %d given in string '%s'\n", n, buf);

                }

            }

            if (key == 'd' || key == 'D'){

                int debug=0;

                if(key == 'D') {

                    debug = input_streams[0]->st->codec->debug<<1;

                    if(!debug) debug = 1;

                    while(debug & (FF_DEBUG_DCT_COEFF|FF_DEBUG_VIS_QP|FF_DEBUG_VIS_MB_TYPE)) //unsupported, would just crash

                        debug += debug;

                }else

                    if(scanf("%d", &debug)!=1)

                        fprintf(stderr,"error parsing debug value\n");

                for(i=0;i<nb_input_streams;i++) {

                    input_streams[i]->st->codec->debug = debug;

                }

                for(i=0;i<nb_output_streams;i++) {

                    ost = output_streams[i];

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



        /* check if there's any stream where output is still needed */

        if (!need_output()) {

            av_log(NULL, AV_LOG_VERBOSE, "No more output streams to write to, finishing.\n");

            break;

        }



        /* select the stream that we must read now */

        file_index = select_input_file(no_packet);

        /* if none, if is finished */

        if (file_index < 0) {

            if (no_packet_count) {

                no_packet_count = 0;

                memset(no_packet, 0, nb_input_files);

                usleep(10000);

                continue;

            }

            break;

        }



        /* read a frame from it and output it in the fifo */

        is  = input_files[file_index]->ctx;

        ret = av_read_frame(is, &pkt);

        if (ret == AVERROR(EAGAIN)) {

            no_packet[file_index] = 1;

            no_packet_count++;

            continue;

        }

        if (ret < 0) {

            input_files[file_index]->eof_reached = 1;



            for (i = 0; i < input_files[file_index]->nb_streams; i++) {

                ist = input_streams[input_files[file_index]->ist_index + i];

                if (ist->decoding_needed)

                    output_packet(ist, NULL);

            }



            if (opt_shortest)

                break;

            else

                continue;

        }



        no_packet_count = 0;

        memset(no_packet, 0, nb_input_files);



        if (do_pkt_dump) {

            av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                             is->streams[pkt.stream_index]);

        }

        /* the following test is needed in case new streams appear

           dynamically in stream : we ignore them */

        if (pkt.stream_index >= input_files[file_index]->nb_streams)

            goto discard_packet;

        ist_index = input_files[file_index]->ist_index + pkt.stream_index;

        ist = input_streams[ist_index];

        if (ist->discard)

            goto discard_packet;



        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts += av_rescale_q(input_files[ist->file_index]->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts += av_rescale_q(input_files[ist->file_index]->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);



        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts *= ist->ts_scale;

        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts *= ist->ts_scale;



        if (debug_ts) {

            av_log(NULL, AV_LOG_INFO, "demuxer -> ist_index:%d type:%s "

                    "next_dts:%s next_dts_time:%s next_pts:%s next_pts_time:%s  pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s off:%"PRId64"\n",

                    ist_index, av_get_media_type_string(ist->st->codec->codec_type),

                    av_ts2str(ist->next_dts), av_ts2timestr(ist->next_dts, &ist->st->time_base),

                    av_ts2str(ist->next_pts), av_ts2timestr(ist->next_pts, &ist->st->time_base),

                    av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ist->st->time_base),

                    av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ist->st->time_base),

                    input_files[ist->file_index]->ts_offset);

        }



        if (pkt.dts != AV_NOPTS_VALUE && ist->next_dts != AV_NOPTS_VALUE && !copy_ts) {

            int64_t pkt_dts = av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

            int64_t delta   = pkt_dts - ist->next_dts;

            if (is->iformat->flags & AVFMT_TS_DISCONT) {

            if(delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||

                (delta > 1LL*dts_delta_threshold*AV_TIME_BASE &&

                 ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

                pkt_dts+1<ist->pts){

                input_files[ist->file_index]->ts_offset -= delta;

                av_log(NULL, AV_LOG_DEBUG,

                       "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",

                       delta, input_files[ist->file_index]->ts_offset);

                pkt.dts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

                if (pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            }

            } else {

                if ( delta < -1LL*dts_error_threshold*AV_TIME_BASE ||

                    (delta > 1LL*dts_error_threshold*AV_TIME_BASE && ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

                     pkt_dts+1<ist->pts){

                    av_log(NULL, AV_LOG_WARNING, "DTS %"PRId64", next:%"PRId64" st:%d invalid dropping\n", pkt.dts, ist->next_dts, pkt.stream_index);

                    pkt.dts = AV_NOPTS_VALUE;

                }

                if (pkt.pts != AV_NOPTS_VALUE){

                    int64_t pkt_pts = av_rescale_q(pkt.pts, ist->st->time_base, AV_TIME_BASE_Q);

                    delta   = pkt_pts - ist->next_dts;

                    if ( delta < -1LL*dts_error_threshold*AV_TIME_BASE ||

                        (delta > 1LL*dts_error_threshold*AV_TIME_BASE && ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

                        pkt_pts+1<ist->pts) {

                        av_log(NULL, AV_LOG_WARNING, "PTS %"PRId64", next:%"PRId64" invalid dropping st:%d\n", pkt.pts, ist->next_dts, pkt.stream_index);

                        pkt.pts = AV_NOPTS_VALUE;

                    }

                }

            }

        }



        // fprintf(stderr,"read #%d.%d size=%d\n", ist->file_index, ist->st->index, pkt.size);

        if ((ret = output_packet(ist, &pkt)) < 0 ||

            ((ret = poll_filters()) < 0 && ret != AVERROR_EOF)) {

            char buf[128];

            av_strerror(ret, buf, sizeof(buf));

            av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d: %s\n",

                   ist->file_index, ist->st->index, buf);

            if (exit_on_error)

                exit_program(1);

            av_free_packet(&pkt);

            continue;

        }



    discard_packet:

        av_free_packet(&pkt);



        /* dump report by using the output first video and audio streams */

        print_report(0, timer_start, cur_time);

    }



    /* at the end of stream, we must flush the decoder buffers */

    for (i = 0; i < nb_input_streams; i++) {

        ist = input_streams[i];

        if (!input_files[ist->file_index]->eof_reached && ist->decoding_needed) {

            output_packet(ist, NULL);

        }

    }

    poll_filters();

    flush_encoders();



    term_exit();



    /* write the trailer if needed and close file */

    for (i = 0; i < nb_output_files; i++) {

        os = output_files[i]->ctx;

        av_write_trailer(os);

    }



    /* dump report by using the first video and audio streams */

    print_report(1, timer_start, av_gettime());



    /* close each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = output_streams[i];

        if (ost->encoding_needed) {

            av_freep(&ost->st->codec->stats_in);

            avcodec_close(ost->st->codec);

        }

    }



    /* close each decoder */

    for (i = 0; i < nb_input_streams; i++) {

        ist = input_streams[i];

        if (ist->decoding_needed) {

            avcodec_close(ist->st->codec);

        }

    }



    /* finished ! */

    ret = 0;



 fail:

    av_freep(&no_packet);



    if (output_streams) {

        for (i = 0; i < nb_output_streams; i++) {

            ost = output_streams[i];

            if (ost) {

                if (ost->stream_copy)

                    av_freep(&ost->st->codec->extradata);

                if (ost->logfile) {

                    fclose(ost->logfile);

                    ost->logfile = NULL;

                }

                av_freep(&ost->st->codec->subtitle_header);

                av_free(ost->forced_kf_pts);

                av_dict_free(&ost->opts);

            }

        }

    }

    return ret;

}
