static int process_input(void)

{

    InputFile *ifile;

    AVFormatContext *is;

    InputStream *ist;

    AVPacket pkt;

    int ret, i, j;

    int file_index;



    /* select the stream that we must read now */

    file_index = select_input_file();

    /* if none, if is finished */

    if (file_index == -2) {

        poll_filters() ;

        return AVERROR(EAGAIN);

    }

    if (file_index < 0) {

        if (got_eagain()) {

            reset_eagain();

            av_usleep(10000);

            return AVERROR(EAGAIN);

        }

        av_log(NULL, AV_LOG_VERBOSE, "No more inputs to read from, finishing.\n");

        return AVERROR_EOF;

    }

    ifile = input_files[file_index];



    is  = ifile->ctx;

    ret = get_input_packet(ifile, &pkt);



    if (ret == AVERROR(EAGAIN)) {

        ifile->eagain = 1;

        return ret;

    }

    if (ret < 0) {

        if (ret != AVERROR_EOF) {

            print_error(is->filename, ret);

            if (exit_on_error)

                exit_program(1);

        }

        ifile->eof_reached = 1;



        for (i = 0; i < ifile->nb_streams; i++) {

            ist = input_streams[ifile->ist_index + i];

            if (ist->decoding_needed)

                output_packet(ist, NULL);

            poll_filters();

        }



        if (opt_shortest)

            return AVERROR_EOF;

        else

            return AVERROR(EAGAIN);

    }



    reset_eagain();



    if (do_pkt_dump) {

        av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &pkt, do_hex_dump,

                         is->streams[pkt.stream_index]);

    }

    /* the following test is needed in case new streams appear

       dynamically in stream : we ignore them */

    if (pkt.stream_index >= ifile->nb_streams) {

        report_new_stream(file_index, &pkt);

        goto discard_packet;

    }



    ist = input_streams[ifile->ist_index + pkt.stream_index];

    if (ist->discard)

        goto discard_packet;



    if(!ist->wrap_correction_done && input_files[file_index]->ctx->start_time != AV_NOPTS_VALUE && ist->st->pts_wrap_bits < 64){

        uint64_t stime = av_rescale_q(input_files[file_index]->ctx->start_time, AV_TIME_BASE_Q, ist->st->time_base);

        uint64_t stime2= stime + (1LL<<ist->st->pts_wrap_bits);

        ist->wrap_correction_done = 1;

        if(pkt.dts != AV_NOPTS_VALUE && pkt.dts > stime && pkt.dts - stime > stime2 - pkt.dts) {

            pkt.dts -= 1LL<<ist->st->pts_wrap_bits;

            ist->wrap_correction_done = 0;

        }

        if(pkt.pts != AV_NOPTS_VALUE && pkt.pts > stime && pkt.pts - stime > stime2 - pkt.pts) {

            pkt.pts -= 1LL<<ist->st->pts_wrap_bits;

            ist->wrap_correction_done = 0;

        }

    }



    if (pkt.dts != AV_NOPTS_VALUE)

        pkt.dts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);

    if (pkt.pts != AV_NOPTS_VALUE)

        pkt.pts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, ist->st->time_base);



    if (pkt.pts != AV_NOPTS_VALUE)

        pkt.pts *= ist->ts_scale;

    if (pkt.dts != AV_NOPTS_VALUE)

        pkt.dts *= ist->ts_scale;



    if (debug_ts) {

        av_log(NULL, AV_LOG_INFO, "demuxer -> ist_index:%d type:%s "

                "next_dts:%s next_dts_time:%s next_pts:%s next_pts_time:%s  pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s off:%"PRId64"\n",

                ifile->ist_index + pkt.stream_index, av_get_media_type_string(ist->st->codec->codec_type),

                av_ts2str(ist->next_dts), av_ts2timestr(ist->next_dts, &AV_TIME_BASE_Q),

                av_ts2str(ist->next_pts), av_ts2timestr(ist->next_pts, &AV_TIME_BASE_Q),

                av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ist->st->time_base),

                av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ist->st->time_base),

                input_files[ist->file_index]->ts_offset);

    }



    if (pkt.dts != AV_NOPTS_VALUE && ist->next_dts != AV_NOPTS_VALUE &&

        !copy_ts) {

        int64_t pkt_dts = av_rescale_q(pkt.dts, ist->st->time_base, AV_TIME_BASE_Q);

        int64_t delta   = pkt_dts - ist->next_dts;

        if (is->iformat->flags & AVFMT_TS_DISCONT) {

        if(delta < -1LL*dts_delta_threshold*AV_TIME_BASE ||

            (delta > 1LL*dts_delta_threshold*AV_TIME_BASE &&

                ist->st->codec->codec_type != AVMEDIA_TYPE_SUBTITLE) ||

            pkt_dts+1<ist->pts){

            ifile->ts_offset -= delta;

            av_log(NULL, AV_LOG_DEBUG,

                   "timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",

                   delta, ifile->ts_offset);

            pkt.dts -= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

            if (pkt.pts != AV_NOPTS_VALUE)

                pkt.pts -= av_rescale_q(delta, AV_TIME_BASE_Q, ist->st->time_base);

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



    sub2video_heartbeat(ist, pkt.pts);



    if ((ret = output_packet(ist, &pkt)) < 0 ||

        ((ret = poll_filters()) < 0 && ret != AVERROR_EOF)) {

        char buf[128];

        av_strerror(ret, buf, sizeof(buf));

        av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d: %s\n",

                ist->file_index, ist->st->index, buf);

        if (exit_on_error)

            exit_program(1);

        av_free_packet(&pkt);

        return AVERROR(EAGAIN);

    }



discard_packet:

    av_free_packet(&pkt);



    return 0;

}
