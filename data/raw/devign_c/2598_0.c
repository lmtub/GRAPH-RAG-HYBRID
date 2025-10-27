static int process_input_packet(InputStream *ist, const AVPacket *pkt, int no_eof)

{

    int ret = 0, i;

    int repeating = 0;

    int eof_reached = 0;



    AVPacket avpkt;

    if (!ist->saw_first_ts) {

        ist->dts = ist->st->avg_frame_rate.num ? - ist->dec_ctx->has_b_frames * AV_TIME_BASE / av_q2d(ist->st->avg_frame_rate) : 0;

        ist->pts = 0;

        if (pkt && pkt->pts != AV_NOPTS_VALUE && !ist->decoding_needed) {

            ist->dts += av_rescale_q(pkt->pts, ist->st->time_base, AV_TIME_BASE_Q);

            ist->pts = ist->dts; //unused but better to set it to a value thats not totally wrong

        }

        ist->saw_first_ts = 1;

    }



    if (ist->next_dts == AV_NOPTS_VALUE)

        ist->next_dts = ist->dts;

    if (ist->next_pts == AV_NOPTS_VALUE)

        ist->next_pts = ist->pts;



    if (!pkt) {

        /* EOF handling */

        av_init_packet(&avpkt);

        avpkt.data = NULL;

        avpkt.size = 0;

    } else {

        avpkt = *pkt;

    }



    if (pkt && pkt->dts != AV_NOPTS_VALUE) {

        ist->next_dts = ist->dts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);

        if (ist->dec_ctx->codec_type != AVMEDIA_TYPE_VIDEO || !ist->decoding_needed)

            ist->next_pts = ist->pts = ist->dts;

    }



    // while we have more to decode or while the decoder did output something on EOF

    while (ist->decoding_needed) {

        int duration = 0;

        int got_output = 0;



        ist->pts = ist->next_pts;

        ist->dts = ist->next_dts;



        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ret = decode_audio    (ist, repeating ? NULL : &avpkt, &got_output);

            break;

        case AVMEDIA_TYPE_VIDEO:

            ret = decode_video    (ist, repeating ? NULL : &avpkt, &got_output, !pkt);

            if (!repeating || !pkt || got_output) {

                if (pkt && pkt->duration) {

                    duration = av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

                } else if(ist->dec_ctx->framerate.num != 0 && ist->dec_ctx->framerate.den != 0) {

                    int ticks= av_stream_get_parser(ist->st) ? av_stream_get_parser(ist->st)->repeat_pict+1 : ist->dec_ctx->ticks_per_frame;

                    duration = ((int64_t)AV_TIME_BASE *

                                    ist->dec_ctx->framerate.den * ticks) /

                                    ist->dec_ctx->framerate.num / ist->dec_ctx->ticks_per_frame;

                }



                if(ist->dts != AV_NOPTS_VALUE && duration) {

                    ist->next_dts += duration;

                }else

                    ist->next_dts = AV_NOPTS_VALUE;

            }



            if (got_output)

                ist->next_pts += duration; //FIXME the duration is not correct in some cases

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            if (repeating)

                break;

            ret = transcode_subtitles(ist, &avpkt, &got_output);

            if (!pkt && ret >= 0)

                ret = AVERROR_EOF;

            break;

        default:

            return -1;

        }



        if (ret == AVERROR_EOF) {

            eof_reached = 1;

            break;

        }



        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error while decoding stream #%d:%d: %s\n",

                   ist->file_index, ist->st->index, av_err2str(ret));

            if (exit_on_error)

                exit_program(1);

            // Decoding might not terminate if we're draining the decoder, and

            // the decoder keeps returning an error.

            // This should probably be considered a libavcodec issue.

            // Sample: fate-vsynth1-dnxhd-720p-hr-lb

            if (!pkt)

                eof_reached = 1;

            break;

        }



        if (got_output)

            ist->got_output = 1;



        if (!got_output)

            break;



        // During draining, we might get multiple output frames in this loop.

        // ffmpeg.c does not drain the filter chain on configuration changes,

        // which means if we send multiple frames at once to the filters, and

        // one of those frames changes configuration, the buffered frames will

        // be lost. This can upset certain FATE tests.

        // Decode only 1 frame per call on EOF to appease these FATE tests.

        // The ideal solution would be to rewrite decoding to use the new

        // decoding API in a better way.

        if (!pkt)

            break;



        repeating = 1;

    }



    /* after flushing, send an EOF on all the filter inputs attached to the stream */

    /* except when looping we need to flush but not to send an EOF */

    if (!pkt && ist->decoding_needed && eof_reached && !no_eof) {

        int ret = send_filter_eof(ist);

        if (ret < 0) {

            av_log(NULL, AV_LOG_FATAL, "Error marking filters as finished\n");

            exit_program(1);

        }

    }



    /* handle stream copy */

    if (!ist->decoding_needed) {

        ist->dts = ist->next_dts;

        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ist->next_dts += ((int64_t)AV_TIME_BASE * ist->dec_ctx->frame_size) /

                             ist->dec_ctx->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (ist->framerate.num) {

                // TODO: Remove work-around for c99-to-c89 issue 7

                AVRational time_base_q = AV_TIME_BASE_Q;

                int64_t next_dts = av_rescale_q(ist->next_dts, time_base_q, av_inv_q(ist->framerate));

                ist->next_dts = av_rescale_q(next_dts + 1, av_inv_q(ist->framerate), time_base_q);

            } else if (pkt->duration) {

                ist->next_dts += av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

            } else if(ist->dec_ctx->framerate.num != 0) {

                int ticks= av_stream_get_parser(ist->st) ? av_stream_get_parser(ist->st)->repeat_pict + 1 : ist->dec_ctx->ticks_per_frame;

                ist->next_dts += ((int64_t)AV_TIME_BASE *

                                  ist->dec_ctx->framerate.den * ticks) /

                                  ist->dec_ctx->framerate.num / ist->dec_ctx->ticks_per_frame;

            }

            break;

        }

        ist->pts = ist->dts;

        ist->next_pts = ist->next_dts;

    }

    for (i = 0; pkt && i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        if (!check_output_constraints(ist, ost) || ost->encoding_needed)

            continue;



        do_streamcopy(ist, ost, pkt);

    }



    return !eof_reached;

}
