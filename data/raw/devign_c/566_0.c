static int process_input_packet(InputStream *ist, const AVPacket *pkt)

{

    int i;

    int got_output;

    AVPacket avpkt;



    if (ist->next_dts == AV_NOPTS_VALUE)

        ist->next_dts = ist->last_dts;



    if (pkt == NULL) {

        /* EOF handling */

        av_init_packet(&avpkt);

        avpkt.data = NULL;

        avpkt.size = 0;

        goto handle_eof;

    } else {

        avpkt = *pkt;

    }



    if (pkt->dts != AV_NOPTS_VALUE)

        ist->next_dts = ist->last_dts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);



    // while we have more to decode or while the decoder did output something on EOF

    while (ist->decoding_needed && (avpkt.size > 0 || (!pkt && got_output))) {

        int ret = 0;

    handle_eof:



        ist->last_dts = ist->next_dts;



        if (avpkt.size && avpkt.size != pkt->size &&

            !(ist->dec->capabilities & CODEC_CAP_SUBFRAMES)) {

            av_log(NULL, ist->showed_multi_packet_warning ? AV_LOG_VERBOSE : AV_LOG_WARNING,

                   "Multiple frames in a packet from stream %d\n", pkt->stream_index);

            ist->showed_multi_packet_warning = 1;

        }



        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ret = decode_audio    (ist, &avpkt, &got_output);

            break;

        case AVMEDIA_TYPE_VIDEO:

            ret = decode_video    (ist, &avpkt, &got_output);

            if (avpkt.duration)

                ist->next_dts += av_rescale_q(avpkt.duration, ist->st->time_base, AV_TIME_BASE_Q);

            else if (ist->st->avg_frame_rate.num)

                ist->next_dts += av_rescale_q(1, av_inv_q(ist->st->avg_frame_rate),

                                              AV_TIME_BASE_Q);

            else if (ist->dec_ctx->time_base.num != 0) {

                int ticks      = ist->st->parser ? ist->st->parser->repeat_pict + 1 :

                                                   ist->dec_ctx->ticks_per_frame;

                ist->next_dts += av_rescale_q(ticks, ist->dec_ctx->time_base, AV_TIME_BASE_Q);

            }

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            ret = transcode_subtitles(ist, &avpkt, &got_output);

            break;

        default:

            return -1;

        }



        if (ret < 0)

            return ret;

        // touch data and size only if not EOF

        if (pkt) {

            avpkt.data += ret;

            avpkt.size -= ret;

        }

        if (!got_output) {

            continue;

        }

    }



    /* handle stream copy */

    if (!ist->decoding_needed) {

        ist->last_dts = ist->next_dts;

        switch (ist->dec_ctx->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ist->next_dts += ((int64_t)AV_TIME_BASE * ist->dec_ctx->frame_size) /

                             ist->dec_ctx->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (ist->dec_ctx->time_base.num != 0) {

                int ticks = ist->st->parser ? ist->st->parser->repeat_pict + 1 : ist->dec_ctx->ticks_per_frame;

                ist->next_dts += ((int64_t)AV_TIME_BASE *

                                  ist->dec_ctx->time_base.num * ticks) /

                                  ist->dec_ctx->time_base.den;

            }

            break;

        }

    }

    for (i = 0; pkt && i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        if (!check_output_constraints(ist, ost) || ost->encoding_needed)

            continue;



        do_streamcopy(ist, ost, pkt);

    }



    return 0;

}
