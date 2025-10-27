static int output_packet(InputStream *ist, const AVPacket *pkt)

{

    int ret = 0, i;

    int got_output;



    AVPacket avpkt;

    if (!ist->saw_first_ts) {

        ist->dts = ist->st->avg_frame_rate.num ? - ist->st->codec->has_b_frames * AV_TIME_BASE / av_q2d(ist->st->avg_frame_rate) : 0;

        ist->pts = 0;

        if (pkt != NULL && pkt->pts != AV_NOPTS_VALUE && !ist->decoding_needed) {

            ist->dts += av_rescale_q(pkt->pts, ist->st->time_base, AV_TIME_BASE_Q);

            ist->pts = ist->dts; //unused but better to set it to a value thats not totally wrong

        }

        ist->saw_first_ts = 1;

    }



    if (ist->next_dts == AV_NOPTS_VALUE)

        ist->next_dts = ist->dts;

    if (ist->next_pts == AV_NOPTS_VALUE)

        ist->next_pts = ist->pts;



    if (pkt == NULL) {

        /* EOF handling */

        av_init_packet(&avpkt);

        avpkt.data = NULL;

        avpkt.size = 0;

        goto handle_eof;

    } else {

        avpkt = *pkt;

    }



    if (pkt->dts != AV_NOPTS_VALUE) {

        ist->next_dts = ist->dts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);

        if (ist->st->codec->codec_type != AVMEDIA_TYPE_VIDEO || !ist->decoding_needed)

            ist->next_pts = ist->pts = ist->dts;

    }



    // while we have more to decode or while the decoder did output something on EOF

    while (ist->decoding_needed && (avpkt.size > 0 || (!pkt && got_output))) {

        int duration;

    handle_eof:



        ist->pts = ist->next_pts;

        ist->dts = ist->next_dts;



        if (avpkt.size && avpkt.size != pkt->size) {

            av_log(NULL, ist->showed_multi_packet_warning ? AV_LOG_VERBOSE : AV_LOG_WARNING,

                   "Multiple frames in a packet from stream %d\n", pkt->stream_index);

            ist->showed_multi_packet_warning = 1;

        }



        switch (ist->st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ret = decode_audio    (ist, &avpkt, &got_output);

            break;

        case AVMEDIA_TYPE_VIDEO:

            ret = decode_video    (ist, &avpkt, &got_output);

            if (avpkt.duration) {

                duration = av_rescale_q(avpkt.duration, ist->st->time_base, AV_TIME_BASE_Q);

            } else if(ist->st->codec->time_base.num != 0 && ist->st->codec->time_base.den != 0) {

                int ticks= ist->st->parser ? ist->st->parser->repeat_pict+1 : ist->st->codec->ticks_per_frame;

                duration = ((int64_t)AV_TIME_BASE *

                                ist->st->codec->time_base.num * ticks) /

                                ist->st->codec->time_base.den;

            } else

                duration = 0;



            if(ist->dts != AV_NOPTS_VALUE && duration) {

                ist->next_dts += duration;

            }else

                ist->next_dts = AV_NOPTS_VALUE;



            if (got_output)

                ist->next_pts += duration; //FIXME the duration is not correct in some cases

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            ret = transcode_subtitles(ist, &avpkt, &got_output);

            break;

        default:

            return -1;

        }



        if (ret < 0)

            return ret;



        avpkt.dts=

        avpkt.pts= AV_NOPTS_VALUE;



        // touch data and size only if not EOF

        if (pkt) {

            if(ist->st->codec->codec_type != AVMEDIA_TYPE_AUDIO)

                ret = avpkt.size;

            avpkt.data += ret;

            avpkt.size -= ret;

        }

        if (!got_output) {

            continue;

        }

    }



    /* handle stream copy */

    if (!ist->decoding_needed) {

        rate_emu_sleep(ist);

        ist->dts = ist->next_dts;

        switch (ist->st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            ist->next_dts += ((int64_t)AV_TIME_BASE * ist->st->codec->frame_size) /

                             ist->st->codec->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (pkt->duration) {

                ist->next_dts += av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

            } else if(ist->st->codec->time_base.num != 0) {

                int ticks= ist->st->parser ? ist->st->parser->repeat_pict + 1 : ist->st->codec->ticks_per_frame;

                ist->next_dts += ((int64_t)AV_TIME_BASE *

                                  ist->st->codec->time_base.num * ticks) /

                                  ist->st->codec->time_base.den;

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



    return 0;

}
