static void do_streamcopy(InputStream *ist, OutputStream *ost, const AVPacket *pkt)

{

    OutputFile *of = output_files[ost->file_index];

    int64_t ost_tb_start_time = av_rescale_q(of->start_time, AV_TIME_BASE_Q, ost->st->time_base);

    AVPacket opkt;



    av_init_packet(&opkt);



    if ((!ost->frame_number && !(pkt->flags & AV_PKT_FLAG_KEY)) &&

        !ost->copy_initial_nonkeyframes)

        return;



    if (of->recording_time != INT64_MAX &&

        ist->last_dts >= of->recording_time + of->start_time) {

        ost->finished = 1;

        return;

    }



    /* force the input stream PTS */

    if (ost->st->codec->codec_type == AVMEDIA_TYPE_AUDIO)

        audio_size += pkt->size;

    else if (ost->st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        video_size += pkt->size;

        ost->sync_opts++;

    }



    if (pkt->pts != AV_NOPTS_VALUE)

        opkt.pts = av_rescale_q(pkt->pts, ist->st->time_base, ost->st->time_base) - ost_tb_start_time;

    else

        opkt.pts = AV_NOPTS_VALUE;



    if (pkt->dts == AV_NOPTS_VALUE)

        opkt.dts = av_rescale_q(ist->last_dts, AV_TIME_BASE_Q, ost->st->time_base);

    else

        opkt.dts = av_rescale_q(pkt->dts, ist->st->time_base, ost->st->time_base);

    opkt.dts -= ost_tb_start_time;



    opkt.duration = av_rescale_q(pkt->duration, ist->st->time_base, ost->st->time_base);

    opkt.flags    = pkt->flags;



    // FIXME remove the following 2 lines they shall be replaced by the bitstream filters

    if (  ost->st->codec->codec_id != AV_CODEC_ID_H264

       && ost->st->codec->codec_id != AV_CODEC_ID_MPEG1VIDEO

       && ost->st->codec->codec_id != AV_CODEC_ID_MPEG2VIDEO

       && ost->st->codec->codec_id != AV_CODEC_ID_VC1

       ) {

        if (av_parser_change(ist->st->parser, ost->st->codec, &opkt.data, &opkt.size, pkt->data, pkt->size, pkt->flags & AV_PKT_FLAG_KEY)) {

            opkt.buf = av_buffer_create(opkt.data, opkt.size, av_buffer_default_free, NULL, 0);

            if (!opkt.buf)

                exit(1);

        }

    } else {

        opkt.data = pkt->data;

        opkt.size = pkt->size;

    }



    write_frame(of->ctx, &opkt, ost);

    ost->st->codec->frame_number++;

}
