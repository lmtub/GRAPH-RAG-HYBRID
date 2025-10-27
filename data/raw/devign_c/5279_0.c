static int concat_read_packet(AVFormatContext *avf, AVPacket *pkt)

{

    ConcatContext *cat = avf->priv_data;

    int ret;

    int64_t delta;

    ConcatStream *cs;



    while (1) {

        ret = av_read_frame(cat->avf, pkt);

        if (ret == AVERROR_EOF) {

            if ((ret = open_next_file(avf)) < 0)

                return ret;

            continue;

        }

        if (ret < 0)

            return ret;

        if (cat->match_streams) {

            match_streams(avf);

            cs = &cat->cur_file->streams[pkt->stream_index];

            if (cs->out_stream_index < 0) {

                av_packet_unref(pkt);

                continue;

            }

            pkt->stream_index = cs->out_stream_index;

        }

        break;

    }



    delta = av_rescale_q(cat->cur_file->start_time - cat->avf->start_time,

                         AV_TIME_BASE_Q,

                         cat->avf->streams[pkt->stream_index]->time_base);

    if (pkt->pts != AV_NOPTS_VALUE)

        pkt->pts += delta;

    if (pkt->dts != AV_NOPTS_VALUE)

        pkt->dts += delta;

    return ret;

}
