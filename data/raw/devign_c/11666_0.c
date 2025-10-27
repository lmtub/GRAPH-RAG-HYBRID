int av_read_frame(AVFormatContext *s, AVPacket *pkt)

{

    const int genpts = s->flags & AVFMT_FLAG_GENPTS;

    int eof = 0;



    if (!genpts)

        return s->internal->packet_buffer

               ? read_from_packet_buffer(&s->internal->packet_buffer,

                                         &s->internal->packet_buffer_end, pkt)

               : read_frame_internal(s, pkt);



    for (;;) {

        int ret;

        AVPacketList *pktl = s->internal->packet_buffer;



        if (pktl) {

            AVPacket *next_pkt = &pktl->pkt;



            if (next_pkt->dts != AV_NOPTS_VALUE) {

                int wrap_bits = s->streams[next_pkt->stream_index]->pts_wrap_bits;

                while (pktl && next_pkt->pts == AV_NOPTS_VALUE) {

                    if (pktl->pkt.stream_index == next_pkt->stream_index &&

                        (av_compare_mod(next_pkt->dts, pktl->pkt.dts, 2LL << (wrap_bits - 1)) < 0) &&

                         av_compare_mod(pktl->pkt.pts, pktl->pkt.dts, 2LL << (wrap_bits - 1))) {

                        // not B-frame

                        next_pkt->pts = pktl->pkt.dts;

                    }

                    pktl = pktl->next;

                }

                pktl = s->internal->packet_buffer;

            }



            /* read packet from packet buffer, if there is data */

            if (!(next_pkt->pts == AV_NOPTS_VALUE &&

                  next_pkt->dts != AV_NOPTS_VALUE && !eof))

                return read_from_packet_buffer(&s->internal->packet_buffer,

                                               &s->internal->packet_buffer_end, pkt);

        }



        ret = read_frame_internal(s, pkt);

        if (ret < 0) {

            if (pktl && ret != AVERROR(EAGAIN)) {

                eof = 1;

                continue;

            } else

                return ret;

        }



        if (av_dup_packet(add_to_pktbuf(&s->internal->packet_buffer, pkt,

                                        &s->internal->packet_buffer_end)) < 0)

            return AVERROR(ENOMEM);

    }

}
