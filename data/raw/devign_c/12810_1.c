static int write_packet(AVFormatContext *s, AVPacket *pkt)
{
    int ret, did_split;
    if (s->output_ts_offset) {
        AVStream *st = s->streams[pkt->stream_index];
        int64_t offset = av_rescale_q(s->output_ts_offset, AV_TIME_BASE_Q, st->time_base);
        if (pkt->dts != AV_NOPTS_VALUE)
            pkt->dts += offset;
        if (pkt->pts != AV_NOPTS_VALUE)
            pkt->pts += offset;
    }
    if (s->avoid_negative_ts > 0) {
        AVStream *st = s->streams[pkt->stream_index];
        int64_t offset = st->mux_ts_offset;
        int64_t ts = s->internal->avoid_negative_ts_use_pts ? pkt->pts : pkt->dts;
        if (s->internal->offset == AV_NOPTS_VALUE && ts != AV_NOPTS_VALUE &&
            (ts < 0 || s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_MAKE_ZERO)) {
            s->internal->offset = -ts;
            s->internal->offset_timebase = st->time_base;
        }
        if (s->internal->offset != AV_NOPTS_VALUE && !offset) {
            offset = st->mux_ts_offset =
                av_rescale_q_rnd(s->internal->offset,
                                 s->internal->offset_timebase,
                                 st->time_base,
                                 AV_ROUND_UP);
        }
        if (pkt->dts != AV_NOPTS_VALUE)
            pkt->dts += offset;
        if (pkt->pts != AV_NOPTS_VALUE)
            pkt->pts += offset;
        if (s->internal->avoid_negative_ts_use_pts) {
            if (pkt->pts != AV_NOPTS_VALUE && pkt->pts < 0) {
                av_log(s, AV_LOG_WARNING, "failed to avoid negative "
                    "pts %s in stream %d.\n"
                    "Try -avoid_negative_ts 1 as a possible workaround.\n",
                    av_ts2str(pkt->dts),
                    pkt->stream_index
                );
            }
        } else {
            av_assert2(pkt->dts == AV_NOPTS_VALUE || pkt->dts >= 0 || s->max_interleave_delta > 0);
            if (pkt->dts != AV_NOPTS_VALUE && pkt->dts < 0) {
                av_log(s, AV_LOG_WARNING,
                    "Packets poorly interleaved, failed to avoid negative "
                    "timestamp %s in stream %d.\n"
                    "Try -max_interleave_delta 0 as a possible workaround.\n",
                    av_ts2str(pkt->dts),
                    pkt->stream_index
                );
            }
        }
    }
    did_split = av_packet_split_side_data(pkt);
    if (!s->internal->header_written) {
        ret = s->internal->write_header_ret ? s->internal->write_header_ret : write_header_internal(s);
        if (ret < 0)
            goto fail;
    }
    if ((pkt->flags & AV_PKT_FLAG_UNCODED_FRAME)) {
        AVFrame *frame = (AVFrame *)pkt->data;
        av_assert0(pkt->size == UNCODED_FRAME_PACKET_SIZE);
        ret = s->oformat->write_uncoded_frame(s, pkt->stream_index, &frame, 0);
        av_frame_free(&frame);
    } else {
        ret = s->oformat->write_packet(s, pkt);
    }
    if (s->pb && ret >= 0) {
        if (s->flush_packets && s->flags & AVFMT_FLAG_FLUSH_PACKETS)
            avio_flush(s->pb);
        if (s->pb->error < 0)
            ret = s->pb->error;
    }
fail:
    if (did_split)
        av_packet_merge_side_data(pkt);
    if (ret < 0) {
        pkt->pts = pts_backup;
        pkt->dts = dts_backup;
    }
    return ret;
}