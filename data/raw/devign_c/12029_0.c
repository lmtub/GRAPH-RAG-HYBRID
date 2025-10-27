static int prepare_input_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret;



    ret = check_packet(s, pkt);

    if (ret < 0)

        return ret;



#if !FF_API_COMPUTE_PKT_FIELDS2

    /* sanitize the timestamps */

    if (!(s->oformat->flags & AVFMT_NOTIMESTAMPS)) {

        AVStream *st = s->streams[pkt->stream_index];



        /* when there is no reordering (so dts is equal to pts), but

         * only one of them is set, set the other as well */

        if (!st->internal->reorder) {

            if (pkt->pts == AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE)

                pkt->pts = pkt->dts;

            if (pkt->dts == AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE)

                pkt->dts = pkt->pts;

        }



        /* check that the timestamps are set */

        if (pkt->pts == AV_NOPTS_VALUE || pkt->dts == AV_NOPTS_VALUE) {

            av_log(s, AV_LOG_ERROR,

                   "Timestamps are unset in a packet for stream %d\n", st->index);

            return AVERROR(EINVAL);

        }



        /* check that the dts are increasing (or at least non-decreasing,

         * if the format allows it */

        if (st->cur_dts != AV_NOPTS_VALUE &&

            ((!(s->oformat->flags & AVFMT_TS_NONSTRICT) && st->cur_dts >= pkt->dts) ||

             st->cur_dts > pkt->dts)) {

            av_log(s, AV_LOG_ERROR,

                   "Application provided invalid, non monotonically increasing "

                   "dts to muxer in stream %d: %" PRId64 " >= %" PRId64 "\n",

                   st->index, st->cur_dts, pkt->dts);

            return AVERROR(EINVAL);

        }



        if (pkt->pts < pkt->dts) {

            av_log(s, AV_LOG_ERROR, "pts %" PRId64 " < dts %" PRId64 " in stream %d\n",

                   pkt->pts, pkt->dts, st->index);

            return AVERROR(EINVAL);

        }

    }

#endif



    return 0;

}
