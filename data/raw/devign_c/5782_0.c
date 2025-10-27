static int check_pkt(AVFormatContext *s, AVPacket *pkt)

{

    MOVMuxContext *mov = s->priv_data;

    MOVTrack *trk = &mov->tracks[pkt->stream_index];



    if (trk->entry) {

        int64_t duration = pkt->dts - trk->cluster[trk->entry - 1].dts;

        if (duration < 0 || duration > INT_MAX) {

            av_log(s, AV_LOG_ERROR, "Application provided duration: %"PRId64" / timestamp: %"PRId64" is out of range for mov/mp4 format\n",

                duration, pkt->dts

            );



            pkt->dts = trk->cluster[trk->entry - 1].dts + 1;

            pkt->pts = AV_NOPTS_VALUE;

        }

    } else if (pkt->dts <= INT_MIN || pkt->dts >= INT_MAX) {

            av_log(s, AV_LOG_ERROR, "Application provided initial timestamp: %"PRId64" is out of range for mov/mp4 format\n",

                pkt->dts

            );



            pkt->dts = 0;

            pkt->pts = AV_NOPTS_VALUE;

    }

    if (pkt->duration < 0 || pkt->duration > INT_MAX) {

        av_log(s, AV_LOG_ERROR, "Application provided duration: %"PRId64" is invalid\n", pkt->duration);

        return AVERROR(EINVAL);

    }

    return 0;

}
