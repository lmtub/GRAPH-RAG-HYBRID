static int check_pkt(AVFormatContext *s, AVPacket *pkt)
{
    MOVMuxContext *mov = s->priv_data;
    MOVTrack *trk = &mov->tracks[pkt->stream_index];
    int64_t ref;
    uint64_t duration;
    if (trk->entry) {
        ref = trk->cluster[trk->entry - 1].dts;
    } else if (   trk->start_dts != AV_NOPTS_VALUE
               && !trk->frag_discont) {
        ref = trk->start_dts + trk->track_duration;
    } else
        ref = pkt->dts; // Skip tests for the first packet
    duration = pkt->dts - ref;
    if (pkt->dts < ref || duration >= INT_MAX) {
        av_log(s, AV_LOG_ERROR, "Application provided duration: %"PRId64" / timestamp: %"PRId64" is out of range for mov/mp4 format\n",
            duration, pkt->dts
        );
        pkt->dts = ref + 1;
        pkt->pts = AV_NOPTS_VALUE;
    if (pkt->duration < 0 || pkt->duration > INT_MAX) {
        av_log(s, AV_LOG_ERROR, "Application provided duration: %"PRId64" is invalid\n", pkt->duration);
        return AVERROR(EINVAL);
    return 0;