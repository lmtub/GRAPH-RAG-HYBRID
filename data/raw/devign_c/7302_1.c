static int mpegts_read_packet(AVFormatContext *s,

                              AVPacket *pkt)

{

    MpegTSContext *ts = s->priv_data;

    int ret, i;




    ts->pkt = pkt;

    ret = handle_packets(ts, 0);

    if (ret < 0) {

        /* flush pes data left */

        for (i = 0; i < NB_PID_MAX; i++) {

            if (ts->pids[i] && ts->pids[i]->type == MPEGTS_PES) {

                PESContext *pes = ts->pids[i]->u.pes_filter.opaque;

                if (pes->state == MPEGTS_PAYLOAD && pes->data_index > 0) {

                    new_pes_packet(pes, pkt);

                    pes->state = MPEGTS_SKIP;

                    ret = 0;

                    break;

                }

            }

        }

    }



    if (!ret && pkt->size < 0)

        ret = AVERROR(EINTR);

    return ret;

}