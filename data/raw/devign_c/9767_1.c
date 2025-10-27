static int dshow_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    struct dshow_ctx *ctx = s->priv_data;

    AVPacketList *pktl = NULL;



    while (!ctx->eof && !pktl) {

        WaitForSingleObject(ctx->mutex, INFINITE);

        pktl = ctx->pktl;

        if (pktl) {

            *pkt = pktl->pkt;

            ctx->pktl = ctx->pktl->next;

            av_free(pktl);

            ctx->curbufsize -= pkt->size;

        }

        ResetEvent(ctx->event[1]);

        ReleaseMutex(ctx->mutex);

        if (!pktl) {

            if (dshow_check_event_queue(ctx->media_event) < 0) {

                ctx->eof = 1;

            } else if (s->flags & AVFMT_FLAG_NONBLOCK) {

                return AVERROR(EAGAIN);

            } else {

                WaitForMultipleObjects(2, ctx->event, 0, INFINITE);

            }

        }

    }



    return ctx->eof ? AVERROR(EIO) : pkt->size;

}
