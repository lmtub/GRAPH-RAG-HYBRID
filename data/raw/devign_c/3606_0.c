rdt_parse_packet (PayloadContext *rdt, AVStream *st,

                  AVPacket *pkt, uint32_t *timestamp,

                  const uint8_t *buf, int len, int flags)

{

    int seq = 1, res;

    ByteIOContext pb;



    if (rdt->audio_pkt_cnt == 0) {

        int pos;



        init_put_byte(&pb, buf, len, 0, NULL, NULL, NULL, NULL);

        flags = (flags & PKT_FLAG_KEY) ? 2 : 0;

        res = ff_rm_parse_packet (rdt->rmctx, &pb, st, rdt->rmst[0], len, pkt,

                                  &seq, &flags, timestamp);

        pos = url_ftell(&pb);

        if (res < 0)

            return res;

        rdt->audio_pkt_cnt[st->id] = res;

        if (rdt->audio_pkt_cnt[st->id] > 0 &&

            st->codec->codec_id == CODEC_ID_AAC) {

            memcpy (rdt->buffer, buf + pos, len - pos);

            rdt->rmctx->pb = av_alloc_put_byte (rdt->buffer, len - pos, 0,

                                                NULL, NULL, NULL, NULL);

        }

    } else {

        ff_rm_retrieve_cache (rdt->rmctx, rdt->rmctx->pb, st, rdt->rmst[0], pkt);

        if (rdt->audio_pkt_cnt[st->id] == 0 &&

            st->codec->codec_id == CODEC_ID_AAC)

            av_freep(&rdt->rmctx->pb);

    }

    pkt->stream_index = st->index;

    pkt->pts = *timestamp;



    return rdt->audio_pkt_cnt[st->id] > 0;

}
