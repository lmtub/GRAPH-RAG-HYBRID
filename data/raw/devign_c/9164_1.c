static int aac_parse_packet(AVFormatContext *ctx, PayloadContext *data,

                            AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                            const uint8_t *buf, int len, uint16_t seq,

                            int flags)

{

    int ret;

    if (rtp_parse_mp4_au(data, buf))

        return -1;



    buf += data->au_headers_length_bytes + 2;

    len -= data->au_headers_length_bytes + 2;



    /* XXX: Fixme we only handle the case where rtp_parse_mp4_au define

                    one au_header */

    if ((ret = av_new_packet(pkt, data->au_headers[0].size)) < 0)

        return ret;

    memcpy(pkt->data, buf, data->au_headers[0].size);



    pkt->stream_index = st->index;

    return 0;

}
