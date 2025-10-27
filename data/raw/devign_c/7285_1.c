static int tcp_write_packet(AVFormatContext *s, RTSPStream *rtsp_st)

{

    RTSPState *rt = s->priv_data;

    AVFormatContext *rtpctx = rtsp_st->transport_priv;

    uint8_t *buf, *ptr;

    int size;

    uint8_t *interleave_header, *interleaved_packet;



    size = avio_close_dyn_buf(rtpctx->pb, &buf);

    ptr = buf;

    while (size > 4) {

        uint32_t packet_len = AV_RB32(ptr);

        int id;

        /* The interleaving header is exactly 4 bytes, which happens to be

         * the same size as the packet length header from

         * ffio_open_dyn_packet_buf. So by writing the interleaving header

         * over these bytes, we get a consecutive interleaved packet

         * that can be written in one call. */

        interleaved_packet = interleave_header = ptr;

        ptr += 4;

        size -= 4;

        if (packet_len > size || packet_len < 2)

            break;

        if (RTP_PT_IS_RTCP(ptr[1]))

            id = rtsp_st->interleaved_max; /* RTCP */

        else

            id = rtsp_st->interleaved_min; /* RTP */

        interleave_header[0] = '$';

        interleave_header[1] = id;

        AV_WB16(interleave_header + 2, packet_len);

        ffurl_write(rt->rtsp_hd_out, interleaved_packet, 4 + packet_len);

        ptr += packet_len;

        size -= packet_len;

    }

    av_free(buf);

    ffio_open_dyn_packet_buf(&rtpctx->pb, RTSP_TCP_MAX_PACKET_SIZE);

    return 0;

}
