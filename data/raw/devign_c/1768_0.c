static int sap_fetch_packet(AVFormatContext *s, AVPacket *pkt)

{

    struct SAPState *sap = s->priv_data;

    int fd = url_get_file_handle(sap->ann_fd);

    int n, ret;

    fd_set rfds;

    struct timeval tv;

    uint8_t recvbuf[1500];



    if (sap->eof)

        return AVERROR_EOF;



    while (1) {

        FD_ZERO(&rfds);

        FD_SET(fd, &rfds);

        tv.tv_sec = tv.tv_usec = 0;

        n = select(fd + 1, &rfds, NULL, NULL, &tv);

        if (n <= 0 || !FD_ISSET(fd, &rfds))

            break;

        ret = url_read(sap->ann_fd, recvbuf, sizeof(recvbuf));

        if (ret >= 8) {

            uint16_t hash = AV_RB16(&recvbuf[2]);

            /* Should ideally check the source IP address, too */

            if (recvbuf[0] & 0x04 && hash == sap->hash) {

                /* Stream deletion */

                sap->eof = 1;

                return AVERROR_EOF;

            }

        }

    }

    ret = av_read_frame(sap->sdp_ctx, pkt);

    if (ret < 0)

        return ret;

    if (s->ctx_flags & AVFMTCTX_NOHEADER) {

        while (sap->sdp_ctx->nb_streams > s->nb_streams) {

            int i = s->nb_streams;

            AVStream *st = av_new_stream(s, i);

            if (!st) {

                av_free_packet(pkt);

                return AVERROR(ENOMEM);

            }

            avcodec_copy_context(st->codec, sap->sdp_ctx->streams[i]->codec);

            st->time_base = sap->sdp_ctx->streams[i]->time_base;

        }

    }

    return ret;

}
