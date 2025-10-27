static int idcin_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    IdcinContext *s = avctx->priv_data;

    const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);



    s->buf = buf;

    s->size = buf_size;



    if (s->frame.data[0])

        avctx->release_buffer(avctx, &s->frame);



    if (avctx->get_buffer(avctx, &s->frame)) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    idcin_decode_vlcs(s);



    if (pal) {

        s->frame.palette_has_changed = 1;

        memcpy(s->pal, pal, AVPALETTE_SIZE);

    }

    /* make the palette available on the way out */

    memcpy(s->frame.data[1], s->pal, AVPALETTE_SIZE);



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    /* report that the buffer was completely consumed */

    return buf_size;

}
