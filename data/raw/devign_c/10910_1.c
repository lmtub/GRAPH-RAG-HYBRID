static int vqa_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    VqaContext *s = avctx->priv_data;



    s->buf = buf;

    s->size = buf_size;



    if (s->frame.data[0])

        avctx->release_buffer(avctx, &s->frame);



    if (avctx->get_buffer(avctx, &s->frame)) {

        av_log(s->avctx, AV_LOG_ERROR, "  VQA Video: get_buffer() failed\n");

        return -1;

    }



    vqa_decode_chunk(s);



    /* make the palette available on the way out */

    memcpy(s->frame.data[1], s->palette, PALETTE_COUNT * 4);

    s->frame.palette_has_changed = 1;



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    /* report that the buffer was completely consumed */

    return buf_size;

}
