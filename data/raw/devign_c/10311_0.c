int avio_close_dyn_buf(AVIOContext *s, uint8_t **pbuffer)

{

    DynBuffer *d;

    int size;

    static const char padbuf[AV_INPUT_BUFFER_PADDING_SIZE] = {0};

    int padding = 0;



    if (!s) {

        *pbuffer = NULL;

        return 0;

    }



    /* don't attempt to pad fixed-size packet buffers */

    if (!s->max_packet_size) {

        avio_write(s, padbuf, sizeof(padbuf));

        padding = AV_INPUT_BUFFER_PADDING_SIZE;

    }



    avio_flush(s);



    d = s->opaque;

    *pbuffer = d->buffer;

    size = d->size;

    av_free(d);

    av_free(s);

    return size - padding;

}
