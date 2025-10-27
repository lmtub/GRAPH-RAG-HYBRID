static int libspeex_decode_frame(AVCodecContext *avctx, void *data,

                                 int *got_frame_ptr, AVPacket *avpkt)

{

    uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    LibSpeexContext *s = avctx->priv_data;

    int16_t *output;

    int ret, consumed = 0;



    /* get output buffer */

    s->frame.nb_samples = s->frame_size;

    if ((ret = ff_get_buffer(avctx, &s->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    output = (int16_t *)s->frame.data[0];



    /* if there is not enough data left for the smallest possible frame or the

       next 5 bits are a terminator code, reset the libspeex buffer using the

       current packet, otherwise ignore the current packet and keep decoding

       frames from the libspeex buffer. */

    if (speex_bits_remaining(&s->bits) < 5 ||

        speex_bits_peek_unsigned(&s->bits, 5) == 0x1F) {

        /* check for flush packet */

        if (!buf || !buf_size) {

            *got_frame_ptr = 0;

            return buf_size;

        }

        /* set new buffer */

        speex_bits_read_from(&s->bits, buf, buf_size);

        consumed = buf_size;

    }



    /* decode a single frame */

    ret = speex_decode_int(s->dec_state, &s->bits, output);

    if (ret <= -2) {

        av_log(avctx, AV_LOG_ERROR, "Error decoding Speex frame.\n");

        return AVERROR_INVALIDDATA;

    }

    if (avctx->channels == 2)

        speex_decode_stereo_int(output, s->frame_size, &s->stereo);



    *got_frame_ptr   = 1;

    *(AVFrame *)data = s->frame;



    return consumed;

}
