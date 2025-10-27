static int libshine_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                 const AVFrame *frame, int *got_packet_ptr)

{

    SHINEContext *s = avctx->priv_data;

    MPADecodeHeader hdr;

    unsigned char *data;

    long written;

    int ret, len;



    if (frame)

        data = shine_encode_buffer(s->shine, (int16_t **)frame->data, &written);

    else

        data = shine_flush(s->shine, &written);

    if (written < 0)

        return -1;

    if (written > 0) {

        if (s->buffer_index + written > BUFFER_SIZE) {

            av_log(avctx, AV_LOG_ERROR, "internal buffer too small\n");

            return AVERROR_BUG;

        }

        memcpy(s->buffer + s->buffer_index, data, written);

        s->buffer_index += written;

    }

    if (frame) {

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0)

            return ret;

    }



    if (s->buffer_index < 4 || !s->afq.frame_count)

        return 0;

    if (avpriv_mpegaudio_decode_header(&hdr, AV_RB32(s->buffer))) {

        av_log(avctx, AV_LOG_ERROR, "free format output not supported\n");

        return -1;

    }



    len = hdr.frame_size;

    if (len <= s->buffer_index) {

        if ((ret = ff_alloc_packet2(avctx, avpkt, len)))

            return ret;

        memcpy(avpkt->data, s->buffer, len);

        s->buffer_index -= len;

        memmove(s->buffer, s->buffer + len, s->buffer_index);



        ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                           &avpkt->duration);



        avpkt->size = len;

        *got_packet_ptr = 1;

    }

    return 0;

}
