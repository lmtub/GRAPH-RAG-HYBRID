static int amr_nb_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                               const AVFrame *frame, int *got_packet_ptr)

{

    AMRContext *s = avctx->priv_data;

    int written, ret;

    int16_t *flush_buf = NULL;

    const int16_t *samples = frame ? (const int16_t *)frame->data[0] : NULL;



    if (s->enc_bitrate != avctx->bit_rate) {

        s->enc_mode    = get_bitrate_mode(avctx->bit_rate, avctx);

        s->enc_bitrate = avctx->bit_rate;

    }



    if ((ret = ff_alloc_packet(avpkt, 32))) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

        return ret;

    }



    if (frame) {

        if (frame->nb_samples < avctx->frame_size) {

            flush_buf = av_mallocz(avctx->frame_size * sizeof(*flush_buf));

            if (!flush_buf)

                return AVERROR(ENOMEM);

            memcpy(flush_buf, samples, frame->nb_samples * sizeof(*flush_buf));

            samples = flush_buf;

            if (frame->nb_samples < avctx->frame_size - avctx->delay)

                s->enc_last_frame = -1;

        }

        if ((ret = ff_af_queue_add(&s->afq, frame)) < 0) {

            av_freep(&flush_buf);

            return ret;

        }

    } else {

        if (s->enc_last_frame < 0)

            return 0;

        flush_buf = av_mallocz(avctx->frame_size * sizeof(*flush_buf));

        if (!flush_buf)

            return AVERROR(ENOMEM);

        samples = flush_buf;

        s->enc_last_frame = -1;

    }



    written = Encoder_Interface_Encode(s->enc_state, s->enc_mode, samples,

                                       avpkt->data, 0);

    av_dlog(avctx, "amr_nb_encode_frame encoded %u bytes, bitrate %u, first byte was %#02x\n",

            written, s->enc_mode, frame[0]);



    /* Get the next frame pts/duration */

    ff_af_queue_remove(&s->afq, avctx->frame_size, &avpkt->pts,

                       &avpkt->duration);



    avpkt->size = written;

    *got_packet_ptr = 1;

    av_freep(&flush_buf);

    return 0;

}
