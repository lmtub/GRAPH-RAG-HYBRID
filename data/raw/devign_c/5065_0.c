int ff_twinvq_decode_frame(AVCodecContext *avctx, void *data,

                           int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    TwinVQContext *tctx = avctx->priv_data;

    const TwinVQModeTab *mtab = tctx->mtab;

    float **out = NULL;

    int ret;



    /* get output buffer */

    if (tctx->discarded_packets >= 2) {

        frame->nb_samples = mtab->size;

        if ((ret = ff_get_buffer(avctx, frame, 0)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return ret;

        }

        out = (float **)frame->extended_data;

    }



    if ((ret = tctx->read_bitstream(avctx, tctx, buf, buf_size)) < 0)

        return ret;



    read_and_decode_spectrum(tctx, tctx->spectrum, tctx->bits.ftype);



    imdct_output(tctx, tctx->bits.ftype, tctx->bits.window_type, out);



    FFSWAP(float *, tctx->curr_frame, tctx->prev_frame);



    if (tctx->discarded_packets < 2) {

        tctx->discarded_packets++;

        *got_frame_ptr = 0;

        return buf_size;

    }



    *got_frame_ptr = 1;



    return buf_size;

}
