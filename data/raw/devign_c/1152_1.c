static int ra288_decode_frame(AVCodecContext * avctx, void *data,

                              int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    float *out;

    int i, ret;

    RA288Context *ractx = avctx->priv_data;

    GetBitContext gb;



    if (buf_size < avctx->block_align) {

        av_log(avctx, AV_LOG_ERROR,

               "Error! Input buffer is too small [%d<%d]\n",

               buf_size, avctx->block_align);

        return AVERROR_INVALIDDATA;

    }



    /* get output buffer */

    frame->nb_samples = RA288_BLOCK_SIZE * RA288_BLOCKS_PER_FRAME;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;

    out = (float *)frame->data[0];



    init_get_bits8(&gb, buf, avctx->block_align);



    for (i=0; i < RA288_BLOCKS_PER_FRAME; i++) {

        float gain = amptable[get_bits(&gb, 3)];

        int cb_coef = get_bits(&gb, 6 + (i&1));



        decode(ractx, gain, cb_coef);



        memcpy(out, &ractx->sp_hist[70 + 36], RA288_BLOCK_SIZE * sizeof(*out));

        out += RA288_BLOCK_SIZE;



        if ((i & 7) == 3) {

            backward_filter(ractx, ractx->sp_hist, ractx->sp_rec, syn_window,

                            ractx->sp_lpc, syn_bw_tab, 36, 40, 35, 70);



            backward_filter(ractx, ractx->gain_hist, ractx->gain_rec, gain_window,

                            ractx->gain_lpc, gain_bw_tab, 10, 8, 20, 28);

        }

    }



    *got_frame_ptr = 1;



    return avctx->block_align;

}
