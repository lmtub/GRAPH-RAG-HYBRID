static void apply_window_and_mdct(AVCodecContext *avctx, const AVFrame *frame)

{

    WMACodecContext *s = avctx->priv_data;

    float **audio      = (float **) frame->extended_data;

    int len            = frame->nb_samples;

    int window_index   = s->frame_len_bits - s->block_len_bits;

    FFTContext *mdct   = &s->mdct_ctx[window_index];

    int ch;

    const float *win   = s->windows[window_index];

    int window_len     = 1 << s->block_len_bits;

    float n            = 2.0 * 32768.0 / window_len;



    for (ch = 0; ch < avctx->channels; ch++) {

        memcpy(s->output, s->frame_out[ch], window_len * sizeof(*s->output));

        s->fdsp->vector_fmul_scalar(s->frame_out[ch], audio[ch], n, len);

        s->fdsp->vector_fmul_reverse(&s->output[window_len], s->frame_out[ch],

                                    win, len);

        s->fdsp->vector_fmul(s->frame_out[ch], s->frame_out[ch], win, len);

        mdct->mdct_calc(mdct, s->coefs[ch], s->output);

    }

}
