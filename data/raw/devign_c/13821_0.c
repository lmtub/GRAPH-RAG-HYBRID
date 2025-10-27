static av_cold int decoder_init(AVCodecContext * avctx)

{

    G729Context* ctx = avctx->priv_data;

    int i,k;



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Only mono sound is supported (requested channels: %d).\n", avctx->channels);

        return AVERROR(EINVAL);

    }

    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    /* Both 8kbit/s and 6.4kbit/s modes uses two subframes per frame. */

    avctx->frame_size = SUBFRAME_SIZE << 1;



    ctx->gain_coeff = 16384; // 1.0 in (1.14)



    for (k = 0; k < MA_NP + 1; k++) {

        ctx->past_quantizer_outputs[k] = ctx->past_quantizer_output_buf[k];

        for (i = 1; i < 11; i++)

            ctx->past_quantizer_outputs[k][i - 1] = (18717 * i) >> 3;

    }



    ctx->lsp[0] = ctx->lsp_buf[0];

    ctx->lsp[1] = ctx->lsp_buf[1];

    memcpy(ctx->lsp[0], lsp_init, 10 * sizeof(int16_t));



    ctx->exc = &ctx->exc_base[PITCH_DELAY_MAX+INTERPOL_LEN];



    /* random seed initialization */

    ctx->rand_value = 21845;



    /* quantized prediction error */

    for(i=0; i<4; i++)

        ctx->quant_energy[i] = -14336; // -14 in (5.10)



    avctx->dsp_mask= ~AV_CPU_FLAG_FORCE;

    dsputil_init(&ctx->dsp, avctx);



    return 0;

}
