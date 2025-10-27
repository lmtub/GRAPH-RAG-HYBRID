static int g723_1_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                            const AVFrame *frame, int *got_packet_ptr)

{

    G723_1_Context *p = avctx->priv_data;

    int16_t unq_lpc[LPC_ORDER * SUBFRAMES];

    int16_t qnt_lpc[LPC_ORDER * SUBFRAMES];

    int16_t cur_lsp[LPC_ORDER];

    int16_t weighted_lpc[LPC_ORDER * SUBFRAMES << 1];

    int16_t vector[FRAME_LEN + PITCH_MAX];

    int offset, ret;

    int16_t *in = (const int16_t *)frame->data[0];



    HFParam hf[4];

    int i, j;



    highpass_filter(in, &p->hpf_fir_mem, &p->hpf_iir_mem);



    memcpy(vector, p->prev_data, HALF_FRAME_LEN * sizeof(int16_t));

    memcpy(vector + HALF_FRAME_LEN, in, FRAME_LEN * sizeof(int16_t));



    comp_lpc_coeff(vector, unq_lpc);

    lpc2lsp(&unq_lpc[LPC_ORDER * 3], p->prev_lsp, cur_lsp);

    lsp_quantize(p->lsp_index, cur_lsp, p->prev_lsp);



    /* Update memory */

    memcpy(vector + LPC_ORDER, p->prev_data + SUBFRAME_LEN,

           sizeof(int16_t) * SUBFRAME_LEN);

    memcpy(vector + LPC_ORDER + SUBFRAME_LEN, in,

           sizeof(int16_t) * (HALF_FRAME_LEN + SUBFRAME_LEN));

    memcpy(p->prev_data, in + HALF_FRAME_LEN,

           sizeof(int16_t) * HALF_FRAME_LEN);

    memcpy(in, vector + LPC_ORDER, sizeof(int16_t) * FRAME_LEN);



    perceptual_filter(p, weighted_lpc, unq_lpc, vector);



    memcpy(in, vector + LPC_ORDER, sizeof(int16_t) * FRAME_LEN);

    memcpy(vector, p->prev_weight_sig, sizeof(int16_t) * PITCH_MAX);

    memcpy(vector + PITCH_MAX, in, sizeof(int16_t) * FRAME_LEN);



    scale_vector(vector, vector, FRAME_LEN + PITCH_MAX);



    p->pitch_lag[0] = estimate_pitch(vector, PITCH_MAX);

    p->pitch_lag[1] = estimate_pitch(vector, PITCH_MAX + HALF_FRAME_LEN);



    for (i = PITCH_MAX, j = 0; j < SUBFRAMES; i += SUBFRAME_LEN, j++)

        comp_harmonic_coeff(vector + i, p->pitch_lag[j >> 1], hf + j);



    memcpy(vector, p->prev_weight_sig, sizeof(int16_t) * PITCH_MAX);

    memcpy(vector + PITCH_MAX, in, sizeof(int16_t) * FRAME_LEN);

    memcpy(p->prev_weight_sig, vector + FRAME_LEN, sizeof(int16_t) * PITCH_MAX);



    for (i = 0, j = 0; j < SUBFRAMES; i += SUBFRAME_LEN, j++)

        harmonic_filter(hf + j, vector + PITCH_MAX + i, in + i);



    inverse_quant(cur_lsp, p->prev_lsp, p->lsp_index, 0);

    lsp_interpolate(qnt_lpc, cur_lsp, p->prev_lsp);



    memcpy(p->prev_lsp, cur_lsp, sizeof(int16_t) * LPC_ORDER);



    offset = 0;

    for (i = 0; i < SUBFRAMES; i++) {

        int16_t impulse_resp[SUBFRAME_LEN];

        int16_t residual[SUBFRAME_LEN + PITCH_ORDER - 1];

        int16_t flt_in[SUBFRAME_LEN];

        int16_t zero[LPC_ORDER], fir[LPC_ORDER], iir[LPC_ORDER];



        /**

         * Compute the combined impulse response of the synthesis filter,

         * formant perceptual weighting filter and harmonic noise shaping filter

         */

        memset(zero, 0, sizeof(int16_t) * LPC_ORDER);

        memset(vector, 0, sizeof(int16_t) * PITCH_MAX);

        memset(flt_in, 0, sizeof(int16_t) * SUBFRAME_LEN);



        flt_in[0] = 1 << 13; /* Unit impulse */

        synth_percept_filter(qnt_lpc + offset, weighted_lpc + (offset << 1),

                             zero, zero, flt_in, vector + PITCH_MAX, 1);

        harmonic_filter(hf + i, vector + PITCH_MAX, impulse_resp);



         /* Compute the combined zero input response */

        flt_in[0] = 0;

        memcpy(fir, p->perf_fir_mem, sizeof(int16_t) * LPC_ORDER);

        memcpy(iir, p->perf_iir_mem, sizeof(int16_t) * LPC_ORDER);



        synth_percept_filter(qnt_lpc + offset, weighted_lpc + (offset << 1),

                             fir, iir, flt_in, vector + PITCH_MAX, 0);

        memcpy(vector, p->harmonic_mem, sizeof(int16_t) * PITCH_MAX);

        harmonic_noise_sub(hf + i, vector + PITCH_MAX, in);



        acb_search(p, residual, impulse_resp, in, i);

        gen_acb_excitation(residual, p->prev_excitation,p->pitch_lag[i >> 1],

                           &p->subframe[i], p->cur_rate);

        sub_acb_contrib(residual, impulse_resp, in);



        fcb_search(p, impulse_resp, in, i);



        /* Reconstruct the excitation */

        gen_acb_excitation(impulse_resp, p->prev_excitation, p->pitch_lag[i >> 1],

                           &p->subframe[i], RATE_6300);



        memmove(p->prev_excitation, p->prev_excitation + SUBFRAME_LEN,

               sizeof(int16_t) * (PITCH_MAX - SUBFRAME_LEN));

        for (j = 0; j < SUBFRAME_LEN; j++)

            in[j] = av_clip_int16((in[j] << 1) + impulse_resp[j]);

        memcpy(p->prev_excitation + PITCH_MAX - SUBFRAME_LEN, in,

               sizeof(int16_t) * SUBFRAME_LEN);



        /* Update filter memories */

        synth_percept_filter(qnt_lpc + offset, weighted_lpc + (offset << 1),

                             p->perf_fir_mem, p->perf_iir_mem,

                             in, vector + PITCH_MAX, 0);

        memmove(p->harmonic_mem, p->harmonic_mem + SUBFRAME_LEN,

                sizeof(int16_t) * (PITCH_MAX - SUBFRAME_LEN));

        memcpy(p->harmonic_mem + PITCH_MAX - SUBFRAME_LEN, vector + PITCH_MAX,

               sizeof(int16_t) * SUBFRAME_LEN);



        in += SUBFRAME_LEN;

        offset += LPC_ORDER;

    }



    if ((ret = ff_alloc_packet2(avctx, avpkt, 24)))

        return ret;



    *got_packet_ptr = 1;

    avpkt->size = pack_bitstream(p, avpkt->data, avpkt->size);

    return 0;

}
