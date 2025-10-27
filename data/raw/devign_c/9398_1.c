static int g723_1_decode_frame(AVCodecContext *avctx, void *data,

                               int *got_frame_ptr, AVPacket *avpkt)

{

    G723_1_Context *p  = avctx->priv_data;

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    int dec_mode       = buf[0] & 3;



    PPFParam ppf[SUBFRAMES];

    int16_t cur_lsp[LPC_ORDER];

    int16_t lpc[SUBFRAMES * LPC_ORDER];

    int16_t acb_vector[SUBFRAME_LEN];

    int16_t *out;

    int bad_frame = 0, i, j, ret;

    int16_t *audio = p->audio;



    if (buf_size < frame_size[dec_mode]) {

        if (buf_size)

            av_log(avctx, AV_LOG_WARNING,

                   "Expected %d bytes, got %d - skipping packet\n",

                   frame_size[dec_mode], buf_size);

        *got_frame_ptr = 0;

        return buf_size;

    }



    if (unpack_bitstream(p, buf, buf_size) < 0) {

        bad_frame = 1;

        if (p->past_frame_type == ACTIVE_FRAME)

            p->cur_frame_type = ACTIVE_FRAME;

        else

            p->cur_frame_type = UNTRANSMITTED_FRAME;

    }



    frame->nb_samples = FRAME_LEN;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;



    out = (int16_t *)frame->data[0];



    if (p->cur_frame_type == ACTIVE_FRAME) {

        if (!bad_frame)

            p->erased_frames = 0;

        else if (p->erased_frames != 3)

            p->erased_frames++;



        ff_g723_1_inverse_quant(cur_lsp, p->prev_lsp, p->lsp_index, bad_frame);

        ff_g723_1_lsp_interpolate(lpc, cur_lsp, p->prev_lsp);



        /* Save the lsp_vector for the next frame */

        memcpy(p->prev_lsp, cur_lsp, LPC_ORDER * sizeof(*p->prev_lsp));



        /* Generate the excitation for the frame */

        memcpy(p->excitation, p->prev_excitation,

               PITCH_MAX * sizeof(*p->excitation));

        if (!p->erased_frames) {

            int16_t *vector_ptr = p->excitation + PITCH_MAX;



            /* Update interpolation gain memory */

            p->interp_gain = fixed_cb_gain[(p->subframe[2].amp_index +

                                            p->subframe[3].amp_index) >> 1];

            for (i = 0; i < SUBFRAMES; i++) {

                gen_fcb_excitation(vector_ptr, &p->subframe[i], p->cur_rate,

                                   p->pitch_lag[i >> 1], i);

                ff_g723_1_gen_acb_excitation(acb_vector,

                                             &p->excitation[SUBFRAME_LEN * i],

                                             p->pitch_lag[i >> 1],

                                             &p->subframe[i], p->cur_rate);

                /* Get the total excitation */

                for (j = 0; j < SUBFRAME_LEN; j++) {

                    int v = av_clip_int16(vector_ptr[j] << 1);

                    vector_ptr[j] = av_clip_int16(v + acb_vector[j]);

                }

                vector_ptr += SUBFRAME_LEN;

            }



            vector_ptr = p->excitation + PITCH_MAX;



            p->interp_index = comp_interp_index(p, p->pitch_lag[1],

                                                &p->sid_gain, &p->cur_gain);



            /* Perform pitch postfiltering */

            if (p->postfilter) {

                i = PITCH_MAX;

                for (j = 0; j < SUBFRAMES; i += SUBFRAME_LEN, j++)

                    comp_ppf_coeff(p, i, p->pitch_lag[j >> 1],

                                   ppf + j, p->cur_rate);



                for (i = 0, j = 0; j < SUBFRAMES; i += SUBFRAME_LEN, j++)

                    ff_acelp_weighted_vector_sum(p->audio + LPC_ORDER + i,

                                                 vector_ptr + i,

                                                 vector_ptr + i + ppf[j].index,

                                                 ppf[j].sc_gain,

                                                 ppf[j].opt_gain,

                                                 1 << 14, 15, SUBFRAME_LEN);

            } else {

                audio = vector_ptr - LPC_ORDER;

            }



            /* Save the excitation for the next frame */

            memcpy(p->prev_excitation, p->excitation + FRAME_LEN,

                   PITCH_MAX * sizeof(*p->excitation));

        } else {

            p->interp_gain = (p->interp_gain * 3 + 2) >> 2;

            if (p->erased_frames == 3) {

                /* Mute output */

                memset(p->excitation, 0,

                       (FRAME_LEN + PITCH_MAX) * sizeof(*p->excitation));

                memset(p->prev_excitation, 0,

                       PITCH_MAX * sizeof(*p->excitation));

                memset(frame->data[0], 0,

                       (FRAME_LEN + LPC_ORDER) * sizeof(int16_t));

            } else {

                int16_t *buf = p->audio + LPC_ORDER;



                /* Regenerate frame */

                residual_interp(p->excitation, buf, p->interp_index,

                                p->interp_gain, &p->random_seed);



                /* Save the excitation for the next frame */

                memcpy(p->prev_excitation, buf + (FRAME_LEN - PITCH_MAX),

                       PITCH_MAX * sizeof(*p->excitation));

            }

        }

        p->cng_random_seed = CNG_RANDOM_SEED;

    } else {

        if (p->cur_frame_type == SID_FRAME) {

            p->sid_gain = sid_gain_to_lsp_index(p->subframe[0].amp_index);

            ff_g723_1_inverse_quant(p->sid_lsp, p->prev_lsp, p->lsp_index, 0);

        } else if (p->past_frame_type == ACTIVE_FRAME) {

            p->sid_gain = estimate_sid_gain(p);

        }



        if (p->past_frame_type == ACTIVE_FRAME)

            p->cur_gain = p->sid_gain;

        else

            p->cur_gain = (p->cur_gain * 7 + p->sid_gain) >> 3;

        generate_noise(p);

        ff_g723_1_lsp_interpolate(lpc, p->sid_lsp, p->prev_lsp);

        /* Save the lsp_vector for the next frame */

        memcpy(p->prev_lsp, p->sid_lsp, LPC_ORDER * sizeof(*p->prev_lsp));

    }



    p->past_frame_type = p->cur_frame_type;



    memcpy(p->audio, p->synth_mem, LPC_ORDER * sizeof(*p->audio));

    for (i = LPC_ORDER, j = 0; j < SUBFRAMES; i += SUBFRAME_LEN, j++)

        ff_celp_lp_synthesis_filter(p->audio + i, &lpc[j * LPC_ORDER],

                                    audio + i, SUBFRAME_LEN, LPC_ORDER,

                                    0, 1, 1 << 12);

    memcpy(p->synth_mem, p->audio + FRAME_LEN, LPC_ORDER * sizeof(*p->audio));



    if (p->postfilter) {

        formant_postfilter(p, lpc, p->audio, out);

    } else { // if output is not postfiltered it should be scaled by 2

        for (i = 0; i < FRAME_LEN; i++)

            out[i] = av_clip_int16(p->audio[LPC_ORDER + i] << 1);

    }



    *got_frame_ptr = 1;



    return frame_size[dec_mode];

}
