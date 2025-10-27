static int evrc_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    AVFrame *frame     = data;

    EVRCContext *e     = avctx->priv_data;

    int buf_size       = avpkt->size;

    float ilspf[FILTER_ORDER], ilpc[FILTER_ORDER], idelay[NB_SUBFRAMES];

    float *samples;

    int   i, j, ret, error_flag = 0;



    frame->nb_samples = 160;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;

    samples = (float *)frame->data[0];



    if ((e->bitrate = determine_bitrate(avctx, &buf_size, &buf)) == RATE_ERRS) {

        warn_insufficient_frame_quality(avctx, "bitrate cannot be determined.");

        goto erasure;

    }

    if (e->bitrate <= SILENCE || e->bitrate == RATE_QUARTER)

        goto erasure;

    if (e->bitrate == RATE_QUANT && e->last_valid_bitrate == RATE_FULL

                                 && !e->prev_error_flag)

        goto erasure;



    init_get_bits(&e->gb, buf, 8 * buf_size);

    memset(&e->frame, 0, sizeof(EVRCAFrame));



    unpack_frame(e);



    if (e->bitrate != RATE_QUANT) {

        uint8_t *p = (uint8_t *) &e->frame;

        for (i = 0; i < sizeof(EVRCAFrame); i++) {

            if (p[i])

                break;

        }

        if (i == sizeof(EVRCAFrame))

            goto erasure;

    } else if (e->frame.lsp[0] == 0xf &&

               e->frame.lsp[1] == 0xf &&

               e->frame.energy_gain == 0xff) {

        goto erasure;

    }



    if (decode_lspf(e) < 0)

        goto erasure;



    if (e->bitrate == RATE_FULL || e->bitrate == RATE_HALF) {

        /* Pitch delay parameter checking as per TIA/IS-127 5.1.5.1 */

        if (e->frame.pitch_delay > MAX_DELAY - MIN_DELAY)

            goto erasure;



        e->pitch_delay = e->frame.pitch_delay + MIN_DELAY;



        /* Delay diff parameter checking as per TIA/IS-127 5.1.5.2 */

        if (e->frame.delay_diff) {

            int p = e->pitch_delay - e->frame.delay_diff + 16;

            if (p < MIN_DELAY || p > MAX_DELAY)

                goto erasure;

        }



        /* Delay contour reconstruction as per TIA/IS-127 5.2.2.2 */

        if (e->frame.delay_diff &&

            e->bitrate == RATE_FULL && e->prev_error_flag) {

            float delay;



            memcpy(e->pitch, e->pitch_back, ACB_SIZE * sizeof(float));



            delay = e->prev_pitch_delay;

            e->prev_pitch_delay = delay - e->frame.delay_diff + 16.0;



            if (fabs(e->pitch_delay - delay) > 15)

                delay = e->pitch_delay;



            for (i = 0; i < NB_SUBFRAMES; i++) {

                int subframe_size = subframe_sizes[i];



                interpolate_delay(idelay, delay, e->prev_pitch_delay, i);

                acb_excitation(e, e->pitch + ACB_SIZE, e->avg_acb_gain, idelay, subframe_size);

                memcpy(e->pitch, e->pitch + subframe_size, ACB_SIZE * sizeof(float));

            }

        }



        /* Smoothing of the decoded delay as per TIA/IS-127 5.2.2.5 */

        if (fabs(e->pitch_delay - e->prev_pitch_delay) > 15)

            e->prev_pitch_delay = e->pitch_delay;



        e->avg_acb_gain = e->avg_fcb_gain = 0.0;

    } else {

        idelay[0] = idelay[1] = idelay[2] = MIN_DELAY;



        /* Decode frame energy vectors as per TIA/IS-127 5.7.2 */

        for (i = 0; i < NB_SUBFRAMES; i++)

            e->energy_vector[i] = pow(10, evrc_energy_quant[e->frame.energy_gain][i]);

        e->prev_energy_gain = e->frame.energy_gain;

    }



    for (i = 0; i < NB_SUBFRAMES; i++) {

        float tmp[SUBFRAME_SIZE + 6] = { 0 };

        int subframe_size = subframe_sizes[i];

        int pitch_lag;



        interpolate_lsp(ilspf, e->lspf, e->prev_lspf, i);



        if (e->bitrate != RATE_QUANT)

            interpolate_delay(idelay, e->pitch_delay, e->prev_pitch_delay, i);



        pitch_lag = lrintf((idelay[1] + idelay[0]) / 2.0);

        decode_predictor_coeffs(ilspf, ilpc);



        /* Bandwidth expansion as per TIA/IS-127 5.2.3.3 */

        if (e->frame.lpc_flag && e->prev_error_flag)

            bandwidth_expansion(ilpc, ilpc, 0.75);



        if (e->bitrate != RATE_QUANT) {

            float acb_sum, f;



            f = exp((e->bitrate == RATE_HALF ? 0.5 : 0.25)

                         * (e->frame.fcb_gain[i] + 1));

            acb_sum = pitch_gain_vq[e->frame.acb_gain[i]];

            e->avg_acb_gain += acb_sum / NB_SUBFRAMES;

            e->avg_fcb_gain += f / NB_SUBFRAMES;



            acb_excitation(e, e->pitch + ACB_SIZE,

                           acb_sum, idelay, subframe_size);

            fcb_excitation(e, e->frame.fcb_shape[i], tmp,

                           acb_sum, pitch_lag, subframe_size);



            /* Total excitation generation as per TIA/IS-127 5.2.3.9 */

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] += f * tmp[j];

            e->fade_scale = FFMIN(e->fade_scale + 0.2, 1.0);

        } else {

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] = e->energy_vector[i];

        }



        memcpy(e->pitch, e->pitch + subframe_size, ACB_SIZE * sizeof(float));



        synthesis_filter(e->pitch + ACB_SIZE, ilpc,

                         e->synthesis, subframe_size, tmp);

        postfilter(e, tmp, ilpc, samples, pitch_lag,

                   &postfilter_coeffs[e->bitrate], subframe_size);



        samples += subframe_size;

    }



    if (error_flag) {

erasure:

        error_flag = 1;

        av_log(avctx, AV_LOG_WARNING, "frame erasure\n");

        frame_erasure(e, samples);

    }



    memcpy(e->prev_lspf, e->lspf, sizeof(e->prev_lspf));

    e->prev_error_flag    = error_flag;

    e->last_valid_bitrate = e->bitrate;



    if (e->bitrate != RATE_QUANT)

        e->prev_pitch_delay = e->pitch_delay;



    samples = (float *)frame->data[0];

    for (i = 0; i < 160; i++)

        samples[i] /= 32768;



    *got_frame_ptr   = 1;



    return avpkt->size;

}
