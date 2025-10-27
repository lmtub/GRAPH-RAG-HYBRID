static void frame_erasure(EVRCContext *e, float *samples)

{

    float ilspf[FILTER_ORDER], ilpc[FILTER_ORDER], idelay[NB_SUBFRAMES],

          tmp[SUBFRAME_SIZE + 6], f;

    int i, j;



    for (i = 0; i < FILTER_ORDER; i++) {

        if (e->bitrate != RATE_QUANT)

            e->lspf[i] = e->prev_lspf[i] * 0.875 + 0.125 * (i + 1) * 0.048;

        else

            e->lspf[i] = e->prev_lspf[i];

    }



    if (e->prev_error_flag)

        e->avg_acb_gain *= 0.75;

    if (e->bitrate == RATE_FULL)

        memcpy(e->pitch_back, e->pitch, ACB_SIZE * sizeof(float));

    if (e->last_valid_bitrate == RATE_QUANT)

        e->bitrate = RATE_QUANT;

    else

        e->bitrate = RATE_FULL;



    if (e->bitrate == RATE_FULL || e->bitrate == RATE_HALF) {

        e->pitch_delay = e->prev_pitch_delay;

    } else {

        float sum = 0;



        idelay[0] = idelay[1] = idelay[2] = MIN_DELAY;



        for (i = 0; i < NB_SUBFRAMES; i++)

            sum += evrc_energy_quant[e->prev_energy_gain][i];

        sum /= (float) NB_SUBFRAMES;

        sum  = pow(10, sum);

        for (i = 0; i < NB_SUBFRAMES; i++)

            e->energy_vector[i] = sum;

    }



    if (fabs(e->pitch_delay - e->prev_pitch_delay) > 15)

        e->prev_pitch_delay = e->pitch_delay;



    for (i = 0; i < NB_SUBFRAMES; i++) {

        int subframe_size = subframe_sizes[i];

        int pitch_lag;



        interpolate_lsp(ilspf, e->lspf, e->prev_lspf, i);



        if (e->bitrate != RATE_QUANT) {

            if (e->avg_acb_gain < 0.3) {

                idelay[0] = estimation_delay[i];

                idelay[1] = estimation_delay[i + 1];

                idelay[2] = estimation_delay[i + 2];

            } else {

                interpolate_delay(idelay, e->pitch_delay, e->prev_pitch_delay, i);

            }

        }



        pitch_lag = lrintf((idelay[1] + idelay[0]) / 2.0);

        decode_predictor_coeffs(ilspf, ilpc);



        if (e->bitrate != RATE_QUANT) {

            acb_excitation(e, e->pitch + ACB_SIZE,

                           e->avg_acb_gain, idelay, subframe_size);

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] *= e->fade_scale;

            e->fade_scale = FFMAX(e->fade_scale - 0.05, 0.0);

        } else {

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] = e->energy_vector[i];

        }



        memcpy(e->pitch, e->pitch + subframe_size, ACB_SIZE * sizeof(float));



        if (e->bitrate != RATE_QUANT && e->avg_acb_gain < 0.4) {

            f = 0.1 * e->avg_fcb_gain;

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] += f;

        } else if (e->bitrate == RATE_QUANT) {

            for (j = 0; j < subframe_size; j++)

                e->pitch[ACB_SIZE + j] = e->energy_vector[i];

        }



        synthesis_filter(e->pitch + ACB_SIZE, ilpc,

                         e->synthesis, subframe_size, tmp);

        postfilter(e, tmp, ilpc, samples, pitch_lag,

                   &postfilter_coeffs[e->bitrate], subframe_size);



        samples += subframe_size;

    }

}
