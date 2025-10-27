static void windowing_and_mdct_ltp(AACContext *ac, float *out,

                                   float *in, IndividualChannelStream *ics)

{

    const float *lwindow      = ics->use_kb_window[0] ? ff_aac_kbd_long_1024 : ff_sine_1024;

    const float *swindow      = ics->use_kb_window[0] ? ff_aac_kbd_short_128 : ff_sine_128;

    const float *lwindow_prev = ics->use_kb_window[1] ? ff_aac_kbd_long_1024 : ff_sine_1024;

    const float *swindow_prev = ics->use_kb_window[1] ? ff_aac_kbd_short_128 : ff_sine_128;



    if (ics->window_sequence[0] != LONG_STOP_SEQUENCE) {

        ac->dsp.vector_fmul(in, in, lwindow_prev, 1024);

    } else {

        memset(in, 0, 448 * sizeof(float));

        ac->dsp.vector_fmul(in + 448, in + 448, swindow_prev, 128);

        memcpy(in + 576, in + 576, 448 * sizeof(float));

    }

    if (ics->window_sequence[0] != LONG_START_SEQUENCE) {

        ac->dsp.vector_fmul_reverse(in + 1024, in + 1024, lwindow, 1024);

    } else {

        memcpy(in + 1024, in + 1024, 448 * sizeof(float));

        ac->dsp.vector_fmul_reverse(in + 1024 + 448, in + 1024 + 448, swindow, 128);

        memset(in + 1024 + 576, 0, 448 * sizeof(float));

    }

    ac->mdct_ltp.mdct_calc(&ac->mdct_ltp, out, in);

}
