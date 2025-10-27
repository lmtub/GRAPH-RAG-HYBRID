WINDOW_FUNC(eight_short)

{

    const float *swindow = sce->ics.use_kb_window[0] ? ff_aac_kbd_short_128 : ff_sine_128;

    const float *pwindow = sce->ics.use_kb_window[1] ? ff_aac_kbd_short_128 : ff_sine_128;

    const float *in = audio + 448;

    float *out = sce->ret;



    for (int w = 0; w < 8; w++) {

        dsp->vector_fmul        (out, in, w ? pwindow : swindow, 128);

        out += 128;

        in  += 128;

        dsp->vector_fmul_reverse(out, in, swindow, 128);

        out += 128;

    }

}
