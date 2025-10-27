static av_cold int dsp_init(AVCodecContext *avctx, AACEncContext *s)

{

    int ret = 0;



    s->fdsp = avpriv_float_dsp_alloc(avctx->flags & CODEC_FLAG_BITEXACT);

    if (!s->fdsp)

        return AVERROR(ENOMEM);



    // window init

    ff_kbd_window_init(ff_aac_kbd_long_1024, 4.0, 1024);

    ff_kbd_window_init(ff_aac_kbd_short_128, 6.0, 128);

    ff_init_ff_sine_windows(10);

    ff_init_ff_sine_windows(7);



    if (ret = ff_mdct_init(&s->mdct1024, 11, 0, 32768.0))

        return ret;

    if (ret = ff_mdct_init(&s->mdct128,   8, 0, 32768.0))

        return ret;



    return 0;

}
