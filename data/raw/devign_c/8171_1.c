static av_cold int ac3_decode_init(AVCodecContext *avctx)

{

    AC3DecodeContext *s = avctx->priv_data;

    s->avctx = avctx;



    ff_ac3_common_init();

    ac3_tables_init();

    ff_mdct_init(&s->imdct_256, 8, 1, 1.0);

    ff_mdct_init(&s->imdct_512, 9, 1, 1.0);

    ff_kbd_window_init(s->window, 5.0, 256);

    dsputil_init(&s->dsp, avctx);

    ff_fmt_convert_init(&s->fmt_conv, avctx);

    av_lfg_init(&s->dith_state, 0);



    /* set scale value for float to int16 conversion */

    s->mul_bias = 32767.0f;



    /* allow downmixing to stereo or mono */

    if (avctx->channels > 0 && avctx->request_channels > 0 &&

            avctx->request_channels < avctx->channels &&

            avctx->request_channels <= 2) {

        avctx->channels = avctx->request_channels;

    }

    s->downmixed = 1;



    /* allocate context input buffer */

    if (avctx->error_recognition >= FF_ER_CAREFUL) {

        s->input_buffer = av_mallocz(AC3_FRAME_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!s->input_buffer)

            return AVERROR(ENOMEM);

    }



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    return 0;

}
