static av_cold int validate_options(AVCodecContext *avctx, AC3EncodeContext *s)

{

    int i, j;



    if (!avctx->channel_layout) {

        av_log(avctx, AV_LOG_WARNING, "No channel layout specified. The "

                                      "encoder will guess the layout, but it "

                                      "might be incorrect.\n");

    }

    if (set_channel_info(s, avctx->channels, &avctx->channel_layout)) {

        av_log(avctx, AV_LOG_ERROR, "invalid channel layout\n");

        return -1;

    }



    /* frequency */

    for (i = 0; i < 3; i++) {

        for (j = 0; j < 3; j++)

            if ((ff_ac3_sample_rate_tab[j] >> i) == avctx->sample_rate)

                goto found;

    }

    return -1;

 found:

    s->sample_rate        = avctx->sample_rate;

    s->bit_alloc.sr_shift = i;

    s->bit_alloc.sr_code  = j;

    s->bitstream_id       = 8 + s->bit_alloc.sr_shift;

    s->bitstream_mode     = 0; /* complete main audio service */



    /* bitrate & frame size */

    for (i = 0; i < 19; i++) {

        if ((ff_ac3_bitrate_tab[i] >> s->bit_alloc.sr_shift)*1000 == avctx->bit_rate)

            break;

    }

    if (i == 19)

        return -1;

    s->bit_rate        = avctx->bit_rate;

    s->frame_size_code = i << 1;



    return 0;

}
