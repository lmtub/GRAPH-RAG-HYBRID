static av_cold int ape_decode_init(AVCodecContext * avctx)

{

    APEContext *s = avctx->priv_data;

    int i;



    if (avctx->extradata_size != 6) {

        av_log(avctx, AV_LOG_ERROR, "Incorrect extradata\n");

        return -1;

    }

    if (avctx->bits_per_coded_sample != 16) {

        av_log(avctx, AV_LOG_ERROR, "Only 16-bit samples are supported\n");

        return -1;

    }

    if (avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "Only mono and stereo is supported\n");

        return -1;

    }

    s->avctx             = avctx;

    s->channels          = avctx->channels;

    s->fileversion       = AV_RL16(avctx->extradata);

    s->compression_level = AV_RL16(avctx->extradata + 2);

    s->flags             = AV_RL16(avctx->extradata + 4);



    av_log(avctx, AV_LOG_DEBUG, "Compression Level: %d - Flags: %d\n", s->compression_level, s->flags);

    if (s->compression_level % 1000 || s->compression_level > COMPRESSION_LEVEL_INSANE) {

        av_log(avctx, AV_LOG_ERROR, "Incorrect compression level %d\n", s->compression_level);

        return -1;

    }

    s->fset = s->compression_level / 1000 - 1;

    for (i = 0; i < APE_FILTER_LEVELS; i++) {

        if (!ape_filter_orders[s->fset][i])

            break;

        s->filterbuf[i] = av_malloc((ape_filter_orders[s->fset][i] * 3 + HISTORY_SIZE) * 4);

    }



    dsputil_init(&s->dsp, avctx);

    avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    avctx->channel_layout = (avctx->channels==2) ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;

    return 0;

}
