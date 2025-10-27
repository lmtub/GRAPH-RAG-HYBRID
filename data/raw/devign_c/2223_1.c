static av_cold int opus_decode_init(AVCodecContext *avctx)

{

    OpusContext *c = avctx->priv_data;

    int ret, i, j;



    avctx->sample_fmt  = AV_SAMPLE_FMT_FLTP;

    avctx->sample_rate = 48000;



    c->fdsp = avpriv_float_dsp_alloc(0);

    if (!c->fdsp)

        return AVERROR(ENOMEM);



    /* find out the channel configuration */

    ret = ff_opus_parse_extradata(avctx, c);

    if (ret < 0)

        return ret;



    /* allocate and init each independent decoder */

    c->streams = av_mallocz_array(c->nb_streams, sizeof(*c->streams));

    c->out             = av_mallocz_array(c->nb_streams, 2 * sizeof(*c->out));

    c->out_size        = av_mallocz_array(c->nb_streams, sizeof(*c->out_size));

    c->sync_buffers    = av_mallocz_array(c->nb_streams, sizeof(*c->sync_buffers));

    c->decoded_samples = av_mallocz_array(c->nb_streams, sizeof(*c->decoded_samples));

    if (!c->streams || !c->sync_buffers || !c->decoded_samples || !c->out || !c->out_size) {

        c->nb_streams = 0;

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    for (i = 0; i < c->nb_streams; i++) {

        OpusStreamContext *s = &c->streams[i];

        uint64_t layout;



        s->output_channels = (i < c->nb_stereo_streams) ? 2 : 1;



        s->avctx = avctx;



        for (j = 0; j < s->output_channels; j++) {

            s->silk_output[j]       = s->silk_buf[j];

            s->celt_output[j]       = s->celt_buf[j];

            s->redundancy_output[j] = s->redundancy_buf[j];

        }



        s->fdsp = c->fdsp;



        s->swr =swr_alloc();

        if (!s->swr)

            goto fail;



        layout = (s->output_channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

        av_opt_set_int(s->swr, "in_sample_fmt",      avctx->sample_fmt,  0);

        av_opt_set_int(s->swr, "out_sample_fmt",     avctx->sample_fmt,  0);

        av_opt_set_int(s->swr, "in_channel_layout",  layout,             0);

        av_opt_set_int(s->swr, "out_channel_layout", layout,             0);

        av_opt_set_int(s->swr, "out_sample_rate",    avctx->sample_rate, 0);

        av_opt_set_int(s->swr, "filter_size",        16,                 0);



        ret = ff_silk_init(avctx, &s->silk, s->output_channels);

        if (ret < 0)

            goto fail;



        ret = ff_celt_init(avctx, &s->celt, s->output_channels);

        if (ret < 0)

            goto fail;



        s->celt_delay = av_audio_fifo_alloc(avctx->sample_fmt,

                                            s->output_channels, 1024);

        if (!s->celt_delay) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        c->sync_buffers[i] = av_audio_fifo_alloc(avctx->sample_fmt,

                                                 s->output_channels, 32);

        if (!c->sync_buffers[i]) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

    }



    return 0;

fail:

    opus_decode_close(avctx);

    return ret;

}
