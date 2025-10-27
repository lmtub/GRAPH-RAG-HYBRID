static av_cold int sonic_encode_init(AVCodecContext *avctx)

{

    SonicContext *s = avctx->priv_data;

    PutBitContext pb;

    int i, version = 0;



    if (avctx->channels > MAX_CHANNELS)

    {

        av_log(avctx, AV_LOG_ERROR, "Only mono and stereo streams are supported by now\n");

        return AVERROR(EINVAL); /* only stereo or mono for now */

    }



    if (avctx->channels == 2)

        s->decorrelation = MID_SIDE;

    else

        s->decorrelation = 3;



    if (avctx->codec->id == AV_CODEC_ID_SONIC_LS)

    {

        s->lossless = 1;

        s->num_taps = 32;

        s->downsampling = 1;

        s->quantization = 0.0;

    }

    else

    {

        s->num_taps = 128;

        s->downsampling = 2;

        s->quantization = 1.0;

    }



    // max tap 2048

    if ((s->num_taps < 32) || (s->num_taps > 1024) ||

        ((s->num_taps>>5)<<5 != s->num_taps))

    {

        av_log(avctx, AV_LOG_ERROR, "Invalid number of taps\n");

        return AVERROR_INVALIDDATA;

    }



    // generate taps

    s->tap_quant = av_calloc(s->num_taps, sizeof(*s->tap_quant));

    for (i = 0; i < s->num_taps; i++)

        s->tap_quant[i] = ff_sqrt(i+1);



    s->channels = avctx->channels;

    s->samplerate = avctx->sample_rate;



    s->block_align = 2048LL*s->samplerate/(44100*s->downsampling);

    s->frame_size = s->channels*s->block_align*s->downsampling;



    s->tail_size = s->num_taps*s->channels;

    s->tail = av_calloc(s->tail_size, sizeof(*s->tail));

    if (!s->tail)

        return AVERROR(ENOMEM);



    s->predictor_k = av_calloc(s->num_taps, sizeof(*s->predictor_k) );

    if (!s->predictor_k)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->channels; i++)

    {

        s->coded_samples[i] = av_calloc(s->block_align, sizeof(**s->coded_samples));

        if (!s->coded_samples[i])

            return AVERROR(ENOMEM);

    }



    s->int_samples = av_calloc(s->frame_size, sizeof(*s->int_samples));



    s->window_size = ((2*s->tail_size)+s->frame_size);

    s->window = av_calloc(s->window_size, sizeof(*s->window));

    if (!s->window)

        return AVERROR(ENOMEM);



    avctx->extradata = av_mallocz(16);

    if (!avctx->extradata)

        return AVERROR(ENOMEM);

    init_put_bits(&pb, avctx->extradata, 16*8);



    put_bits(&pb, 2, version); // version

    if (version == 1)

    {

        put_bits(&pb, 2, s->channels);

        put_bits(&pb, 4, code_samplerate(s->samplerate));

    }

    put_bits(&pb, 1, s->lossless);

    if (!s->lossless)

        put_bits(&pb, 3, SAMPLE_SHIFT); // XXX FIXME: sample precision

    put_bits(&pb, 2, s->decorrelation);

    put_bits(&pb, 2, s->downsampling);

    put_bits(&pb, 5, (s->num_taps >> 5)-1); // 32..1024

    put_bits(&pb, 1, 0); // XXX FIXME: no custom tap quant table



    flush_put_bits(&pb);

    avctx->extradata_size = put_bits_count(&pb)/8;



    av_log(avctx, AV_LOG_INFO, "Sonic: ver: %d ls: %d dr: %d taps: %d block: %d frame: %d downsamp: %d\n",

        version, s->lossless, s->decorrelation, s->num_taps, s->block_align, s->frame_size, s->downsampling);



    avctx->frame_size = s->block_align*s->downsampling;



    return 0;

}
