static av_cold int sonic_decode_init(AVCodecContext *avctx)

{

    SonicContext *s = avctx->priv_data;

    GetBitContext gb;

    int i;



    s->channels = avctx->channels;

    s->samplerate = avctx->sample_rate;



    if (!avctx->extradata)

    {

        av_log(avctx, AV_LOG_ERROR, "No mandatory headers present\n");

        return AVERROR_INVALIDDATA;

    }



    init_get_bits8(&gb, avctx->extradata, avctx->extradata_size);



    s->version = get_bits(&gb, 2);

    if (s->version >= 2) {

        s->version       = get_bits(&gb, 8);

        s->minor_version = get_bits(&gb, 8);

    }

    if (s->version != 2)

    {

        av_log(avctx, AV_LOG_ERROR, "Unsupported Sonic version, please report\n");

        return AVERROR_INVALIDDATA;

    }



    if (s->version >= 1)

    {

        int sample_rate_index;

        s->channels = get_bits(&gb, 2);

        sample_rate_index = get_bits(&gb, 4);

        if (sample_rate_index >= FF_ARRAY_ELEMS(samplerate_table)) {

            av_log(avctx, AV_LOG_ERROR, "Invalid sample_rate_index %d\n", sample_rate_index);

            return AVERROR_INVALIDDATA;

        }

        s->samplerate = samplerate_table[sample_rate_index];

        av_log(avctx, AV_LOG_INFO, "Sonicv2 chans: %d samprate: %d\n",

            s->channels, s->samplerate);

    }



    if (s->channels > MAX_CHANNELS || s->channels < 1)

    {

        av_log(avctx, AV_LOG_ERROR, "Only mono and stereo streams are supported by now\n");

        return AVERROR_INVALIDDATA;

    }




    s->lossless = get_bits1(&gb);

    if (!s->lossless)

        skip_bits(&gb, 3); // XXX FIXME

    s->decorrelation = get_bits(&gb, 2);

    if (s->decorrelation != 3 && s->channels != 2) {

        av_log(avctx, AV_LOG_ERROR, "invalid decorrelation %d\n", s->decorrelation);

        return AVERROR_INVALIDDATA;

    }



    s->downsampling = get_bits(&gb, 2);

    if (!s->downsampling) {

        av_log(avctx, AV_LOG_ERROR, "invalid downsampling value\n");

        return AVERROR_INVALIDDATA;

    }



    s->num_taps = (get_bits(&gb, 5)+1)<<5;

    if (get_bits1(&gb)) // XXX FIXME

        av_log(avctx, AV_LOG_INFO, "Custom quant table\n");



    s->block_align = 2048LL*s->samplerate/(44100*s->downsampling);

    s->frame_size = s->channels*s->block_align*s->downsampling;

//    avctx->frame_size = s->block_align;



    av_log(avctx, AV_LOG_INFO, "Sonic: ver: %d.%d ls: %d dr: %d taps: %d block: %d frame: %d downsamp: %d\n",

        s->version, s->minor_version, s->lossless, s->decorrelation, s->num_taps, s->block_align, s->frame_size, s->downsampling);



    // generate taps

    s->tap_quant = av_calloc(s->num_taps, sizeof(*s->tap_quant));

    if (!s->tap_quant)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->num_taps; i++)

        s->tap_quant[i] = ff_sqrt(i+1);



    s->predictor_k = av_calloc(s->num_taps, sizeof(*s->predictor_k));



    for (i = 0; i < s->channels; i++)

    {

        s->predictor_state[i] = av_calloc(s->num_taps, sizeof(**s->predictor_state));

        if (!s->predictor_state[i])

            return AVERROR(ENOMEM);

    }



    for (i = 0; i < s->channels; i++)

    {

        s->coded_samples[i] = av_calloc(s->block_align, sizeof(**s->coded_samples));

        if (!s->coded_samples[i])

            return AVERROR(ENOMEM);

    }

    s->int_samples = av_calloc(s->frame_size, sizeof(*s->int_samples));

    if (!s->int_samples)

        return AVERROR(ENOMEM);



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    return 0;

}