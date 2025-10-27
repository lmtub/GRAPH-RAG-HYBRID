ResampleContext *ff_audio_resample_init(AVAudioResampleContext *avr)

{

    ResampleContext *c;

    int out_rate    = avr->out_sample_rate;

    int in_rate     = avr->in_sample_rate;

    double factor   = FFMIN(out_rate * avr->cutoff / in_rate, 1.0);

    int phase_count = 1 << avr->phase_shift;

    int felem_size;



    if (avr->internal_sample_fmt != AV_SAMPLE_FMT_S16P &&

        avr->internal_sample_fmt != AV_SAMPLE_FMT_S32P &&

        avr->internal_sample_fmt != AV_SAMPLE_FMT_FLTP &&

        avr->internal_sample_fmt != AV_SAMPLE_FMT_DBLP) {

        av_log(avr, AV_LOG_ERROR, "Unsupported internal format for "

               "resampling: %s\n",

               av_get_sample_fmt_name(avr->internal_sample_fmt));

        return NULL;

    }

    c = av_mallocz(sizeof(*c));

    if (!c)

        return NULL;



    c->avr           = avr;

    c->phase_shift   = avr->phase_shift;

    c->phase_mask    = phase_count - 1;

    c->linear        = avr->linear_interp;

    c->factor        = factor;

    c->filter_length = FFMAX((int)ceil(avr->filter_size / factor), 1);

    c->filter_type   = avr->filter_type;

    c->kaiser_beta   = avr->kaiser_beta;



    switch (avr->internal_sample_fmt) {

    case AV_SAMPLE_FMT_DBLP:

        c->resample_one  = resample_one_dbl;

        c->resample_nearest = resample_nearest_dbl;

        c->set_filter    = set_filter_dbl;

        break;

    case AV_SAMPLE_FMT_FLTP:

        c->resample_one  = resample_one_flt;

        c->resample_nearest = resample_nearest_flt;

        c->set_filter    = set_filter_flt;

        break;

    case AV_SAMPLE_FMT_S32P:

        c->resample_one  = resample_one_s32;

        c->resample_nearest = resample_nearest_s32;

        c->set_filter    = set_filter_s32;

        break;

    case AV_SAMPLE_FMT_S16P:

        c->resample_one  = resample_one_s16;

        c->resample_nearest = resample_nearest_s16;

        c->set_filter    = set_filter_s16;

        break;

    }



    felem_size = av_get_bytes_per_sample(avr->internal_sample_fmt);

    c->filter_bank = av_mallocz(c->filter_length * (phase_count + 1) * felem_size);

    if (!c->filter_bank)

        goto error;



    if (build_filter(c) < 0)

        goto error;



    memcpy(&c->filter_bank[(c->filter_length * phase_count + 1) * felem_size],

           c->filter_bank, (c->filter_length - 1) * felem_size);

    memcpy(&c->filter_bank[c->filter_length * phase_count * felem_size],

           &c->filter_bank[(c->filter_length - 1) * felem_size], felem_size);



    c->compensation_distance = 0;

    if (!av_reduce(&c->src_incr, &c->dst_incr, out_rate,

                   in_rate * (int64_t)phase_count, INT32_MAX / 2))

        goto error;

    c->ideal_dst_incr = c->dst_incr;



    c->padding_size   = (c->filter_length - 1) / 2;

    c->index = -phase_count * ((c->filter_length - 1) / 2);

    c->frac  = 0;



    /* allocate internal buffer */

    c->buffer = ff_audio_data_alloc(avr->resample_channels, 0,

                                    avr->internal_sample_fmt,

                                    "resample buffer");

    if (!c->buffer)

        goto error;



    av_log(avr, AV_LOG_DEBUG, "resample: %s from %d Hz to %d Hz\n",

           av_get_sample_fmt_name(avr->internal_sample_fmt),

           avr->in_sample_rate, avr->out_sample_rate);



    return c;



error:

    ff_audio_data_free(&c->buffer);

    av_free(c->filter_bank);

    av_free(c);

    return NULL;

}
