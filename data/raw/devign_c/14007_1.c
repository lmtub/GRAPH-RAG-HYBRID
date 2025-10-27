ReSampleContext *av_audio_resample_init(int output_channels, int input_channels,

                                        int output_rate, int input_rate,

                                        enum AVSampleFormat sample_fmt_out,

                                        enum AVSampleFormat sample_fmt_in,

                                        int filter_length, int log2_phase_count,

                                        int linear, double cutoff)

{

    ReSampleContext *s;



    if (input_channels > MAX_CHANNELS) {

        av_log(NULL, AV_LOG_ERROR,

               "Resampling with input channels greater than %d is unsupported.\n",

               MAX_CHANNELS);

        return NULL;

    }

    if (output_channels > 2 &&

        !(output_channels == 6 && input_channels == 2) &&

        output_channels != input_channels) {

        av_log(NULL, AV_LOG_ERROR,

               "Resampling output channel count must be 1 or 2 for mono input; 1, 2 or 6 for stereo input; or N for N channel input.\n");

        return NULL;

    }



    s = av_mallocz(sizeof(ReSampleContext));

    if (!s) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate memory for resample context.\n");

        return NULL;

    }



    s->ratio = (float)output_rate / (float)input_rate;



    s->input_channels = input_channels;

    s->output_channels = output_channels;



    s->filter_channels = s->input_channels;

    if (s->output_channels < s->filter_channels)

        s->filter_channels = s->output_channels;



    s->sample_fmt[0]  = sample_fmt_in;

    s->sample_fmt[1]  = sample_fmt_out;

    s->sample_size[0] = av_get_bits_per_sample_fmt(s->sample_fmt[0]) >> 3;

    s->sample_size[1] = av_get_bits_per_sample_fmt(s->sample_fmt[1]) >> 3;



    if (s->sample_fmt[0] != AV_SAMPLE_FMT_S16) {

        if (!(s->convert_ctx[0] = av_audio_convert_alloc(AV_SAMPLE_FMT_S16, 1,

                                                         s->sample_fmt[0], 1, NULL, 0))) {

            av_log(s, AV_LOG_ERROR,

                   "Cannot convert %s sample format to s16 sample format\n",

                   av_get_sample_fmt_name(s->sample_fmt[0]));

            av_free(s);

            return NULL;

        }

    }



    if (s->sample_fmt[1] != AV_SAMPLE_FMT_S16) {

        if (!(s->convert_ctx[1] = av_audio_convert_alloc(s->sample_fmt[1], 1,

                                                         AV_SAMPLE_FMT_S16, 1, NULL, 0))) {

            av_log(s, AV_LOG_ERROR,

                   "Cannot convert s16 sample format to %s sample format\n",

                   av_get_sample_fmt_name(s->sample_fmt[1]));

            av_audio_convert_free(s->convert_ctx[0]);

            av_free(s);

            return NULL;

        }

    }



#define TAPS 16

    s->resample_context = av_resample_init(output_rate, input_rate,

                                           filter_length, log2_phase_count,

                                           linear, cutoff);



    *(const AVClass**)s->resample_context = &audioresample_context_class;



    return s;

}
