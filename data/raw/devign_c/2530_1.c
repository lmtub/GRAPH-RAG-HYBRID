int ff_audio_mix_set_matrix(AudioMix *am, const double *matrix, int stride)

{

    int i, o, i0, o0, ret;

    char in_layout_name[128];

    char out_layout_name[128];



    if ( am->in_channels <= 0 ||  am->in_channels > AVRESAMPLE_MAX_CHANNELS ||

        am->out_channels <= 0 || am->out_channels > AVRESAMPLE_MAX_CHANNELS) {

        av_log(am->avr, AV_LOG_ERROR, "Invalid channel counts\n");

        return AVERROR(EINVAL);

    }



    if (am->matrix) {

        av_free(am->matrix[0]);

        am->matrix = NULL;

    }



    am->in_matrix_channels  = am->in_channels;

    am->out_matrix_channels = am->out_channels;



    reduce_matrix(am, matrix, stride);



#define CONVERT_MATRIX(type, expr)                                          \

    am->matrix_## type[0] = av_mallocz(am->out_matrix_channels *            \

                                       am->in_matrix_channels  *            \

                                       sizeof(*am->matrix_## type[0]));     \

    if (!am->matrix_## type[0])                                             \

        return AVERROR(ENOMEM);                                             \

    for (o = 0, o0 = 0; o < am->out_channels; o++) {                        \

        if (am->output_zero[o] || am->output_skip[o])                       \

            continue;                                                       \

        if (o0 > 0)                                                         \

            am->matrix_## type[o0] = am->matrix_## type[o0 - 1] +           \

                                     am->in_matrix_channels;                \

        for (i = 0, i0 = 0; i < am->in_channels; i++) {                     \

            double v;                                                       \

            if (am->input_skip[i])                                          \

                continue;                                                   \

            v = matrix[o * stride + i];                                     \

            am->matrix_## type[o0][i0] = expr;                              \

            i0++;                                                           \

        }                                                                   \

        o0++;                                                               \

    }                                                                       \

    am->matrix = (void **)am->matrix_## type;



    if (am->in_matrix_channels && am->out_matrix_channels) {

        switch (am->coeff_type) {

        case AV_MIX_COEFF_TYPE_Q8:

            CONVERT_MATRIX(q8, av_clip_int16(lrint(256.0 * v)))

            break;

        case AV_MIX_COEFF_TYPE_Q15:

            CONVERT_MATRIX(q15, av_clipl_int32(llrint(32768.0 * v)))

            break;

        case AV_MIX_COEFF_TYPE_FLT:

            CONVERT_MATRIX(flt, v)

            break;

        default:

            av_log(am->avr, AV_LOG_ERROR, "Invalid mix coeff type\n");

            return AVERROR(EINVAL);

        }

    }



    ret = mix_function_init(am);

    if (ret < 0)

        return ret;



    av_get_channel_layout_string(in_layout_name, sizeof(in_layout_name),

                                 am->in_channels, am->in_layout);

    av_get_channel_layout_string(out_layout_name, sizeof(out_layout_name),

                                 am->out_channels, am->out_layout);

    av_log(am->avr, AV_LOG_DEBUG, "audio_mix: %s to %s\n",

           in_layout_name, out_layout_name);

    av_log(am->avr, AV_LOG_DEBUG, "matrix size: %d x %d\n",

           am->in_matrix_channels, am->out_matrix_channels);

    for (o = 0; o < am->out_channels; o++) {

        for (i = 0; i < am->in_channels; i++) {

            if (am->output_zero[o])

                av_log(am->avr, AV_LOG_DEBUG, "  (ZERO)");

            else if (am->input_skip[i] || am->output_skip[o])

                av_log(am->avr, AV_LOG_DEBUG, "  (SKIP)");

            else

                av_log(am->avr, AV_LOG_DEBUG, "  %0.3f ",

                       matrix[o * am->in_channels + i]);

        }

        av_log(am->avr, AV_LOG_DEBUG, "\n");

    }



    return 0;

}
