int ff_audio_mix_init(AVAudioResampleContext *avr)

{

    int ret;



    if (avr->internal_sample_fmt != AV_SAMPLE_FMT_S16P &&

        avr->internal_sample_fmt != AV_SAMPLE_FMT_FLTP) {

        av_log(avr, AV_LOG_ERROR, "Unsupported internal format for "

               "mixing: %s\n",

               av_get_sample_fmt_name(avr->internal_sample_fmt));

        return AVERROR(EINVAL);

    }



    /* build matrix if the user did not already set one */

    if (!avr->am->matrix) {

        int i, j;

        char in_layout_name[128];

        char out_layout_name[128];

        double *matrix_dbl = av_mallocz(avr->out_channels * avr->in_channels *

                                        sizeof(*matrix_dbl));

        if (!matrix_dbl)

            return AVERROR(ENOMEM);



        ret = avresample_build_matrix(avr->in_channel_layout,

                                      avr->out_channel_layout,

                                      avr->center_mix_level,

                                      avr->surround_mix_level,

                                      avr->lfe_mix_level, 1, matrix_dbl,

                                      avr->in_channels,

                                      avr->matrix_encoding);

        if (ret < 0) {

            av_free(matrix_dbl);

            return ret;

        }



        av_get_channel_layout_string(in_layout_name, sizeof(in_layout_name),

                                     avr->in_channels, avr->in_channel_layout);

        av_get_channel_layout_string(out_layout_name, sizeof(out_layout_name),

                                     avr->out_channels, avr->out_channel_layout);

        av_log(avr, AV_LOG_DEBUG, "audio_mix: %s to %s\n",

               in_layout_name, out_layout_name);

        for (i = 0; i < avr->out_channels; i++) {

            for (j = 0; j < avr->in_channels; j++) {

                av_log(avr, AV_LOG_DEBUG, "  %0.3f ",

                       matrix_dbl[i * avr->in_channels + j]);

            }

            av_log(avr, AV_LOG_DEBUG, "\n");

        }



        ret = avresample_set_matrix(avr, matrix_dbl, avr->in_channels);

        if (ret < 0) {

            av_free(matrix_dbl);

            return ret;

        }

        av_free(matrix_dbl);

    }



    avr->am->fmt          = avr->internal_sample_fmt;

    avr->am->coeff_type   = avr->mix_coeff_type;

    avr->am->in_layout    = avr->in_channel_layout;

    avr->am->out_layout   = avr->out_channel_layout;

    avr->am->in_channels  = avr->in_channels;

    avr->am->out_channels = avr->out_channels;



    ret = mix_function_init(avr->am);

    if (ret < 0)

        return ret;



    return 0;

}
