int audio_resample(ReSampleContext *s, short *output, short *input, int nb_samples)

{

    int i, nb_samples1;

    short *bufin[2];

    short *bufout[2];

    short *buftmp2[2], *buftmp3[2];

    int lenout;



    if (s->input_channels == s->output_channels && s->ratio == 1.0 && 0) {

        /* nothing to do */

        memcpy(output, input, nb_samples * s->input_channels * sizeof(short));

        return nb_samples;

    }



    /* XXX: move those malloc to resample init code */

    for(i=0; i<s->filter_channels; i++){

        bufin[i]= (short*) av_malloc( (nb_samples + s->temp_len) * sizeof(short) );

        memcpy(bufin[i], s->temp[i], s->temp_len * sizeof(short));

        buftmp2[i] = bufin[i] + s->temp_len;

    }



    /* make some zoom to avoid round pb */

    lenout= (int)(4*nb_samples * s->ratio) + 16;

    bufout[0]= (short*) av_malloc( lenout * sizeof(short) );

    bufout[1]= (short*) av_malloc( lenout * sizeof(short) );



    if (s->input_channels == 2 &&

        s->output_channels == 1) {

        buftmp3[0] = output;

        stereo_to_mono(buftmp2[0], input, nb_samples);

    } else if (s->output_channels >= 2 && s->input_channels == 1) {

        buftmp3[0] = bufout[0];

        memcpy(buftmp2[0], input, nb_samples*sizeof(short));

    } else if (s->output_channels >= 2) {

        buftmp3[0] = bufout[0];

        buftmp3[1] = bufout[1];

        stereo_split(buftmp2[0], buftmp2[1], input, nb_samples);

    } else {

        buftmp3[0] = output;

        memcpy(buftmp2[0], input, nb_samples*sizeof(short));

    }



    nb_samples += s->temp_len;



    /* resample each channel */

    nb_samples1 = 0; /* avoid warning */

    for(i=0;i<s->filter_channels;i++) {

        int consumed;

        int is_last= i+1 == s->filter_channels;



        nb_samples1 = av_resample(s->resample_context, buftmp3[i], bufin[i], &consumed, nb_samples, lenout, is_last);

        s->temp_len= nb_samples - consumed;

        s->temp[i]= av_realloc(s->temp[i], s->temp_len*sizeof(short));

        memcpy(s->temp[i], bufin[i] + consumed, s->temp_len*sizeof(short));

    }



    if (s->output_channels == 2 && s->input_channels == 1) {

        mono_to_stereo(output, buftmp3[0], nb_samples1);

    } else if (s->output_channels == 2) {

        stereo_mux(output, buftmp3[0], buftmp3[1], nb_samples1);

    } else if (s->output_channels == 6) {

        ac3_5p1_mux(output, buftmp3[0], buftmp3[1], nb_samples1);

    }



    for(i=0; i<s->filter_channels; i++)

        av_free(bufin[i]);



    av_free(bufout[0]);

    av_free(bufout[1]);

    return nb_samples1;

}
