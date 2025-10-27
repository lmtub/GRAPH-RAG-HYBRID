static int apply_window_and_mdct(vorbis_enc_context *venc,

                                 float **audio, int samples)

{

    int channel;

    const float * win = venc->win[0];

    int window_len = 1 << (venc->log2_blocksize[0] - 1);

    float n = (float)(1 << venc->log2_blocksize[0]) / 4.0;

    AVFloatDSPContext *fdsp = venc->fdsp;



    if (!venc->have_saved && !samples)

        return 0;



    if (venc->have_saved) {

        for (channel = 0; channel < venc->channels; channel++)

            memcpy(venc->samples + channel * window_len * 2,

                   venc->saved + channel * window_len, sizeof(float) * window_len);

    } else {

        for (channel = 0; channel < venc->channels; channel++)

            memset(venc->samples + channel * window_len * 2, 0,

                   sizeof(float) * window_len);

    }



    if (samples) {

        for (channel = 0; channel < venc->channels; channel++) {

            float *offset = venc->samples + channel * window_len * 2 + window_len;



            fdsp->vector_fmul_reverse(offset, audio[channel], win, samples);

            fdsp->vector_fmul_scalar(offset, offset, 1/n, samples);

        }

    } else {

        for (channel = 0; channel < venc->channels; channel++)

            memset(venc->samples + channel * window_len * 2 + window_len,

                   0, sizeof(float) * window_len);

    }



    for (channel = 0; channel < venc->channels; channel++)

        venc->mdct[0].mdct_calc(&venc->mdct[0], venc->coeffs + channel * window_len,

                     venc->samples + channel * window_len * 2);



    if (samples) {

        for (channel = 0; channel < venc->channels; channel++) {

            float *offset = venc->saved + channel * window_len;



            fdsp->vector_fmul(offset, audio[channel], win, samples);

            fdsp->vector_fmul_scalar(offset, offset, 1/n, samples);

        }

        venc->have_saved = 1;

    } else {

        venc->have_saved = 0;

    }

    return 1;

}
