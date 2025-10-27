void swri_get_dither(SwrContext *s, void *dst, int len, unsigned seed, enum AVSampleFormat noise_fmt) {

    double scale = s->dither.noise_scale;

#define TMP_EXTRA 2

    double *tmp = av_malloc_array(len + TMP_EXTRA, sizeof(double));

    int i;



    for(i=0; i<len + TMP_EXTRA; i++){

        double v;

        seed = seed* 1664525 + 1013904223;



        switch(s->dither.method){

            case SWR_DITHER_RECTANGULAR: v= ((double)seed) / UINT_MAX - 0.5; break;

            default:

                av_assert0(s->dither.method < SWR_DITHER_NB);

                v = ((double)seed) / UINT_MAX;

                seed = seed*1664525 + 1013904223;

                v-= ((double)seed) / UINT_MAX;

                break;

        }

        tmp[i] = v;

    }



    for(i=0; i<len; i++){

        double v;



        switch(s->dither.method){

            default:

                av_assert0(s->dither.method < SWR_DITHER_NB);

                v = tmp[i];

                break;

            case SWR_DITHER_TRIANGULAR_HIGHPASS :

                v = (- tmp[i] + 2*tmp[i+1] - tmp[i+2]) / sqrt(6);

                break;

        }



        v*= scale;



        switch(noise_fmt){

            case AV_SAMPLE_FMT_S16P: ((int16_t*)dst)[i] = v; break;

            case AV_SAMPLE_FMT_S32P: ((int32_t*)dst)[i] = v; break;

            case AV_SAMPLE_FMT_FLTP: ((float  *)dst)[i] = v; break;

            case AV_SAMPLE_FMT_DBLP: ((double *)dst)[i] = v; break;

            default: av_assert0(0);

        }

    }



    av_free(tmp);

}
