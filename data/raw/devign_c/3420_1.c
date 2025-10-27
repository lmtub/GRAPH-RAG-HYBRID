static int build_filter(ResampleContext *c, void *filter, double factor, int tap_count, int alloc, int phase_count, int scale,

                        int filter_type, int kaiser_beta){

    int ph, i;

    double x, y, w;

    double *tab = av_malloc_array(tap_count,  sizeof(*tab));

    const int center= (tap_count-1)/2;



    if (!tab)

        return AVERROR(ENOMEM);



    /* if upsampling, only need to interpolate, no filter */

    if (factor > 1.0)

        factor = 1.0;



    for(ph=0;ph<phase_count;ph++) {

        double norm = 0;

        for(i=0;i<tap_count;i++) {

            x = M_PI * ((double)(i - center) - (double)ph / phase_count) * factor;

            if (x == 0) y = 1.0;

            else        y = sin(x) / x;

            switch(filter_type){

            case SWR_FILTER_TYPE_CUBIC:{

                const float d= -0.5; //first order derivative = -0.5

                x = fabs(((double)(i - center) - (double)ph / phase_count) * factor);

                if(x<1.0) y= 1 - 3*x*x + 2*x*x*x + d*(            -x*x + x*x*x);

                else      y=                       d*(-4 + 8*x - 5*x*x + x*x*x);

                break;}

            case SWR_FILTER_TYPE_BLACKMAN_NUTTALL:

                w = 2.0*x / (factor*tap_count) + M_PI;

                y *= 0.3635819 - 0.4891775 * cos(w) + 0.1365995 * cos(2*w) - 0.0106411 * cos(3*w);

                break;

            case SWR_FILTER_TYPE_KAISER:

                w = 2.0*x / (factor*tap_count*M_PI);

                y *= bessel(kaiser_beta*sqrt(FFMAX(1-w*w, 0)));

                break;

            default:

                av_assert0(0);

            }



            tab[i] = y;

            norm += y;

        }



        /* normalize so that an uniform color remains the same */

        switch(c->format){

        case AV_SAMPLE_FMT_S16P:

            for(i=0;i<tap_count;i++)

                ((int16_t*)filter)[ph * alloc + i] = av_clip(lrintf(tab[i] * scale / norm), INT16_MIN, INT16_MAX);

            break;

        case AV_SAMPLE_FMT_S32P:

            for(i=0;i<tap_count;i++)

                ((int32_t*)filter)[ph * alloc + i] = av_clipl_int32(llrint(tab[i] * scale / norm));

            break;

        case AV_SAMPLE_FMT_FLTP:

            for(i=0;i<tap_count;i++)

                ((float*)filter)[ph * alloc + i] = tab[i] * scale / norm;

            break;

        case AV_SAMPLE_FMT_DBLP:

            for(i=0;i<tap_count;i++)

                ((double*)filter)[ph * alloc + i] = tab[i] * scale / norm;

            break;

        }

    }

#if 0

    {

#define LEN 1024

        int j,k;

        double sine[LEN + tap_count];

        double filtered[LEN];

        double maxff=-2, minff=2, maxsf=-2, minsf=2;

        for(i=0; i<LEN; i++){

            double ss=0, sf=0, ff=0;

            for(j=0; j<LEN+tap_count; j++)

                sine[j]= cos(i*j*M_PI/LEN);

            for(j=0; j<LEN; j++){

                double sum=0;

                ph=0;

                for(k=0; k<tap_count; k++)

                    sum += filter[ph * tap_count + k] * sine[k+j];

                filtered[j]= sum / (1<<FILTER_SHIFT);

                ss+= sine[j + center] * sine[j + center];

                ff+= filtered[j] * filtered[j];

                sf+= sine[j + center] * filtered[j];

            }

            ss= sqrt(2*ss/LEN);

            ff= sqrt(2*ff/LEN);

            sf= 2*sf/LEN;

            maxff= FFMAX(maxff, ff);

            minff= FFMIN(minff, ff);

            maxsf= FFMAX(maxsf, sf);

            minsf= FFMIN(minsf, sf);

            if(i%11==0){

                av_log(NULL, AV_LOG_ERROR, "i:%4d ss:%f ff:%13.6e-%13.6e sf:%13.6e-%13.6e\n", i, ss, maxff, minff, maxsf, minsf);

                minff=minsf= 2;

                maxff=maxsf= -2;

            }

        }

    }

#endif



    av_free(tab);

    return 0;

}
