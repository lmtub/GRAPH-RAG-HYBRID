static void audiogen(void *data, enum AVSampleFormat sample_fmt,

                     int channels, int sample_rate, int nb_samples)

{

    int i, ch, k;

    double v, f, a, ampa;

    double tabf1[SWR_CH_MAX];

    double tabf2[SWR_CH_MAX];

    double taba[SWR_CH_MAX];

    unsigned static rnd;



#define PUT_SAMPLE set(data, ch, k, channels, sample_fmt, v);

#define uint_rand(x) (x = x * 1664525 + 1013904223)

#define dbl_rand(x) (uint_rand(x)*2.0 / (double)UINT_MAX - 1)

    k = 0;



    /* 1 second of single freq sinus at 1000 Hz */

    a = 0;

    for (i = 0; i < 1 * sample_rate && k < nb_samples; i++, k++) {

        v = sin(a) * 0.30;

        for (ch = 0; ch < channels; ch++)

            PUT_SAMPLE

        a += M_PI * 1000.0 * 2.0 / sample_rate;

    }



    /* 1 second of varying frequency between 100 and 10000 Hz */

    a = 0;

    for (i = 0; i < 1 * sample_rate && k < nb_samples; i++, k++) {

        v = sin(a) * 0.30;

        for (ch = 0; ch < channels; ch++)

            PUT_SAMPLE

        f  = 100.0 + (((10000.0 - 100.0) * i) / sample_rate);

        a += M_PI * f * 2.0 / sample_rate;

    }



    /* 0.5 second of low amplitude white noise */

    for (i = 0; i < sample_rate / 2 && k < nb_samples; i++, k++) {

        v = dbl_rand(rnd) * 0.30;

        for (ch = 0; ch < channels; ch++)

            PUT_SAMPLE

    }



    /* 0.5 second of high amplitude white noise */

    for (i = 0; i < sample_rate / 2 && k < nb_samples; i++, k++) {

        v = dbl_rand(rnd);

        for (ch = 0; ch < channels; ch++)

            PUT_SAMPLE

    }



    /* 1 second of unrelated ramps for each channel */

    for (ch = 0; ch < channels; ch++) {

        taba[ch]  = 0;

        tabf1[ch] = 100 + uint_rand(rnd) % 5000;

        tabf2[ch] = 100 + uint_rand(rnd) % 5000;

    }

    for (i = 0; i < 1 * sample_rate && k < nb_samples; i++, k++) {

        for (ch = 0; ch < channels; ch++) {

            v = sin(taba[ch]) * 0.30;

            PUT_SAMPLE

            f = tabf1[ch] + (((tabf2[ch] - tabf1[ch]) * i) / sample_rate);

            taba[ch] += M_PI * f * 2.0 / sample_rate;

        }

    }



    /* 2 seconds of 500 Hz with varying volume */

    a    = 0;

    ampa = 0;

    for (i = 0; i < 2 * sample_rate && k < nb_samples; i++, k++) {

        for (ch = 0; ch < channels; ch++) {

            double amp = (1.0 + sin(ampa)) * 0.15;

            if (ch & 1)

                amp = 0.30 - amp;

            v = sin(a) * amp;

            PUT_SAMPLE

            a    += M_PI * 500.0 * 2.0 / sample_rate;

            ampa += M_PI *  2.0 / sample_rate;

        }

    }

}
