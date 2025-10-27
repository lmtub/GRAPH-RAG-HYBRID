static void fft_ref_init(int nbits, int inverse)

{

    int i, n = 1 << nbits;



    exptab = av_malloc((n / 2) * sizeof(*exptab));



    for (i = 0; i < (n/2); i++) {

        double alpha = 2 * M_PI * (float)i / (float)n;

        double c1 = cos(alpha), s1 = sin(alpha);

        if (!inverse)

            s1 = -s1;

        exptab[i].re = c1;

        exptab[i].im = s1;

    }

}
