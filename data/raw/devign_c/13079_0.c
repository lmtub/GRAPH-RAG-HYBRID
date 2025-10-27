av_cold int ff_mdct_init(FFTContext *s, int nbits, int inverse, double scale)

{

    int n, n4, i;

    double alpha, theta;

    int tstep;



    memset(s, 0, sizeof(*s));

    n = 1 << nbits;

    s->mdct_bits = nbits;

    s->mdct_size = n;

    n4 = n >> 2;

    s->permutation = FF_MDCT_PERM_NONE;



    if (ff_fft_init(s, s->mdct_bits - 2, inverse) < 0)

        goto fail;



    s->tcos = av_malloc(n/2 * sizeof(FFTSample));

    if (!s->tcos)

        goto fail;



    switch (s->permutation) {

    case FF_MDCT_PERM_NONE:

        s->tsin = s->tcos + n4;

        tstep = 1;

        break;

    case FF_MDCT_PERM_INTERLEAVE:

        s->tsin = s->tcos + 1;

        tstep = 2;

        break;

    default:

        goto fail;

    }



    theta = 1.0 / 8.0 + (scale < 0 ? n4 : 0);

    scale = sqrt(fabs(scale));

    for(i=0;i<n4;i++) {

        alpha = 2 * M_PI * (i + theta) / n;

        s->tcos[i*tstep] = -cos(alpha) * scale;

        s->tsin[i*tstep] = -sin(alpha) * scale;

    }

    return 0;

 fail:

    ff_mdct_end(s);

    return -1;

}
