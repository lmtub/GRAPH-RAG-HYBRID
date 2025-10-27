void ff_imdct_calc(MDCTContext *s, FFTSample *output,

                   const FFTSample *input, FFTSample *tmp)

{

    int k, n8, n4, n2, n, j;

    const uint16_t *revtab = s->fft.revtab;

    const FFTSample *tcos = s->tcos;

    const FFTSample *tsin = s->tsin;

    const FFTSample *in1, *in2;

    FFTComplex *z = (FFTComplex *)tmp;



    n = 1 << s->nbits;

    n2 = n >> 1;

    n4 = n >> 2;

    n8 = n >> 3;



    /* pre rotation */

    in1 = input;

    in2 = input + n2 - 1;

    for(k = 0; k < n4; k++) {

        j=revtab[k];

        CMUL(z[j].re, z[j].im, *in2, *in1, tcos[k], tsin[k]);

        in1 += 2;

        in2 -= 2;

    }

    ff_fft_calc(&s->fft, z);



    /* post rotation + reordering */

    /* XXX: optimize */

    for(k = 0; k < n4; k++) {

        CMUL(z[k].re, z[k].im, z[k].re, z[k].im, tcos[k], tsin[k]);

    }

    for(k = 0; k < n8; k++) {

        output[2*k] = -z[n8 + k].im;

        output[n2-1-2*k] = z[n8 + k].im;



        output[2*k+1] = z[n8-1-k].re;

        output[n2-1-2*k-1] = -z[n8-1-k].re;



        output[n2 + 2*k]=-z[k+n8].re;

        output[n-1- 2*k]=-z[k+n8].re;



        output[n2 + 2*k+1]=z[n8-k-1].im;

        output[n-2 - 2 * k] = z[n8-k-1].im;

    }

}
