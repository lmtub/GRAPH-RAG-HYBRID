av_cold int ff_dct_init(DCTContext *s, int nbits, enum DCTTransformType inverse)

{

    int n = 1 << nbits;

    int i;



    s->nbits    = nbits;

    s->inverse  = inverse;



    ff_init_ff_cos_tabs(nbits+2);



    s->costab = ff_cos_tabs[nbits+2];



    s->csc2 = av_malloc(n/2 * sizeof(FFTSample));



    if (ff_rdft_init(&s->rdft, nbits, inverse == DCT_III) < 0) {

        av_free(s->csc2);

        return -1;

    }



    for (i = 0; i < n/2; i++)

        s->csc2[i] = 0.5 / sin((M_PI / (2*n) * (2*i + 1)));



    switch(inverse) {

    case DCT_I  : s->dct_calc = ff_dct_calc_I_c; break;

    case DCT_II : s->dct_calc = ff_dct_calc_II_c ; break;

    case DCT_III: s->dct_calc = ff_dct_calc_III_c; break;

    case DST_I  : s->dct_calc = ff_dst_calc_I_c; break;

    }



    if (inverse == DCT_II && nbits == 5)

        s->dct_calc = dct32_func;



    s->dct32 = dct32;

    if (HAVE_MMX)     ff_dct_init_mmx(s);



    return 0;

}
