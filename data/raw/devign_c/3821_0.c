av_cold int ff_fft_init(FFTContext *s, int nbits, int inverse)

{

    int i, j, n;



    if (nbits < 2 || nbits > 16)

        goto fail;

    s->nbits = nbits;

    n = 1 << nbits;



    s->revtab = av_malloc(n * sizeof(uint16_t));

    if (!s->revtab)

        goto fail;

    s->tmp_buf = av_malloc(n * sizeof(FFTComplex));

    if (!s->tmp_buf)

        goto fail;

    s->inverse = inverse;



    s->fft_permute = ff_fft_permute_c;

    s->fft_calc    = ff_fft_calc_c;

#if CONFIG_MDCT

    s->imdct_calc  = ff_imdct_calc_c;

    s->imdct_half  = ff_imdct_half_c;

    s->mdct_calc   = ff_mdct_calc_c;

#endif



    if (ARCH_ARM)     ff_fft_init_arm(s);

    if (HAVE_ALTIVEC) ff_fft_init_altivec(s);

    if (HAVE_MMX)     ff_fft_init_mmx(s);



    for(j=4; j<=nbits; j++) {

        ff_init_ff_cos_tabs(j);

    }

    for(i=0; i<n; i++)

        s->revtab[-split_radix_permutation(i, n, s->inverse) & (n-1)] = i;



    return 0;

 fail:

    av_freep(&s->revtab);

    av_freep(&s->tmp_buf);

    return -1;

}
