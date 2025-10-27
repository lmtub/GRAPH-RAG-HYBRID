av_cold void ff_fft_init_mmx(FFTContext *s)

{

#if HAVE_YASM

    int has_vectors = av_get_cpu_flags();

#if ARCH_X86_32

    if (has_vectors & AV_CPU_FLAG_3DNOW && HAVE_AMD3DNOW) {

        /* 3DNow! for K6-2/3 */

        s->imdct_calc = ff_imdct_calc_3dnow;

        s->imdct_half = ff_imdct_half_3dnow;

        s->fft_calc   = ff_fft_calc_3dnow;

    }

    if (has_vectors & AV_CPU_FLAG_3DNOWEXT && HAVE_AMD3DNOWEXT) {

        /* 3DNowEx for K7 */

        s->imdct_calc = ff_imdct_calc_3dnowext;

        s->imdct_half = ff_imdct_half_3dnowext;

        s->fft_calc   = ff_fft_calc_3dnowext;

    }

#endif

    if (has_vectors & AV_CPU_FLAG_SSE && HAVE_SSE) {

        /* SSE for P3/P4/K8 */

        s->imdct_calc  = ff_imdct_calc_sse;

        s->imdct_half  = ff_imdct_half_sse;

        s->fft_permute = ff_fft_permute_sse;

        s->fft_calc    = ff_fft_calc_sse;

        s->fft_permutation = FF_FFT_PERM_SWAP_LSBS;

    }

    if (has_vectors & AV_CPU_FLAG_AVX && HAVE_AVX && s->nbits >= 5) {

        /* AVX for SB */

        s->imdct_half      = ff_imdct_half_avx;

        s->fft_calc        = ff_fft_calc_avx;

        s->fft_permutation = FF_FFT_PERM_AVX;

    }

#endif

}
