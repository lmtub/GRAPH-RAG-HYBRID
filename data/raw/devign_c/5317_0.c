av_cold void ff_fft_init_x86(FFTContext *s)

{

    int cpu_flags = av_get_cpu_flags();



#if ARCH_X86_32

    if (EXTERNAL_AMD3DNOW(cpu_flags)) {

        /* 3DNow! for K6-2/3 */

        s->imdct_calc = ff_imdct_calc_3dnow;

        s->imdct_half = ff_imdct_half_3dnow;

        s->fft_calc   = ff_fft_calc_3dnow;

    }

    if (EXTERNAL_AMD3DNOWEXT(cpu_flags)) {

        /* 3DNowEx for K7 */

        s->imdct_calc = ff_imdct_calc_3dnowext;

        s->imdct_half = ff_imdct_half_3dnowext;

        s->fft_calc   = ff_fft_calc_3dnowext;

    }

#endif

    if (EXTERNAL_SSE(cpu_flags)) {

        /* SSE for P3/P4/K8 */

        s->imdct_calc  = ff_imdct_calc_sse;

        s->imdct_half  = ff_imdct_half_sse;

        s->fft_permute = ff_fft_permute_sse;

        s->fft_calc    = ff_fft_calc_sse;

        s->fft_permutation = FF_FFT_PERM_SWAP_LSBS;

    }

    if (EXTERNAL_AVX(cpu_flags) && s->nbits >= 5) {

        /* AVX for SB */

        s->imdct_half      = ff_imdct_half_avx;

        s->fft_calc        = ff_fft_calc_avx;

        s->fft_permutation = FF_FFT_PERM_AVX;

    }

}
