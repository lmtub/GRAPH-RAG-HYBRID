av_cold void ff_fft_fixed_init_arm(FFTContext *s)

{

    int cpu_flags = av_get_cpu_flags();



    if (have_neon(cpu_flags)) {

        s->fft_permutation = FF_FFT_PERM_SWAP_LSBS;

        s->fft_calc        = ff_fft_fixed_calc_neon;



#if CONFIG_MDCT

        if (!s->inverse && s->mdct_bits >= 5) {

            s->mdct_permutation = FF_MDCT_PERM_INTERLEAVE;

            s->mdct_calc        = ff_mdct_fixed_calc_neon;

            s->mdct_calcw       = ff_mdct_fixed_calcw_neon;

        }

#endif

    }

}
