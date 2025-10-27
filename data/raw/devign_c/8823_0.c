void ff_float_dsp_init_x86(AVFloatDSPContext *fdsp)

{

#if HAVE_YASM

    int mm_flags = av_get_cpu_flags();



    if (mm_flags & AV_CPU_FLAG_SSE && HAVE_SSE) {

        fdsp->vector_fmul = ff_vector_fmul_sse;

        fdsp->vector_fmac_scalar = ff_vector_fmac_scalar_sse;

    }

    if (mm_flags & AV_CPU_FLAG_AVX && HAVE_AVX) {

        fdsp->vector_fmul = ff_vector_fmul_avx;

        fdsp->vector_fmac_scalar = ff_vector_fmac_scalar_avx;

    }

#endif

}
