av_cold void ff_dct_init_x86(DCTContext *s)

{

    int cpu_flags = av_get_cpu_flags();



    if (EXTERNAL_SSE(cpu_flags))

        s->dct32 = ff_dct32_float_sse;

    if (EXTERNAL_SSE2(cpu_flags))

        s->dct32 = ff_dct32_float_sse2;

    if (EXTERNAL_AVX(cpu_flags))

        s->dct32 = ff_dct32_float_avx;

}
