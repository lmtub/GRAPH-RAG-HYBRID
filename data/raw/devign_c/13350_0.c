av_cold void ff_dcadsp_init_x86(DCADSPContext *s)

{

    int cpu_flags = av_get_cpu_flags();



    if (EXTERNAL_SSE(cpu_flags)) {

#if ARCH_X86_32

        s->int8x8_fmul_int32 = ff_int8x8_fmul_int32_sse;

#endif

        s->lfe_fir[0]        = ff_dca_lfe_fir0_sse;

        s->lfe_fir[1]        = ff_dca_lfe_fir1_sse;

    }



    if (EXTERNAL_SSE2(cpu_flags)) {

        s->int8x8_fmul_int32 = ff_int8x8_fmul_int32_sse2;

    }



    if (EXTERNAL_SSE4(cpu_flags)) {

        s->int8x8_fmul_int32 = ff_int8x8_fmul_int32_sse4;

    }

}
