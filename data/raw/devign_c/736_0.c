SYNTH_FILTER_FUNC(sse2)

SYNTH_FILTER_FUNC(avx)

SYNTH_FILTER_FUNC(fma3)

#endif /* HAVE_YASM */



av_cold void ff_synth_filter_init_x86(SynthFilterContext *s)

{

#if HAVE_YASM

    int cpu_flags = av_get_cpu_flags();



#if ARCH_X86_32

    if (EXTERNAL_SSE(cpu_flags)) {

        s->synth_filter_float = synth_filter_sse;

    }

#endif

    if (EXTERNAL_SSE2(cpu_flags)) {

        s->synth_filter_float = synth_filter_sse2;

    }

    if (EXTERNAL_AVX(cpu_flags)) {

        s->synth_filter_float = synth_filter_avx;

    }

    if (EXTERNAL_FMA3(cpu_flags)) {

        s->synth_filter_float = synth_filter_fma3;

    }

#endif /* HAVE_YASM */

}
