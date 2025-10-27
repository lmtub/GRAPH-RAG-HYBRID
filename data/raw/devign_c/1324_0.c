av_cold void ff_videodsp_init_x86(VideoDSPContext *ctx, int bpc)

{

#if HAVE_YASM

    int cpu_flags = av_get_cpu_flags();



#if ARCH_X86_32

    if (EXTERNAL_MMX(cpu_flags) && bpc <= 8) {

        ctx->emulated_edge_mc = emulated_edge_mc_mmx;

    }

    if (EXTERNAL_AMD3DNOW(cpu_flags)) {

        ctx->prefetch = ff_prefetch_3dnow;

    }

#endif /* ARCH_X86_32 */

    if (EXTERNAL_MMXEXT(cpu_flags)) {

        ctx->prefetch = ff_prefetch_mmxext;

#if ARCH_X86_32

        if (bpc <= 8)

            ctx->emulated_edge_mc = emulated_edge_mc_mmxext;

#endif /* ARCH_X86_32 */

    }

#if ARCH_X86_32

    if (EXTERNAL_SSE(cpu_flags) && bpc <= 8) {

        ctx->emulated_edge_mc = emulated_edge_mc_sse;

    }

#endif /* ARCH_X86_32 */

    if (EXTERNAL_SSE2(cpu_flags) && bpc <= 8) {

        ctx->emulated_edge_mc = emulated_edge_mc_sse2;

    }

#endif /* HAVE_YASM */

}
