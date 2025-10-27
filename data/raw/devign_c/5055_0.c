av_cold void rgb2rgb_init_x86(void)

{

#if HAVE_INLINE_ASM

    int cpu_flags = av_get_cpu_flags();



    if (cpu_flags & AV_CPU_FLAG_MMX)

        rgb2rgb_init_MMX();

    if (HAVE_AMD3DNOW && cpu_flags & AV_CPU_FLAG_3DNOW)

        rgb2rgb_init_3DNOW();

    if (HAVE_MMXEXT   && cpu_flags & AV_CPU_FLAG_MMXEXT)

        rgb2rgb_init_MMX2();

    if (HAVE_SSE      && cpu_flags & AV_CPU_FLAG_SSE2)

        rgb2rgb_init_SSE2();

#endif /* HAVE_INLINE_ASM */

}
