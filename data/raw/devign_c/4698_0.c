void sws_rgb2rgb_init(int flags)

{

#if HAVE_SSE2 || HAVE_MMX2 || HAVE_AMD3DNOW || HAVE_MMX

    if (flags & SWS_CPU_CAPS_SSE2)

        rgb2rgb_init_SSE2();

    else if (flags & SWS_CPU_CAPS_MMX2)

        rgb2rgb_init_MMX2();

    else if (flags & SWS_CPU_CAPS_3DNOW)

        rgb2rgb_init_3DNOW();

    else if (flags & SWS_CPU_CAPS_MMX)

        rgb2rgb_init_MMX();

    else

#endif /* HAVE_MMX2 || HAVE_AMD3DNOW || HAVE_MMX */

        rgb2rgb_init_C();

}
