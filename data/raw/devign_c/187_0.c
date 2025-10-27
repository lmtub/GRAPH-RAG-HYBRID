av_cold void ff_sws_init_swscale_x86(SwsContext *c)

{

    int cpu_flags = av_get_cpu_flags();



#if HAVE_MMX_INLINE

    if (INLINE_MMX(cpu_flags))

        sws_init_swscale_mmx(c);

#endif

#if HAVE_MMXEXT_INLINE

    if (INLINE_MMXEXT(cpu_flags))

        sws_init_swscale_mmxext(c);

#endif



#define ASSIGN_SCALE_FUNC2(hscalefn, filtersize, opt1, opt2) do { \

    if (c->srcBpc == 8) { \

        hscalefn = c->dstBpc <= 10 ? ff_hscale8to15_ ## filtersize ## _ ## opt2 : \

                                     ff_hscale8to19_ ## filtersize ## _ ## opt1; \

    } else if (c->srcBpc == 9) { \

        hscalefn = c->dstBpc <= 10 ? ff_hscale9to15_ ## filtersize ## _ ## opt2 : \

                                     ff_hscale9to19_ ## filtersize ## _ ## opt1; \

    } else if (c->srcBpc == 10) { \

        hscalefn = c->dstBpc <= 10 ? ff_hscale10to15_ ## filtersize ## _ ## opt2 : \

                                     ff_hscale10to19_ ## filtersize ## _ ## opt1; \

    } else /* c->srcBpc == 16 */ { \

        hscalefn = c->dstBpc <= 10 ? ff_hscale16to15_ ## filtersize ## _ ## opt2 : \

                                     ff_hscale16to19_ ## filtersize ## _ ## opt1; \

    } \

} while (0)

#define ASSIGN_MMX_SCALE_FUNC(hscalefn, filtersize, opt1, opt2) \

    switch (filtersize) { \

    case 4:  ASSIGN_SCALE_FUNC2(hscalefn, 4, opt1, opt2); break; \

    case 8:  ASSIGN_SCALE_FUNC2(hscalefn, 8, opt1, opt2); break; \

    default: ASSIGN_SCALE_FUNC2(hscalefn, X, opt1, opt2); break; \

    }

#define ASSIGN_VSCALEX_FUNC(vscalefn, opt, do_16_case, condition_8bit) \

switch(c->dstBpc){ \

    case 16:                          do_16_case;                          break; \

    case 10: if (!isBE(c->dstFormat)) vscalefn = ff_yuv2planeX_10_ ## opt; break; \

    case 9:  if (!isBE(c->dstFormat)) vscalefn = ff_yuv2planeX_9_  ## opt; break; \

    default: if (condition_8bit)      vscalefn = ff_yuv2planeX_8_  ## opt; break; \

    }

#define ASSIGN_VSCALE_FUNC(vscalefn, opt1, opt2, opt2chk) \

    switch(c->dstBpc){ \

    case 16: if (!isBE(c->dstFormat))            vscalefn = ff_yuv2plane1_16_ ## opt1; break; \

    case 10: if (!isBE(c->dstFormat) && opt2chk) vscalefn = ff_yuv2plane1_10_ ## opt2; break; \

    case 9:  if (!isBE(c->dstFormat) && opt2chk) vscalefn = ff_yuv2plane1_9_  ## opt2;  break; \

    default:                                     vscalefn = ff_yuv2plane1_8_  ## opt1;  break; \

    }

#define case_rgb(x, X, opt) \

        case AV_PIX_FMT_ ## X: \

            c->lumToYV12 = ff_ ## x ## ToY_ ## opt; \

            if (!c->chrSrcHSubSample) \

                c->chrToYV12 = ff_ ## x ## ToUV_ ## opt; \

            break

#if ARCH_X86_32

    if (EXTERNAL_MMX(cpu_flags)) {

        ASSIGN_MMX_SCALE_FUNC(c->hyScale, c->hLumFilterSize, mmx, mmx);

        ASSIGN_MMX_SCALE_FUNC(c->hcScale, c->hChrFilterSize, mmx, mmx);

        ASSIGN_VSCALE_FUNC(c->yuv2plane1, mmx, mmxext, cpu_flags & AV_CPU_FLAG_MMXEXT);



        switch (c->srcFormat) {

        case AV_PIX_FMT_YA8:

            c->lumToYV12 = ff_yuyvToY_mmx;

            if (c->alpPixBuf)

                c->alpToYV12 = ff_uyvyToY_mmx;

            break;

        case AV_PIX_FMT_YUYV422:

            c->lumToYV12 = ff_yuyvToY_mmx;

            c->chrToYV12 = ff_yuyvToUV_mmx;

            break;

        case AV_PIX_FMT_UYVY422:

            c->lumToYV12 = ff_uyvyToY_mmx;

            c->chrToYV12 = ff_uyvyToUV_mmx;

            break;

        case AV_PIX_FMT_NV12:

            c->chrToYV12 = ff_nv12ToUV_mmx;

            break;

        case AV_PIX_FMT_NV21:

            c->chrToYV12 = ff_nv21ToUV_mmx;

            break;

        case_rgb(rgb24, RGB24, mmx);

        case_rgb(bgr24, BGR24, mmx);

        case_rgb(bgra,  BGRA,  mmx);

        case_rgb(rgba,  RGBA,  mmx);

        case_rgb(abgr,  ABGR,  mmx);

        case_rgb(argb,  ARGB,  mmx);

        default:

            break;

        }

    }

    if (EXTERNAL_MMXEXT(cpu_flags)) {

        ASSIGN_VSCALEX_FUNC(c->yuv2planeX, mmxext, , 1);

    }

#endif /* ARCH_X86_32 */

#define ASSIGN_SSE_SCALE_FUNC(hscalefn, filtersize, opt1, opt2) \

    switch (filtersize) { \

    case 4:  ASSIGN_SCALE_FUNC2(hscalefn, 4, opt1, opt2); break; \

    case 8:  ASSIGN_SCALE_FUNC2(hscalefn, 8, opt1, opt2); break; \

    default: if (filtersize & 4) ASSIGN_SCALE_FUNC2(hscalefn, X4, opt1, opt2); \

             else                ASSIGN_SCALE_FUNC2(hscalefn, X8, opt1, opt2); \

             break; \

    }

    if (EXTERNAL_SSE2(cpu_flags)) {

        ASSIGN_SSE_SCALE_FUNC(c->hyScale, c->hLumFilterSize, sse2, sse2);

        ASSIGN_SSE_SCALE_FUNC(c->hcScale, c->hChrFilterSize, sse2, sse2);

        ASSIGN_VSCALEX_FUNC(c->yuv2planeX, sse2, ,

                            HAVE_ALIGNED_STACK || ARCH_X86_64);

        ASSIGN_VSCALE_FUNC(c->yuv2plane1, sse2, sse2, 1);



        switch (c->srcFormat) {

        case AV_PIX_FMT_YA8:

            c->lumToYV12 = ff_yuyvToY_sse2;

            if (c->alpPixBuf)

                c->alpToYV12 = ff_uyvyToY_sse2;

            break;

        case AV_PIX_FMT_YUYV422:

            c->lumToYV12 = ff_yuyvToY_sse2;

            c->chrToYV12 = ff_yuyvToUV_sse2;

            break;

        case AV_PIX_FMT_UYVY422:

            c->lumToYV12 = ff_uyvyToY_sse2;

            c->chrToYV12 = ff_uyvyToUV_sse2;

            break;

        case AV_PIX_FMT_NV12:

            c->chrToYV12 = ff_nv12ToUV_sse2;

            break;

        case AV_PIX_FMT_NV21:

            c->chrToYV12 = ff_nv21ToUV_sse2;

            break;

        case_rgb(rgb24, RGB24, sse2);

        case_rgb(bgr24, BGR24, sse2);

        case_rgb(bgra,  BGRA,  sse2);

        case_rgb(rgba,  RGBA,  sse2);

        case_rgb(abgr,  ABGR,  sse2);

        case_rgb(argb,  ARGB,  sse2);

        default:

            break;

        }

    }

    if (EXTERNAL_SSSE3(cpu_flags)) {

        ASSIGN_SSE_SCALE_FUNC(c->hyScale, c->hLumFilterSize, ssse3, ssse3);

        ASSIGN_SSE_SCALE_FUNC(c->hcScale, c->hChrFilterSize, ssse3, ssse3);

        switch (c->srcFormat) {

        case_rgb(rgb24, RGB24, ssse3);

        case_rgb(bgr24, BGR24, ssse3);

        default:

            break;

        }

    }

    if (EXTERNAL_SSE4(cpu_flags)) {

        /* Xto15 don't need special sse4 functions */

        ASSIGN_SSE_SCALE_FUNC(c->hyScale, c->hLumFilterSize, sse4, ssse3);

        ASSIGN_SSE_SCALE_FUNC(c->hcScale, c->hChrFilterSize, sse4, ssse3);

        ASSIGN_VSCALEX_FUNC(c->yuv2planeX, sse4,

                            if (!isBE(c->dstFormat)) c->yuv2planeX = ff_yuv2planeX_16_sse4,

                            HAVE_ALIGNED_STACK || ARCH_X86_64);

        if (c->dstBpc == 16 && !isBE(c->dstFormat))

            c->yuv2plane1 = ff_yuv2plane1_16_sse4;

    }



    if (EXTERNAL_AVX(cpu_flags)) {

        ASSIGN_VSCALEX_FUNC(c->yuv2planeX, avx, ,

                            HAVE_ALIGNED_STACK || ARCH_X86_64);

        ASSIGN_VSCALE_FUNC(c->yuv2plane1, avx, avx, 1);



        switch (c->srcFormat) {

        case AV_PIX_FMT_YUYV422:

            c->chrToYV12 = ff_yuyvToUV_avx;

            break;

        case AV_PIX_FMT_UYVY422:

            c->chrToYV12 = ff_uyvyToUV_avx;

            break;

        case AV_PIX_FMT_NV12:

            c->chrToYV12 = ff_nv12ToUV_avx;

            break;

        case AV_PIX_FMT_NV21:

            c->chrToYV12 = ff_nv21ToUV_avx;

            break;

        case_rgb(rgb24, RGB24, avx);

        case_rgb(bgr24, BGR24, avx);

        case_rgb(bgra,  BGRA,  avx);

        case_rgb(rgba,  RGBA,  avx);

        case_rgb(abgr,  ABGR,  avx);

        case_rgb(argb,  ARGB,  avx);

        default:

            break;

        }

    }

}
