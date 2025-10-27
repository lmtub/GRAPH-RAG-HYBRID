SwsFunc ff_yuv2rgb_init_mmx(SwsContext *c)

{

    int cpu_flags = av_get_cpu_flags();



    if (c->srcFormat != PIX_FMT_YUV420P &&

        c->srcFormat != PIX_FMT_YUVA420P)

        return NULL;



    if (HAVE_MMX2 && cpu_flags & AV_CPU_FLAG_MMX2) {

        switch (c->dstFormat) {

        case PIX_FMT_RGB24:  return yuv420_rgb24_MMX2;

        case PIX_FMT_BGR24:  return yuv420_bgr24_MMX2;

        }

    }



    if (HAVE_MMX && cpu_flags & AV_CPU_FLAG_MMX) {

        switch (c->dstFormat) {

            case PIX_FMT_RGB32:

                if (CONFIG_SWSCALE_ALPHA && c->srcFormat == PIX_FMT_YUVA420P) {

#if HAVE_7REGS

                    return yuva420_rgb32_MMX;

#endif

                    break;

                } else return yuv420_rgb32_MMX;

            case PIX_FMT_BGR32:

                if (CONFIG_SWSCALE_ALPHA && c->srcFormat == PIX_FMT_YUVA420P) {

#if HAVE_7REGS

                    return yuva420_bgr32_MMX;

#endif

                    break;

                } else return yuv420_bgr32_MMX;

            case PIX_FMT_RGB24:  return yuv420_rgb24_MMX;

            case PIX_FMT_BGR24:  return yuv420_bgr24_MMX;

            case PIX_FMT_RGB565: return yuv420_rgb16_MMX;

            case PIX_FMT_RGB555: return yuv420_rgb15_MMX;

        }

    }



    return NULL;

}
