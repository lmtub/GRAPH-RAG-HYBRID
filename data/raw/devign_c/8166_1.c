static av_cold void RENAME(sws_init_swScale)(SwsContext *c)

{

    enum PixelFormat srcFormat = c->srcFormat,

                     dstFormat = c->dstFormat;



    if (!is16BPS(dstFormat) && !is9_OR_10BPS(dstFormat)) {

        if (!(c->flags & SWS_BITEXACT)) {

            if (c->flags & SWS_ACCURATE_RND) {

                c->yuv2yuv1 = RENAME(yuv2yuv1_ar    );

                c->yuv2yuvX = RENAME(yuv2yuvX_ar    );

                if (!(c->flags & SWS_FULL_CHR_H_INT)) {

                    switch (c->dstFormat) {

                    case PIX_FMT_RGB32:   c->yuv2packedX = RENAME(yuv2rgb32_X_ar);   break;

                    case PIX_FMT_BGR24:   c->yuv2packedX = RENAME(yuv2bgr24_X_ar);   break;

                    case PIX_FMT_RGB555:  c->yuv2packedX = RENAME(yuv2rgb555_X_ar);  break;

                    case PIX_FMT_RGB565:  c->yuv2packedX = RENAME(yuv2rgb565_X_ar);  break;

                    case PIX_FMT_YUYV422: c->yuv2packedX = RENAME(yuv2yuyv422_X_ar); break;

                    default: break;

                    }

                }

            } else {

                c->yuv2yuv1 = RENAME(yuv2yuv1    );

                c->yuv2yuvX = RENAME(yuv2yuvX    );

                if (!(c->flags & SWS_FULL_CHR_H_INT)) {

                    switch (c->dstFormat) {

                    case PIX_FMT_RGB32:   c->yuv2packedX = RENAME(yuv2rgb32_X);   break;

                    case PIX_FMT_BGR24:   c->yuv2packedX = RENAME(yuv2bgr24_X);   break;

                    case PIX_FMT_RGB555:  c->yuv2packedX = RENAME(yuv2rgb555_X);  break;

                    case PIX_FMT_RGB565:  c->yuv2packedX = RENAME(yuv2rgb565_X);  break;

                    case PIX_FMT_YUYV422: c->yuv2packedX = RENAME(yuv2yuyv422_X); break;

                    default: break;

                    }

                }

            }

        }

        if (!(c->flags & SWS_FULL_CHR_H_INT)) {

            switch (c->dstFormat) {

            case PIX_FMT_RGB32:

                c->yuv2packed1 = RENAME(yuv2rgb32_1);

                c->yuv2packed2 = RENAME(yuv2rgb32_2);

                break;

            case PIX_FMT_BGR24:

                c->yuv2packed1 = RENAME(yuv2bgr24_1);

                c->yuv2packed2 = RENAME(yuv2bgr24_2);

                break;

            case PIX_FMT_RGB555:

                c->yuv2packed1 = RENAME(yuv2rgb555_1);

                c->yuv2packed2 = RENAME(yuv2rgb555_2);

                break;

            case PIX_FMT_RGB565:

                c->yuv2packed1 = RENAME(yuv2rgb565_1);

                c->yuv2packed2 = RENAME(yuv2rgb565_2);

                break;

            case PIX_FMT_YUYV422:

                c->yuv2packed1 = RENAME(yuv2yuyv422_1);

                c->yuv2packed2 = RENAME(yuv2yuyv422_2);

                break;

            default:

                break;

            }

        }

    }



#if !COMPILE_TEMPLATE_MMX2

    c->hScale       = RENAME(hScale      );

#endif /* !COMPILE_TEMPLATE_MMX2 */



    // Use the new MMX scaler if the MMX2 one can't be used (it is faster than the x86 ASM one).

#if COMPILE_TEMPLATE_MMX2

    if (c->flags & SWS_FAST_BILINEAR && c->canMMX2BeUsed)

    {

        c->hyscale_fast = RENAME(hyscale_fast);

        c->hcscale_fast = RENAME(hcscale_fast);

    } else {

#endif /* COMPILE_TEMPLATE_MMX2 */

        c->hyscale_fast = NULL;

        c->hcscale_fast = NULL;

#if COMPILE_TEMPLATE_MMX2

    }

#endif /* COMPILE_TEMPLATE_MMX2 */



#if !COMPILE_TEMPLATE_MMX2

    switch(srcFormat) {

        case PIX_FMT_YUYV422  : c->chrToYV12 = RENAME(yuy2ToUV); break;

        case PIX_FMT_UYVY422  : c->chrToYV12 = RENAME(uyvyToUV); break;

        case PIX_FMT_NV12     : c->chrToYV12 = RENAME(nv12ToUV); break;

        case PIX_FMT_NV21     : c->chrToYV12 = RENAME(nv21ToUV); break;

        case PIX_FMT_YUV420P16BE:

        case PIX_FMT_YUV422P16BE:

        case PIX_FMT_YUV444P16BE: c->chrToYV12 = RENAME(BEToUV); break;

        case PIX_FMT_YUV420P16LE:

        case PIX_FMT_YUV422P16LE:

        case PIX_FMT_YUV444P16LE: c->chrToYV12 = RENAME(LEToUV); break;

        default: break;

    }

#endif /* !COMPILE_TEMPLATE_MMX2 */

    if (!c->chrSrcHSubSample) {

        switch(srcFormat) {

        case PIX_FMT_BGR24  : c->chrToYV12 = RENAME(bgr24ToUV); break;

        case PIX_FMT_RGB24  : c->chrToYV12 = RENAME(rgb24ToUV); break;

        default: break;

        }

    }



    switch (srcFormat) {

#if !COMPILE_TEMPLATE_MMX2

    case PIX_FMT_YUYV422  :

    case PIX_FMT_YUV420P16BE:

    case PIX_FMT_YUV422P16BE:

    case PIX_FMT_YUV444P16BE:

    case PIX_FMT_Y400A    :

    case PIX_FMT_GRAY16BE : c->lumToYV12 = RENAME(yuy2ToY); break;

    case PIX_FMT_UYVY422  :

    case PIX_FMT_YUV420P16LE:

    case PIX_FMT_YUV422P16LE:

    case PIX_FMT_YUV444P16LE:

    case PIX_FMT_GRAY16LE : c->lumToYV12 = RENAME(uyvyToY); break;

#endif /* !COMPILE_TEMPLATE_MMX2 */

    case PIX_FMT_BGR24    : c->lumToYV12 = RENAME(bgr24ToY); break;

    case PIX_FMT_RGB24    : c->lumToYV12 = RENAME(rgb24ToY); break;

    default: break;

    }

#if !COMPILE_TEMPLATE_MMX2

    if (c->alpPixBuf) {

        switch (srcFormat) {

        case PIX_FMT_Y400A  : c->alpToYV12 = RENAME(yuy2ToY); break;

        default: break;

        }

    }

#endif /* !COMPILE_TEMPLATE_MMX2 */

}
