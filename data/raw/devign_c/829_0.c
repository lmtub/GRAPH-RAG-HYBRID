find_c_packed_planar_out_funcs(SwsContext *c,

                               yuv2planar1_fn *yuv2yuv1,    yuv2planarX_fn *yuv2yuvX,

                               yuv2packed1_fn *yuv2packed1, yuv2packed2_fn *yuv2packed2,

                               yuv2packedX_fn *yuv2packedX)

{

    enum PixelFormat dstFormat = c->dstFormat;



    if (dstFormat == PIX_FMT_NV12 || dstFormat == PIX_FMT_NV21) {

        *yuv2yuvX     = yuv2nv12X_c;

    } else if (is16BPS(dstFormat)) {

        *yuv2yuvX     = isBE(dstFormat) ? yuv2yuvX16BE_c  : yuv2yuvX16LE_c;

    } else if (is9_OR_10BPS(dstFormat)) {

        if (av_pix_fmt_descriptors[dstFormat].comp[0].depth_minus1 == 8) {

            *yuv2yuvX = isBE(dstFormat) ? yuv2yuvX9BE_c :  yuv2yuvX9LE_c;

        } else {

            *yuv2yuvX = isBE(dstFormat) ? yuv2yuvX10BE_c : yuv2yuvX10LE_c;

        }

    } else {

        *yuv2yuv1     = yuv2yuv1_c;

        *yuv2yuvX     = yuv2yuvX_c;

    }

    if(c->flags & SWS_FULL_CHR_H_INT) {

        switch (dstFormat) {

            case PIX_FMT_RGBA:

#if CONFIG_SMALL

                *yuv2packedX = yuv2rgba32_full_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packedX = yuv2rgba32_full_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packedX = yuv2rgbx32_full_X_c;

                }

#endif /* !CONFIG_SMALL */

                break;

            case PIX_FMT_ARGB:

#if CONFIG_SMALL

                *yuv2packedX = yuv2argb32_full_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packedX = yuv2argb32_full_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packedX = yuv2xrgb32_full_X_c;

                }

#endif /* !CONFIG_SMALL */

                break;

            case PIX_FMT_BGRA:

#if CONFIG_SMALL

                *yuv2packedX = yuv2bgra32_full_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packedX = yuv2bgra32_full_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packedX = yuv2bgrx32_full_X_c;

                }

#endif /* !CONFIG_SMALL */

                break;

            case PIX_FMT_ABGR:

#if CONFIG_SMALL

                *yuv2packedX = yuv2abgr32_full_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packedX = yuv2abgr32_full_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packedX = yuv2xbgr32_full_X_c;

                }

#endif /* !CONFIG_SMALL */

                break;

            case PIX_FMT_RGB24:

            *yuv2packedX = yuv2rgb24_full_X_c;

            break;

        case PIX_FMT_BGR24:

            *yuv2packedX = yuv2bgr24_full_X_c;

            break;

        }

    } else {

        switch (dstFormat) {

        case PIX_FMT_GRAY16BE:

            *yuv2packed1 = yuv2gray16BE_1_c;

            *yuv2packed2 = yuv2gray16BE_2_c;

            *yuv2packedX = yuv2gray16BE_X_c;

            break;

        case PIX_FMT_GRAY16LE:

            *yuv2packed1 = yuv2gray16LE_1_c;

            *yuv2packed2 = yuv2gray16LE_2_c;

            *yuv2packedX = yuv2gray16LE_X_c;

            break;

        case PIX_FMT_MONOWHITE:

            *yuv2packed1 = yuv2monowhite_1_c;

            *yuv2packed2 = yuv2monowhite_2_c;

            *yuv2packedX = yuv2monowhite_X_c;

            break;

        case PIX_FMT_MONOBLACK:

            *yuv2packed1 = yuv2monoblack_1_c;

            *yuv2packed2 = yuv2monoblack_2_c;

            *yuv2packedX = yuv2monoblack_X_c;

            break;

        case PIX_FMT_YUYV422:

            *yuv2packed1 = yuv2yuyv422_1_c;

            *yuv2packed2 = yuv2yuyv422_2_c;

            *yuv2packedX = yuv2yuyv422_X_c;

            break;

        case PIX_FMT_UYVY422:

            *yuv2packed1 = yuv2uyvy422_1_c;

            *yuv2packed2 = yuv2uyvy422_2_c;

            *yuv2packedX = yuv2uyvy422_X_c;

            break;

        case PIX_FMT_RGB48LE:

            //*yuv2packed1 = yuv2rgb48le_1_c;

            //*yuv2packed2 = yuv2rgb48le_2_c;

            //*yuv2packedX = yuv2rgb48le_X_c;

            //break;

        case PIX_FMT_RGB48BE:

            *yuv2packed1 = yuv2rgb48be_1_c;

            *yuv2packed2 = yuv2rgb48be_2_c;

            *yuv2packedX = yuv2rgb48be_X_c;

            break;

        case PIX_FMT_BGR48LE:

            //*yuv2packed1 = yuv2bgr48le_1_c;

            //*yuv2packed2 = yuv2bgr48le_2_c;

            //*yuv2packedX = yuv2bgr48le_X_c;

            //break;

        case PIX_FMT_BGR48BE:

            *yuv2packed1 = yuv2bgr48be_1_c;

            *yuv2packed2 = yuv2bgr48be_2_c;

            *yuv2packedX = yuv2bgr48be_X_c;

            break;

        case PIX_FMT_RGB32:

        case PIX_FMT_BGR32:

#if CONFIG_SMALL

            *yuv2packed1 = yuv2rgb32_1_c;

            *yuv2packed2 = yuv2rgb32_2_c;

            *yuv2packedX = yuv2rgb32_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packed1 = yuv2rgba32_1_c;

                    *yuv2packed2 = yuv2rgba32_2_c;

                    *yuv2packedX = yuv2rgba32_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packed1 = yuv2rgbx32_1_c;

                    *yuv2packed2 = yuv2rgbx32_2_c;

                    *yuv2packedX = yuv2rgbx32_X_c;

                }

#endif /* !CONFIG_SMALL */

            break;

        case PIX_FMT_RGB32_1:

        case PIX_FMT_BGR32_1:

#if CONFIG_SMALL

                *yuv2packed1 = yuv2rgb32_1_1_c;

                *yuv2packed2 = yuv2rgb32_1_2_c;

                *yuv2packedX = yuv2rgb32_1_X_c;

#else

#if CONFIG_SWSCALE_ALPHA

                if (c->alpPixBuf) {

                    *yuv2packed1 = yuv2rgba32_1_1_c;

                    *yuv2packed2 = yuv2rgba32_1_2_c;

                    *yuv2packedX = yuv2rgba32_1_X_c;

                } else

#endif /* CONFIG_SWSCALE_ALPHA */

                {

                    *yuv2packed1 = yuv2rgbx32_1_1_c;

                    *yuv2packed2 = yuv2rgbx32_1_2_c;

                    *yuv2packedX = yuv2rgbx32_1_X_c;

                }

#endif /* !CONFIG_SMALL */

                break;

        case PIX_FMT_RGB24:

            *yuv2packed1 = yuv2rgb24_1_c;

            *yuv2packed2 = yuv2rgb24_2_c;

            *yuv2packedX = yuv2rgb24_X_c;

            break;

        case PIX_FMT_BGR24:

            *yuv2packed1 = yuv2bgr24_1_c;

            *yuv2packed2 = yuv2bgr24_2_c;

            *yuv2packedX = yuv2bgr24_X_c;

            break;

        case PIX_FMT_RGB565:

        case PIX_FMT_BGR565:

            *yuv2packed1 = yuv2rgb16_1_c;

            *yuv2packed2 = yuv2rgb16_2_c;

            *yuv2packedX = yuv2rgb16_X_c;

            break;

        case PIX_FMT_RGB555:

        case PIX_FMT_BGR555:

            *yuv2packed1 = yuv2rgb15_1_c;

            *yuv2packed2 = yuv2rgb15_2_c;

            *yuv2packedX = yuv2rgb15_X_c;

            break;

        case PIX_FMT_RGB444:

        case PIX_FMT_BGR444:

            *yuv2packed1 = yuv2rgb12_1_c;

            *yuv2packed2 = yuv2rgb12_2_c;

            *yuv2packedX = yuv2rgb12_X_c;

            break;

        case PIX_FMT_RGB8:

        case PIX_FMT_BGR8:

            *yuv2packed1 = yuv2rgb8_1_c;

            *yuv2packed2 = yuv2rgb8_2_c;

            *yuv2packedX = yuv2rgb8_X_c;

            break;

        case PIX_FMT_RGB4:

        case PIX_FMT_BGR4:

            *yuv2packed1 = yuv2rgb4_1_c;

            *yuv2packed2 = yuv2rgb4_2_c;

            *yuv2packedX = yuv2rgb4_X_c;

            break;

        case PIX_FMT_RGB4_BYTE:

        case PIX_FMT_BGR4_BYTE:

            *yuv2packed1 = yuv2rgb4b_1_c;

            *yuv2packed2 = yuv2rgb4b_2_c;

            *yuv2packedX = yuv2rgb4b_X_c;

            break;

        }

    }

}
