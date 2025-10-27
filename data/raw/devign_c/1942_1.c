av_cold void ff_sws_init_input_funcs(SwsContext *c)

{

    enum PixelFormat srcFormat = c->srcFormat;



    c->chrToYV12 = NULL;

    switch(srcFormat) {

        case PIX_FMT_YUYV422  : c->chrToYV12 = yuy2ToUV_c; break;

        case PIX_FMT_UYVY422  : c->chrToYV12 = uyvyToUV_c; break;

        case PIX_FMT_NV12     : c->chrToYV12 = nv12ToUV_c; break;

        case PIX_FMT_NV21     : c->chrToYV12 = nv21ToUV_c; break;

        case PIX_FMT_RGB8     :

        case PIX_FMT_BGR8     :

        case PIX_FMT_PAL8     :

        case PIX_FMT_BGR4_BYTE:

        case PIX_FMT_RGB4_BYTE: c->chrToYV12 = palToUV_c; break;

        case PIX_FMT_GBRP9LE:

        case PIX_FMT_GBRP10LE:

        case PIX_FMT_GBRP16LE:  c->readChrPlanar = planar_rgb16le_to_uv; break;

        case PIX_FMT_GBRP9BE:

        case PIX_FMT_GBRP10BE:

        case PIX_FMT_GBRP16BE:  c->readChrPlanar = planar_rgb16be_to_uv; break;

        case PIX_FMT_GBRP:      c->readChrPlanar = planar_rgb_to_uv; break;

#if HAVE_BIGENDIAN

        case PIX_FMT_YUV444P9LE:

        case PIX_FMT_YUV422P9LE:

        case PIX_FMT_YUV420P9LE:

        case PIX_FMT_YUV422P10LE:

        case PIX_FMT_YUV444P10LE:

        case PIX_FMT_YUV420P10LE:

        case PIX_FMT_YUV420P16LE:

        case PIX_FMT_YUV422P16LE:

        case PIX_FMT_YUV444P16LE: c->chrToYV12 = bswap16UV_c; break;

#else

        case PIX_FMT_YUV444P9BE:

        case PIX_FMT_YUV422P9BE:

        case PIX_FMT_YUV420P9BE:

        case PIX_FMT_YUV444P10BE:

        case PIX_FMT_YUV422P10BE:

        case PIX_FMT_YUV420P10BE:

        case PIX_FMT_YUV420P16BE:

        case PIX_FMT_YUV422P16BE:

        case PIX_FMT_YUV444P16BE: c->chrToYV12 = bswap16UV_c; break;

#endif

    }

    if (c->chrSrcHSubSample) {

        switch(srcFormat) {

        case PIX_FMT_RGBA64BE: c->chrToYV12 = rgb64BEToUV_half_c; break;

        case PIX_FMT_RGBA64LE: c->chrToYV12 = rgb64LEToUV_half_c; break;

        case PIX_FMT_RGB48BE : c->chrToYV12 = rgb48BEToUV_half_c; break;

        case PIX_FMT_RGB48LE : c->chrToYV12 = rgb48LEToUV_half_c; break;

        case PIX_FMT_BGR48BE : c->chrToYV12 = bgr48BEToUV_half_c; break;

        case PIX_FMT_BGR48LE : c->chrToYV12 = bgr48LEToUV_half_c; break;

        case PIX_FMT_RGB32   : c->chrToYV12 = bgr32ToUV_half_c;   break;

        case PIX_FMT_RGB32_1 : c->chrToYV12 = bgr321ToUV_half_c;  break;

        case PIX_FMT_BGR24   : c->chrToYV12 = bgr24ToUV_half_c;   break;

        case PIX_FMT_BGR565LE: c->chrToYV12 = bgr16leToUV_half_c; break;

        case PIX_FMT_BGR565BE: c->chrToYV12 = bgr16beToUV_half_c; break;

        case PIX_FMT_BGR555LE: c->chrToYV12 = bgr15leToUV_half_c; break;

        case PIX_FMT_BGR555BE: c->chrToYV12 = bgr15beToUV_half_c; break;

        case PIX_FMT_BGR444LE: c->chrToYV12 = bgr12leToUV_half_c; break;

        case PIX_FMT_BGR444BE: c->chrToYV12 = bgr12beToUV_half_c; break;

        case PIX_FMT_BGR32   : c->chrToYV12 = rgb32ToUV_half_c;   break;

        case PIX_FMT_BGR32_1 : c->chrToYV12 = rgb321ToUV_half_c;  break;

        case PIX_FMT_RGB24   : c->chrToYV12 = rgb24ToUV_half_c;   break;

        case PIX_FMT_RGB565LE: c->chrToYV12 = rgb16leToUV_half_c; break;

        case PIX_FMT_RGB565BE: c->chrToYV12 = rgb16beToUV_half_c; break;

        case PIX_FMT_RGB555LE: c->chrToYV12 = rgb15leToUV_half_c; break;

        case PIX_FMT_RGB555BE: c->chrToYV12 = rgb15beToUV_half_c; break;

        case PIX_FMT_GBR24P  : c->chrToYV12 = gbr24pToUV_half_c;  break;

        case PIX_FMT_RGB444LE: c->chrToYV12 = rgb12leToUV_half_c; break;

        case PIX_FMT_RGB444BE: c->chrToYV12 = rgb12beToUV_half_c; break;

        }

    } else {

        switch(srcFormat) {



        case PIX_FMT_RGB48BE : c->chrToYV12 = rgb48BEToUV_c; break;

        case PIX_FMT_RGB48LE : c->chrToYV12 = rgb48LEToUV_c; break;

        case PIX_FMT_BGR48BE : c->chrToYV12 = bgr48BEToUV_c; break;

        case PIX_FMT_BGR48LE : c->chrToYV12 = bgr48LEToUV_c; break;

        case PIX_FMT_RGB32   : c->chrToYV12 = bgr32ToUV_c;   break;

        case PIX_FMT_RGB32_1 : c->chrToYV12 = bgr321ToUV_c;  break;

        case PIX_FMT_BGR24   : c->chrToYV12 = bgr24ToUV_c;   break;

        case PIX_FMT_BGR565LE: c->chrToYV12 = bgr16leToUV_c; break;

        case PIX_FMT_BGR565BE: c->chrToYV12 = bgr16beToUV_c; break;

        case PIX_FMT_BGR555LE: c->chrToYV12 = bgr15leToUV_c; break;

        case PIX_FMT_BGR555BE: c->chrToYV12 = bgr15beToUV_c; break;

        case PIX_FMT_BGR444LE: c->chrToYV12 = bgr12leToUV_c; break;

        case PIX_FMT_BGR444BE: c->chrToYV12 = bgr12beToUV_c; break;

        case PIX_FMT_BGR32   : c->chrToYV12 = rgb32ToUV_c;   break;

        case PIX_FMT_BGR32_1 : c->chrToYV12 = rgb321ToUV_c;  break;

        case PIX_FMT_RGB24   : c->chrToYV12 = rgb24ToUV_c;   break;

        case PIX_FMT_RGB565LE: c->chrToYV12 = rgb16leToUV_c; break;

        case PIX_FMT_RGB565BE: c->chrToYV12 = rgb16beToUV_c; break;

        case PIX_FMT_RGB555LE: c->chrToYV12 = rgb15leToUV_c; break;

        case PIX_FMT_RGB555BE: c->chrToYV12 = rgb15beToUV_c; break;

        case PIX_FMT_RGB444LE: c->chrToYV12 = rgb12leToUV_c; break;

        case PIX_FMT_RGB444BE: c->chrToYV12 = rgb12beToUV_c; break;

        }

    }



    c->lumToYV12 = NULL;

    c->alpToYV12 = NULL;

    switch (srcFormat) {

    case PIX_FMT_GBRP9LE:

    case PIX_FMT_GBRP10LE:

    case PIX_FMT_GBRP16LE: c->readLumPlanar = planar_rgb16le_to_y; break;

    case PIX_FMT_GBRP9BE:

    case PIX_FMT_GBRP10BE:

    case PIX_FMT_GBRP16BE: c->readLumPlanar = planar_rgb16be_to_y; break;

    case PIX_FMT_GBRP:     c->readLumPlanar = planar_rgb_to_y; break;

#if HAVE_BIGENDIAN

    case PIX_FMT_YUV444P9LE:

    case PIX_FMT_YUV422P9LE:

    case PIX_FMT_YUV420P9LE:

    case PIX_FMT_YUV444P10LE:

    case PIX_FMT_YUV422P10LE:

    case PIX_FMT_YUV420P10LE:

    case PIX_FMT_YUV420P16LE:

    case PIX_FMT_YUV422P16LE:

    case PIX_FMT_YUV444P16LE:

    case PIX_FMT_GRAY16LE: c->lumToYV12 = bswap16Y_c; break;

#else

    case PIX_FMT_YUV444P9BE:

    case PIX_FMT_YUV422P9BE:

    case PIX_FMT_YUV420P9BE:

    case PIX_FMT_YUV444P10BE:

    case PIX_FMT_YUV422P10BE:

    case PIX_FMT_YUV420P10BE:

    case PIX_FMT_YUV420P16BE:

    case PIX_FMT_YUV422P16BE:

    case PIX_FMT_YUV444P16BE:

    case PIX_FMT_GRAY16BE: c->lumToYV12 = bswap16Y_c; break;

#endif

    case PIX_FMT_YUYV422  :

    case PIX_FMT_Y400A    : c->lumToYV12 = yuy2ToY_c; break;

    case PIX_FMT_UYVY422  : c->lumToYV12 = uyvyToY_c;    break;

    case PIX_FMT_BGR24    : c->lumToYV12 = bgr24ToY_c;   break;

    case PIX_FMT_BGR565LE : c->lumToYV12 = bgr16leToY_c; break;

    case PIX_FMT_BGR565BE : c->lumToYV12 = bgr16beToY_c; break;

    case PIX_FMT_BGR555LE : c->lumToYV12 = bgr15leToY_c; break;

    case PIX_FMT_BGR555BE : c->lumToYV12 = bgr15beToY_c; break;

    case PIX_FMT_BGR444LE : c->lumToYV12 = bgr12leToY_c; break;

    case PIX_FMT_BGR444BE : c->lumToYV12 = bgr12beToY_c; break;

    case PIX_FMT_RGB24    : c->lumToYV12 = rgb24ToY_c;   break;

    case PIX_FMT_RGB565LE : c->lumToYV12 = rgb16leToY_c; break;

    case PIX_FMT_RGB565BE : c->lumToYV12 = rgb16beToY_c; break;

    case PIX_FMT_RGB555LE : c->lumToYV12 = rgb15leToY_c; break;

    case PIX_FMT_RGB555BE : c->lumToYV12 = rgb15beToY_c; break;

    case PIX_FMT_RGB444LE : c->lumToYV12 = rgb12leToY_c; break;

    case PIX_FMT_RGB444BE : c->lumToYV12 = rgb12beToY_c; break;

    case PIX_FMT_RGB8     :

    case PIX_FMT_BGR8     :

    case PIX_FMT_PAL8     :

    case PIX_FMT_BGR4_BYTE:

    case PIX_FMT_RGB4_BYTE: c->lumToYV12 = palToY_c; break;

    case PIX_FMT_MONOBLACK: c->lumToYV12 = monoblack2Y_c; break;

    case PIX_FMT_MONOWHITE: c->lumToYV12 = monowhite2Y_c; break;

    case PIX_FMT_RGB32  : c->lumToYV12 = bgr32ToY_c;  break;

    case PIX_FMT_RGB32_1: c->lumToYV12 = bgr321ToY_c; break;

    case PIX_FMT_BGR32  : c->lumToYV12 = rgb32ToY_c;  break;

    case PIX_FMT_BGR32_1: c->lumToYV12 = rgb321ToY_c; break;

    case PIX_FMT_RGB48BE: c->lumToYV12 = rgb48BEToY_c; break;

    case PIX_FMT_RGB48LE: c->lumToYV12 = rgb48LEToY_c; break;

    case PIX_FMT_BGR48BE: c->lumToYV12 = bgr48BEToY_c; break;

    case PIX_FMT_BGR48LE: c->lumToYV12 = bgr48LEToY_c; break;

    case PIX_FMT_RGBA64BE:c->lumToYV12 = rgb64BEToY_c; break;

    case PIX_FMT_RGBA64LE:c->lumToYV12 = rgb64LEToY_c; break;

    }

    if (c->alpPixBuf) {

        switch (srcFormat) {

        case PIX_FMT_RGBA64LE:

        case PIX_FMT_RGBA64BE:  c->alpToYV12 = rgba64ToA_c; break;

        case PIX_FMT_BGRA:

        case PIX_FMT_RGBA:  c->alpToYV12 = rgbaToA_c; break;

        case PIX_FMT_ABGR:

        case PIX_FMT_ARGB:  c->alpToYV12 = abgrToA_c; break;

        case PIX_FMT_Y400A: c->alpToYV12 = uyvyToY_c; break;

        case PIX_FMT_PAL8 : c->alpToYV12 = palToA_c; break;

        }

    }

}