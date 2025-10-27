SwsFunc yuv2rgb_init_altivec (SwsContext *c)

{

  if (!(c->flags & SWS_CPU_CAPS_ALTIVEC))

    return NULL;



  /*

    and this seems not to matter too much I tried a bunch of

    videos with abnormal widths and mplayer crashes else where.

    mplayer -vo x11 -rawvideo on:w=350:h=240 raw-350x240.eyuv

    boom with X11 bad match.



  */

  if ((c->srcW & 0xf) != 0)    return NULL;



  switch (c->srcFormat) {

  case PIX_FMT_YUV410P:

  case PIX_FMT_YUV420P:

  /*case IMGFMT_CLPL:	??? */

  case PIX_FMT_GRAY8:

  case PIX_FMT_NV12:

  case PIX_FMT_NV21:

    if ((c->srcH & 0x1) != 0)

      return NULL;



    switch(c->dstFormat){

    case PIX_FMT_RGB24:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space RGB24\n");

      return altivec_yuv2_rgb24;

    case PIX_FMT_BGR24:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space BGR24\n");

      return altivec_yuv2_bgr24;

    case PIX_FMT_ARGB:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space ARGB\n");

      return altivec_yuv2_argb;

    case PIX_FMT_ABGR:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space ABGR\n");

      return altivec_yuv2_abgr;

    case PIX_FMT_RGBA:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space RGBA\n");

      return altivec_yuv2_rgba;

    case PIX_FMT_BGRA:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space BGRA\n");

      return altivec_yuv2_bgra;

    default: return NULL;

    }

    break;



  case PIX_FMT_UYVY422:

    switch(c->dstFormat){

    case PIX_FMT_BGR32:

      av_log(c, AV_LOG_WARNING, "ALTIVEC: Color Space UYVY -> RGB32\n");

      return altivec_uyvy_rgb32;

    default: return NULL;

    }

    break;



  }

  return NULL;

}
