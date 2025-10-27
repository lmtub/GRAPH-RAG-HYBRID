SwsFunc yuv2rgb_get_func_ptr (SwsContext *c)

{

#if defined(HAVE_MMX2) || defined(HAVE_MMX)

    if(c->flags & SWS_CPU_CAPS_MMX2){

	switch(c->dstFormat){

	case PIX_FMT_RGB32: return yuv420_rgb32_MMX2;

	case PIX_FMT_BGR24: return yuv420_rgb24_MMX2;

	case PIX_FMT_BGR565: return yuv420_rgb16_MMX2;

	case PIX_FMT_BGR555: return yuv420_rgb15_MMX2;

	}

    }

    if(c->flags & SWS_CPU_CAPS_MMX){

	switch(c->dstFormat){

	case PIX_FMT_RGB32: return yuv420_rgb32_MMX;

	case PIX_FMT_BGR24: return yuv420_rgb24_MMX;

	case PIX_FMT_BGR565: return yuv420_rgb16_MMX;

	case PIX_FMT_BGR555: return yuv420_rgb15_MMX;

	}

    }

#endif

#ifdef HAVE_MLIB

    {

	SwsFunc t= yuv2rgb_init_mlib(c);

	if(t) return t;

    }

#endif

#ifdef HAVE_ALTIVEC

    if (c->flags & SWS_CPU_CAPS_ALTIVEC)

    {

	SwsFunc t = yuv2rgb_init_altivec(c);

	if(t) return t;

    }

#endif



    av_log(c, AV_LOG_WARNING, "No accelerated colorspace conversion found\n");



    switch(c->dstFormat){

    case PIX_FMT_BGR32:

    case PIX_FMT_RGB32: return yuv2rgb_c_32;

    case PIX_FMT_RGB24: return yuv2rgb_c_24_rgb;

    case PIX_FMT_BGR24: return yuv2rgb_c_24_bgr;

    case PIX_FMT_RGB565:

    case PIX_FMT_BGR565:

    case PIX_FMT_RGB555:

    case PIX_FMT_BGR555: return yuv2rgb_c_16;

    case PIX_FMT_RGB8:

    case PIX_FMT_BGR8:  return yuv2rgb_c_8_ordered_dither;

    case PIX_FMT_RGB4:

    case PIX_FMT_BGR4:  return yuv2rgb_c_4_ordered_dither;

    case PIX_FMT_RGB4_BYTE:

    case PIX_FMT_BGR4_BYTE:  return yuv2rgb_c_4b_ordered_dither;

    case PIX_FMT_MONOBLACK:  return yuv2rgb_c_1_ordered_dither;

    default:

    	assert(0);

    }

    return NULL;

}
