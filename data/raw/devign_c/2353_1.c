SwsFunc yuv2rgb_init_mlib(SwsContext *c)

{

	switch(c->dstFormat){

	case PIX_FMT_RGB24: return mlib_YUV2RGB420_24;

	case PIX_FMT_BGR32: return mlib_YUV2ARGB420_32;

	case PIX_FMT_RGB32: return mlib_YUV2ABGR420_32;

	default: return NULL;

	}

}
