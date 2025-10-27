static inline void yuv2packedXinC(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				    uint8_t *dest, int dstW, int y)

{

	int i;

	switch(c->dstFormat)

	{

	case PIX_FMT_BGR32:

	case PIX_FMT_RGB32:

		YSCALE_YUV_2_RGBX_C(uint32_t)

			((uint32_t*)dest)[i2+0]= r[Y1] + g[Y1] + b[Y1];

			((uint32_t*)dest)[i2+1]= r[Y2] + g[Y2] + b[Y2];

		}

		break;

	case PIX_FMT_RGB24:

		YSCALE_YUV_2_RGBX_C(uint8_t)

			((uint8_t*)dest)[0]= r[Y1];

			((uint8_t*)dest)[1]= g[Y1];

			((uint8_t*)dest)[2]= b[Y1];

			((uint8_t*)dest)[3]= r[Y2];

			((uint8_t*)dest)[4]= g[Y2];

			((uint8_t*)dest)[5]= b[Y2];

			dest+=6;

		}
