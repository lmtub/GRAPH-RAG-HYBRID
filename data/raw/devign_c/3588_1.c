static inline void RENAME(yuv2yuvX)(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, int dstW, int chrDstW)

{

#ifdef HAVE_MMX

	if(uDest != NULL)

	{

		asm volatile(

				YSCALEYUV2YV12X(0, CHR_MMX_FILTER_OFFSET)

				:: "r" (&c->redDither),

				"r" (uDest), "p" ((long)chrDstW)

				: "%"REG_a, "%"REG_d, "%"REG_S

			);



		asm volatile(

				YSCALEYUV2YV12X(4096, CHR_MMX_FILTER_OFFSET)

				:: "r" (&c->redDither),

				"r" (vDest), "p" ((long)chrDstW)

				: "%"REG_a, "%"REG_d, "%"REG_S

			);

	}



	asm volatile(

			YSCALEYUV2YV12X(0, LUM_MMX_FILTER_OFFSET)

			:: "r" (&c->redDither),

			   "r" (dest), "p" ((long)dstW)

			: "%"REG_a, "%"REG_d, "%"REG_S

		);

#else

#ifdef HAVE_ALTIVEC

yuv2yuvX_altivec_real(lumFilter, lumSrc, lumFilterSize,

		      chrFilter, chrSrc, chrFilterSize,

		      dest, uDest, vDest, dstW, chrDstW);

#else //HAVE_ALTIVEC

yuv2yuvXinC(lumFilter, lumSrc, lumFilterSize,

	    chrFilter, chrSrc, chrFilterSize,

	    dest, uDest, vDest, dstW, chrDstW);

#endif //!HAVE_ALTIVEC

#endif

}
