static inline void RENAME(yuv2yuvX)(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, long dstW, long chrDstW)

{

#ifdef HAVE_MMX

        if(c->flags & SWS_ACCURATE_RND){

                if(uDest){

                        YSCALEYUV2YV12X_ACCURATE(   0, CHR_MMX_FILTER_OFFSET, uDest, chrDstW)

                        YSCALEYUV2YV12X_ACCURATE(4096, CHR_MMX_FILTER_OFFSET, vDest, chrDstW)

                }



                YSCALEYUV2YV12X_ACCURATE(0, LUM_MMX_FILTER_OFFSET, dest, dstW)

        }else{

                if(uDest){

                        YSCALEYUV2YV12X(   0, CHR_MMX_FILTER_OFFSET, uDest, chrDstW)

                        YSCALEYUV2YV12X(4096, CHR_MMX_FILTER_OFFSET, vDest, chrDstW)

                }



                YSCALEYUV2YV12X(0, LUM_MMX_FILTER_OFFSET, dest, dstW)

        }

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
