static inline void RENAME(yuv2packedX)(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

			    uint8_t *dest, long dstW, long dstY)

{

	long dummy=0;

	switch(c->dstFormat)

	{

#ifdef HAVE_MMX

	case IMGFMT_BGR32:

		{

			asm volatile(

				YSCALEYUV2RGBX

				WRITEBGR32(%4, %5, %%REGa)



			:: "r" (&c->redDither), 

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_d, "%"REG_S

			);

		}

		break;

	case IMGFMT_BGR24:

		{

			asm volatile(

				YSCALEYUV2RGBX

				"lea (%%"REG_a", %%"REG_a", 2), %%"REG_b"\n\t" //FIXME optimize

				"add %4, %%"REG_b"			\n\t"

				WRITEBGR24(%%REGb, %5, %%REGa)



			:: "r" (&c->redDither), 

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_b, "%"REG_d, "%"REG_S //FIXME ebx

			);

		}

		break;

	case IMGFMT_BGR15:

		{

			asm volatile(

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g5Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR15(%4, %5, %%REGa)



			:: "r" (&c->redDither), 

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_d, "%"REG_S

			);

		}

		break;

	case IMGFMT_BGR16:

		{

			asm volatile(

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g6Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR16(%4, %5, %%REGa)



			:: "r" (&c->redDither), 

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_d, "%"REG_S

			);

		}

		break;

	case IMGFMT_YUY2:

		{

			asm volatile(

				YSCALEYUV2PACKEDX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */



				"psraw $3, %%mm3		\n\t"

				"psraw $3, %%mm4		\n\t"

				"psraw $3, %%mm1		\n\t"

				"psraw $3, %%mm7		\n\t"

				WRITEYUY2(%4, %5, %%REGa)



			:: "r" (&c->redDither), 

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_d, "%"REG_S

			);

		}

		break;

#endif

	default:

#ifdef HAVE_ALTIVEC

		/* The following list of supported dstFormat values should

		   match what's found in the body of altivec_yuv2packedX() */

		if(c->dstFormat==IMGFMT_ABGR  || c->dstFormat==IMGFMT_BGRA  ||

		   c->dstFormat==IMGFMT_BGR24 || c->dstFormat==IMGFMT_RGB24 ||

		   c->dstFormat==IMGFMT_RGBA  || c->dstFormat==IMGFMT_ARGB)

			altivec_yuv2packedX (c, lumFilter, lumSrc, lumFilterSize,

				    chrFilter, chrSrc, chrFilterSize,

				    dest, dstW, dstY);

		else

#endif

			yuv2packedXinC(c, lumFilter, lumSrc, lumFilterSize,

				    chrFilter, chrSrc, chrFilterSize,

				    dest, dstW, dstY);

		break;

	}

}
