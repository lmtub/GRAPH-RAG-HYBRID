static inline int RENAME(yuv420_rgb15)(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

             int srcSliceH, uint8_t* dst[], int dstStride[]){

    int y, h_size;



    if(c->srcFormat == PIX_FMT_YUV422P){

	srcStride[1] *= 2;

	srcStride[2] *= 2;

    }



    h_size= (c->dstW+7)&~7;

    if(h_size*2 > FFABS(dstStride[0])) h_size-=8;



    __asm__ __volatile__ ("pxor %mm4, %mm4;" /* zero mm4 */ );

//printf("%X %X %X %X %X %X %X %X %X %X\n", (int)&c->redDither, (int)&b5Dither, (int)src[0], (int)src[1], (int)src[2], (int)dst[0],

//srcStride[0],srcStride[1],srcStride[2],dstStride[0]);

    for (y= 0; y<srcSliceH; y++ ) {

	uint8_t *_image = dst[0] + (y+srcSliceY)*dstStride[0];

	uint8_t *_py = src[0] + y*srcStride[0];

	uint8_t *_pu = src[1] + (y>>1)*srcStride[1];

	uint8_t *_pv = src[2] + (y>>1)*srcStride[2];

	long index= -h_size/2;



	b5Dither= dither8[y&1];

	g6Dither= dither4[y&1];

	g5Dither= dither8[y&1];

	r5Dither= dither8[(y+1)&1];

	    /* this mmx assembly code deals with SINGLE scan line at a time, it convert 8

	       pixels in each iteration */

	    __asm__ __volatile__ (

	/* load data for start of next scan line */

		     "movd (%2, %0), %%mm0;" /* Load 4 Cb 00 00 00 00 u3 u2 u1 u0 */

		     "movd (%3, %0), %%mm1;" /* Load 4 Cr 00 00 00 00 v3 v2 v1 v0 */

		     "movq (%5, %0, 2), %%mm6;" /* Load 8  Y Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0 */

//		    ".balign 16			\n\t"

		    "1:				\n\t"

YUV2RGB



#ifdef DITHER1XBPP

			"paddusb "MANGLE(b5Dither)", %%mm0	\n\t"

			"paddusb "MANGLE(g5Dither)", %%mm2	\n\t"

			"paddusb "MANGLE(r5Dither)", %%mm1	\n\t"

#endif



		     /* mask unneeded bits off */

		     "pand "MANGLE(mmx_redmask)", %%mm0;" /* b7b6b5b4 b3_0_0_0 b7b6b5b4 b3_0_0_0 */

		     "pand "MANGLE(mmx_redmask)", %%mm2;" /* g7g6g5g4 g3_0_0_0 g7g6g5g4 g3_0_0_0 */

		     "pand "MANGLE(mmx_redmask)", %%mm1;" /* r7r6r5r4 r3_0_0_0 r7r6r5r4 r3_0_0_0 */



		     "psrlw $3,%%mm0;" /* 0_0_0_b7 b6b5b4b3 0_0_0_b7 b6b5b4b3 */

		     "psrlw $1,%%mm1;"            /* 0_r7r6r5  r4r3_0_0 0_r7r6r5 r4r3_0_0 */

		     "pxor %%mm4, %%mm4;" /* zero mm4 */



		     "movq %%mm0, %%mm5;" /* Copy B7-B0 */

		     "movq %%mm2, %%mm7;" /* Copy G7-G0 */



		     /* convert rgb24 plane to rgb16 pack for pixel 0-3 */

		     "punpcklbw %%mm4, %%mm2;" /* 0_0_0_0 0_0_0_0 g7g6g5g4 g3_0_0_0 */

		     "punpcklbw %%mm1, %%mm0;" /* r7r6r5r4 r3_0_0_0 0_0_0_b7 b6b5b4b3 */



		     "psllw $2, %%mm2;" /* 0_0_0_0 0_0_g7g6 g5g4g3_0 0_0_0_0 */

		     "por %%mm2, %%mm0;" /* 0_r7r6r5 r4r3g7g6 g5g4g3b7 b6b5b4b3 */



		     "movq 8 (%5, %0, 2), %%mm6;" /* Load 8 Y Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0 */

		     MOVNTQ " %%mm0, (%1);" /* store pixel 0-3 */



		     /* convert rgb24 plane to rgb16 pack for pixel 0-3 */

		     "punpckhbw %%mm4, %%mm7;" /* 0_0_0_0 0_0_0_0 0_g7g6g5 g4g3_0_0 */

		     "punpckhbw %%mm1, %%mm5;" /* r7r6r5r4 r3_0_0_0 0_0_0_b7 b6b5b4b3 */



		     "psllw $2, %%mm7;" /* 0_0_0_0 0_0_g7g6 g5g4g3_0 0_0_0_0 */

		     "movd 4 (%2, %0), %%mm0;" /* Load 4 Cb 00 00 00 00 u3 u2 u1 u0 */



		     "por %%mm7, %%mm5;" /* 0_r7r6r5 r4r3g7g6 g5g4g3b7 b6b5b4b3 */

		     "movd 4 (%3, %0), %%mm1;" /* Load 4 Cr 00 00 00 00 v3 v2 v1 v0 */



		     MOVNTQ " %%mm5, 8 (%1);" /* store pixel 4-7 */



		     "add $16, %1			\n\t"

		     "add $4, %0			\n\t"

		     " js 1b				\n\t"

		     : "+r" (index), "+r" (_image)

		     : "r" (_pu - index), "r" (_pv - index), "r"(&c->redDither), "r" (_py - 2*index)

		     );

    }



    __asm__ __volatile__ (EMMS);

    return srcSliceH;

}
