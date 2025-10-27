static inline int RENAME(yuv420_rgb24)(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

             int srcSliceH, uint8_t* dst[], int dstStride[]){

    int y, h_size;



    if(c->srcFormat == PIX_FMT_YUV422P){

	srcStride[1] *= 2;

	srcStride[2] *= 2;

    }



    h_size= (c->dstW+7)&~7;

    if(h_size*3 > FFABS(dstStride[0])) h_size-=8;



    __asm__ __volatile__ ("pxor %mm4, %mm4;" /* zero mm4 */ );



    for (y= 0; y<srcSliceH; y++ ) {

	uint8_t *_image = dst[0] + (y+srcSliceY)*dstStride[0];

	uint8_t *_py = src[0] + y*srcStride[0];

	uint8_t *_pu = src[1] + (y>>1)*srcStride[1];

	uint8_t *_pv = src[2] + (y>>1)*srcStride[2];

	long index= -h_size/2;



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

	/* mm0=B, %%mm2=G, %%mm1=R */

#ifdef HAVE_MMX2

			"movq "MANGLE(M24A)", %%mm4	\n\t"

			"movq "MANGLE(M24C)", %%mm7	\n\t"

			"pshufw $0x50, %%mm0, %%mm5	\n\t" /* B3 B2 B3 B2  B1 B0 B1 B0 */

			"pshufw $0x50, %%mm2, %%mm3	\n\t" /* G3 G2 G3 G2  G1 G0 G1 G0 */

			"pshufw $0x00, %%mm1, %%mm6	\n\t" /* R1 R0 R1 R0  R1 R0 R1 R0 */



			"pand %%mm4, %%mm5		\n\t" /*    B2        B1       B0 */

			"pand %%mm4, %%mm3		\n\t" /*    G2        G1       G0 */

			"pand %%mm7, %%mm6		\n\t" /*       R1        R0       */



			"psllq $8, %%mm3		\n\t" /* G2        G1       G0    */

			"por %%mm5, %%mm6		\n\t"

			"por %%mm3, %%mm6		\n\t"

			MOVNTQ" %%mm6, (%1)		\n\t"



			"psrlq $8, %%mm2		\n\t" /* 00 G7 G6 G5  G4 G3 G2 G1 */

			"pshufw $0xA5, %%mm0, %%mm5	\n\t" /* B5 B4 B5 B4  B3 B2 B3 B2 */

			"pshufw $0x55, %%mm2, %%mm3	\n\t" /* G4 G3 G4 G3  G4 G3 G4 G3 */

			"pshufw $0xA5, %%mm1, %%mm6	\n\t" /* R5 R4 R5 R4  R3 R2 R3 R2 */



			"pand "MANGLE(M24B)", %%mm5	\n\t" /* B5       B4        B3    */

			"pand %%mm7, %%mm3		\n\t" /*       G4        G3       */

			"pand %%mm4, %%mm6		\n\t" /*    R4        R3       R2 */



			"por %%mm5, %%mm3		\n\t" /* B5    G4 B4     G3 B3    */

			"por %%mm3, %%mm6		\n\t"

			MOVNTQ" %%mm6, 8(%1)		\n\t"



			"pshufw $0xFF, %%mm0, %%mm5	\n\t" /* B7 B6 B7 B6  B7 B6 B6 B7 */

			"pshufw $0xFA, %%mm2, %%mm3	\n\t" /* 00 G7 00 G7  G6 G5 G6 G5 */

			"pshufw $0xFA, %%mm1, %%mm6	\n\t" /* R7 R6 R7 R6  R5 R4 R5 R4 */

			"movd 4 (%2, %0), %%mm0;" /* Load 4 Cb 00 00 00 00 u3 u2 u1 u0 */



			"pand %%mm7, %%mm5		\n\t" /*       B7        B6       */

			"pand %%mm4, %%mm3		\n\t" /*    G7        G6       G5 */

			"pand "MANGLE(M24B)", %%mm6	\n\t" /* R7       R6        R5    */

			"movd 4 (%3, %0), %%mm1;" /* Load 4 Cr 00 00 00 00 v3 v2 v1 v0 */

\

			"por %%mm5, %%mm3		\n\t"

			"por %%mm3, %%mm6		\n\t"

			MOVNTQ" %%mm6, 16(%1)		\n\t"

			"movq 8 (%5, %0, 2), %%mm6;" /* Load 8 Y Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0 */

			"pxor %%mm4, %%mm4		\n\t"



#else



			"pxor %%mm4, %%mm4		\n\t"

			"movq %%mm0, %%mm5		\n\t" /* B */

			"movq %%mm1, %%mm6		\n\t" /* R */

			"punpcklbw %%mm2, %%mm0		\n\t" /* GBGBGBGB 0 */

			"punpcklbw %%mm4, %%mm1		\n\t" /* 0R0R0R0R 0 */

			"punpckhbw %%mm2, %%mm5		\n\t" /* GBGBGBGB 2 */

			"punpckhbw %%mm4, %%mm6		\n\t" /* 0R0R0R0R 2 */

			"movq %%mm0, %%mm7		\n\t" /* GBGBGBGB 0 */

			"movq %%mm5, %%mm3		\n\t" /* GBGBGBGB 2 */

			"punpcklwd %%mm1, %%mm7		\n\t" /* 0RGB0RGB 0 */

			"punpckhwd %%mm1, %%mm0		\n\t" /* 0RGB0RGB 1 */

			"punpcklwd %%mm6, %%mm5		\n\t" /* 0RGB0RGB 2 */

			"punpckhwd %%mm6, %%mm3		\n\t" /* 0RGB0RGB 3 */



			"movq %%mm7, %%mm2		\n\t" /* 0RGB0RGB 0 */

			"movq %%mm0, %%mm6		\n\t" /* 0RGB0RGB 1 */

			"movq %%mm5, %%mm1		\n\t" /* 0RGB0RGB 2 */

			"movq %%mm3, %%mm4		\n\t" /* 0RGB0RGB 3 */



			"psllq $40, %%mm7		\n\t" /* RGB00000 0 */

			"psllq $40, %%mm0		\n\t" /* RGB00000 1 */

			"psllq $40, %%mm5		\n\t" /* RGB00000 2 */

			"psllq $40, %%mm3		\n\t" /* RGB00000 3 */



			"punpckhdq %%mm2, %%mm7		\n\t" /* 0RGBRGB0 0 */

			"punpckhdq %%mm6, %%mm0		\n\t" /* 0RGBRGB0 1 */

			"punpckhdq %%mm1, %%mm5		\n\t" /* 0RGBRGB0 2 */

			"punpckhdq %%mm4, %%mm3		\n\t" /* 0RGBRGB0 3 */



			"psrlq $8, %%mm7		\n\t" /* 00RGBRGB 0 */

			"movq %%mm0, %%mm6		\n\t" /* 0RGBRGB0 1 */

			"psllq $40, %%mm0		\n\t" /* GB000000 1 */

			"por %%mm0, %%mm7		\n\t" /* GBRGBRGB 0 */

			MOVNTQ" %%mm7, (%1)		\n\t"



			"movd 4 (%2, %0), %%mm0;" /* Load 4 Cb 00 00 00 00 u3 u2 u1 u0 */



			"psrlq $24, %%mm6		\n\t" /* 0000RGBR 1 */

			"movq %%mm5, %%mm1		\n\t" /* 0RGBRGB0 2 */

			"psllq $24, %%mm5		\n\t" /* BRGB0000 2 */

			"por %%mm5, %%mm6		\n\t" /* BRGBRGBR 1 */

			MOVNTQ" %%mm6, 8(%1)		\n\t"



			"movq 8 (%5, %0, 2), %%mm6;" /* Load 8 Y Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0 */



			"psrlq $40, %%mm1		\n\t" /* 000000RG 2 */

			"psllq $8, %%mm3		\n\t" /* RGBRGB00 3 */

			"por %%mm3, %%mm1		\n\t" /* RGBRGBRG 2 */

			MOVNTQ" %%mm1, 16(%1)		\n\t"



			"movd 4 (%3, %0), %%mm1;" /* Load 4 Cr 00 00 00 00 v3 v2 v1 v0 */

			"pxor %%mm4, %%mm4		\n\t"

#endif



		     "add $24, %1			\n\t"

		     "add $4, %0			\n\t"

		     " js 1b				\n\t"



		     : "+r" (index), "+r" (_image)

		     : "r" (_pu - index), "r" (_pv - index), "r"(&c->redDither), "r" (_py - 2*index)

		     );

    }



    __asm__ __volatile__ (EMMS);

    return srcSliceH;

}
