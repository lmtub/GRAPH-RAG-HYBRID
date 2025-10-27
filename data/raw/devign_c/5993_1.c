static inline int RENAME(yuv420_rgb32)(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

             int srcSliceH, uint8_t* dst[], int dstStride[]){

    int y, h_size;



    if(c->srcFormat == PIX_FMT_YUV422P){

	srcStride[1] *= 2;

	srcStride[2] *= 2;

    }



    h_size= (c->dstW+7)&~7;

    if(h_size*4 > FFABS(dstStride[0])) h_size-=8;



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

		     /* convert RGB plane to RGB packed format,

			mm0 -> B, mm1 -> R, mm2 -> G, mm3 -> 0,

			mm4 -> GB, mm5 -> AR pixel 4-7,

			mm6 -> GB, mm7 -> AR pixel 0-3 */

		     "pxor %%mm3, %%mm3;" /* zero mm3 */



		     "movq %%mm0, %%mm6;" /* B7 B6 B5 B4 B3 B2 B1 B0 */

		     "movq %%mm1, %%mm7;" /* R7 R6 R5 R4 R3 R2 R1 R0 */



		     "movq %%mm0, %%mm4;" /* B7 B6 B5 B4 B3 B2 B1 B0 */

		     "movq %%mm1, %%mm5;" /* R7 R6 R5 R4 R3 R2 R1 R0 */



		     "punpcklbw %%mm2, %%mm6;" /* G3 B3 G2 B2 G1 B1 G0 B0 */

		     "punpcklbw %%mm3, %%mm7;" /* 00 R3 00 R2 00 R1 00 R0 */



		     "punpcklwd %%mm7, %%mm6;" /* 00 R1 B1 G1 00 R0 B0 G0 */

		     MOVNTQ " %%mm6, (%1);" /* Store ARGB1 ARGB0 */



		     "movq %%mm0, %%mm6;" /* B7 B6 B5 B4 B3 B2 B1 B0 */

		     "punpcklbw %%mm2, %%mm6;" /* G3 B3 G2 B2 G1 B1 G0 B0 */



		     "punpckhwd %%mm7, %%mm6;" /* 00 R3 G3 B3 00 R2 B3 G2 */

		     MOVNTQ " %%mm6, 8 (%1);" /* Store ARGB3 ARGB2 */



		     "punpckhbw %%mm2, %%mm4;" /* G7 B7 G6 B6 G5 B5 G4 B4 */

		     "punpckhbw %%mm3, %%mm5;" /* 00 R7 00 R6 00 R5 00 R4 */



		     "punpcklwd %%mm5, %%mm4;" /* 00 R5 B5 G5 00 R4 B4 G4 */

		     MOVNTQ " %%mm4, 16 (%1);" /* Store ARGB5 ARGB4 */



		     "movq %%mm0, %%mm4;" /* B7 B6 B5 B4 B3 B2 B1 B0 */

		     "punpckhbw %%mm2, %%mm4;" /* G7 B7 G6 B6 G5 B5 G4 B4 */



		     "punpckhwd %%mm5, %%mm4;" /* 00 R7 G7 B7 00 R6 B6 G6 */

		     MOVNTQ " %%mm4, 24 (%1);" /* Store ARGB7 ARGB6 */



		     "movd 4 (%2, %0), %%mm0;" /* Load 4 Cb 00 00 00 00 u3 u2 u1 u0 */

		     "movd 4 (%3, %0), %%mm1;" /* Load 4 Cr 00 00 00 00 v3 v2 v1 v0 */



		     "pxor %%mm4, %%mm4;" /* zero mm4 */

		     "movq 8 (%5, %0, 2), %%mm6;" /* Load 8 Y Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0 */



		     "add $32, %1			\n\t"

		     "add $4, %0			\n\t"

		     " js 1b				\n\t"



		     : "+r" (index), "+r" (_image)

		     : "r" (_pu - index), "r" (_pv - index), "r"(&c->redDither), "r" (_py - 2*index)

		     );

    }



    __asm__ __volatile__ (EMMS);

    return srcSliceH;

}
