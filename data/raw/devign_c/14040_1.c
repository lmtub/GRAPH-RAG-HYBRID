static inline void RENAME(yvu9_to_yuy2)(const uint8_t *src1, const uint8_t *src2, const uint8_t *src3,

			uint8_t *dst,

			unsigned width, unsigned height,

			int srcStride1, int srcStride2,

			int srcStride3, int dstStride)

{

    unsigned long y,x,w,h;

    w=width/2; h=height;

    for(y=0;y<h;y++){

	const uint8_t* yp=src1+srcStride1*y;

	const uint8_t* up=src2+srcStride2*(y>>2);

	const uint8_t* vp=src3+srcStride3*(y>>2);

	uint8_t* d=dst+dstStride*y;

	x=0;

#ifdef HAVE_MMX

	for(;x<w-7;x+=8)

	{

	    asm volatile(

		PREFETCH" 32(%1, %0)\n\t"

		PREFETCH" 32(%2, %0)\n\t"

		PREFETCH" 32(%3, %0)\n\t"

		"movq	(%1, %0, 4), %%mm0\n\t"       /* Y0Y1Y2Y3Y4Y5Y6Y7 */

		"movq	(%2, %0), %%mm1\n\t"       /* U0U1U2U3U4U5U6U7 */

		"movq	(%3, %0), %%mm2\n\t"	     /* V0V1V2V3V4V5V6V7 */

		"movq	%%mm0, %%mm3\n\t"    /* Y0Y1Y2Y3Y4Y5Y6Y7 */

		"movq	%%mm1, %%mm4\n\t"    /* U0U1U2U3U4U5U6U7 */

		"movq	%%mm2, %%mm5\n\t"    /* V0V1V2V3V4V5V6V7 */

		"punpcklbw %%mm1, %%mm1\n\t" /* U0U0 U1U1 U2U2 U3U3 */

		"punpcklbw %%mm2, %%mm2\n\t" /* V0V0 V1V1 V2V2 V3V3 */

		"punpckhbw %%mm4, %%mm4\n\t" /* U4U4 U5U5 U6U6 U7U7 */

		"punpckhbw %%mm5, %%mm5\n\t" /* V4V4 V5V5 V6V6 V7V7 */



		"movq	%%mm1, %%mm6\n\t"

		"punpcklbw %%mm2, %%mm1\n\t" /* U0V0 U0V0 U1V1 U1V1*/

		"punpcklbw %%mm1, %%mm0\n\t" /* Y0U0 Y1V0 Y2U0 Y3V0*/

		"punpckhbw %%mm1, %%mm3\n\t" /* Y4U1 Y5V1 Y6U1 Y7V1*/

		MOVNTQ"	%%mm0, (%4, %0, 8)\n\t"

		MOVNTQ"	%%mm3, 8(%4, %0, 8)\n\t"

		

		"punpckhbw %%mm2, %%mm6\n\t" /* U2V2 U2V2 U3V3 U3V3*/

		"movq	8(%1, %0, 4), %%mm0\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"punpcklbw %%mm6, %%mm0\n\t" /* Y U2 Y V2 Y U2 Y V2*/

		"punpckhbw %%mm6, %%mm3\n\t" /* Y U3 Y V3 Y U3 Y V3*/

		MOVNTQ"	%%mm0, 16(%4, %0, 8)\n\t"

		MOVNTQ"	%%mm3, 24(%4, %0, 8)\n\t"



		"movq	%%mm4, %%mm6\n\t"

		"movq	16(%1, %0, 4), %%mm0\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"punpcklbw %%mm5, %%mm4\n\t"

		"punpcklbw %%mm4, %%mm0\n\t" /* Y U4 Y V4 Y U4 Y V4*/

		"punpckhbw %%mm4, %%mm3\n\t" /* Y U5 Y V5 Y U5 Y V5*/

		MOVNTQ"	%%mm0, 32(%4, %0, 8)\n\t"

		MOVNTQ"	%%mm3, 40(%4, %0, 8)\n\t"

		

		"punpckhbw %%mm5, %%mm6\n\t"

		"movq	24(%1, %0, 4), %%mm0\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"punpcklbw %%mm6, %%mm0\n\t" /* Y U6 Y V6 Y U6 Y V6*/

		"punpckhbw %%mm6, %%mm3\n\t" /* Y U7 Y V7 Y U7 Y V7*/

		MOVNTQ"	%%mm0, 48(%4, %0, 8)\n\t"

		MOVNTQ"	%%mm3, 56(%4, %0, 8)\n\t"



		: "+r" (x)

                : "r"(yp), "r" (up), "r"(vp), "r"(d)

		:"memory");

	}

#endif

	for(; x<w; x++)

	{

	    const int x2= x<<2;

	    d[8*x+0]=yp[x2];

	    d[8*x+1]=up[x];

	    d[8*x+2]=yp[x2+1];

	    d[8*x+3]=vp[x];

	    d[8*x+4]=yp[x2+2];

	    d[8*x+5]=up[x];

	    d[8*x+6]=yp[x2+3];

	    d[8*x+7]=vp[x];

	}

    }

#ifdef HAVE_MMX

	asm(

		EMMS" \n\t"

		SFENCE" \n\t"

		::: "memory"

		);

#endif

}
