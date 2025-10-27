static inline void RENAME(rgb24toyv12)(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

	unsigned int width, unsigned int height,

	int lumStride, int chromStride, int srcStride)

{

	unsigned y;

	const unsigned chromWidth= width>>1;

#ifdef HAVE_MMX

	for(y=0; y<height-2; y+=2)

	{

		unsigned i;

		for(i=0; i<2; i++)

		{

			asm volatile(

				"mov %2, %%"REG_a"		\n\t"

				"movq "MANGLE(bgr2YCoeff)", %%mm6		\n\t"

				"movq "MANGLE(w1111)", %%mm5		\n\t"

				"pxor %%mm7, %%mm7		\n\t"

				"lea (%%"REG_a", %%"REG_a", 2), %%"REG_b"\n\t"

				".balign 16			\n\t"

				"1:				\n\t"

				PREFETCH" 64(%0, %%"REG_b")	\n\t"

				"movd (%0, %%"REG_b"), %%mm0	\n\t"

				"movd 3(%0, %%"REG_b"), %%mm1	\n\t"

				"punpcklbw %%mm7, %%mm0		\n\t"

				"punpcklbw %%mm7, %%mm1		\n\t"

				"movd 6(%0, %%"REG_b"), %%mm2	\n\t"

				"movd 9(%0, %%"REG_b"), %%mm3	\n\t"

				"punpcklbw %%mm7, %%mm2		\n\t"

				"punpcklbw %%mm7, %%mm3		\n\t"

				"pmaddwd %%mm6, %%mm0		\n\t"

				"pmaddwd %%mm6, %%mm1		\n\t"

				"pmaddwd %%mm6, %%mm2		\n\t"

				"pmaddwd %%mm6, %%mm3		\n\t"

#ifndef FAST_BGR2YV12

				"psrad $8, %%mm0		\n\t"

				"psrad $8, %%mm1		\n\t"

				"psrad $8, %%mm2		\n\t"

				"psrad $8, %%mm3		\n\t"

#endif

				"packssdw %%mm1, %%mm0		\n\t"

				"packssdw %%mm3, %%mm2		\n\t"

				"pmaddwd %%mm5, %%mm0		\n\t"

				"pmaddwd %%mm5, %%mm2		\n\t"

				"packssdw %%mm2, %%mm0		\n\t"

				"psraw $7, %%mm0		\n\t"



				"movd 12(%0, %%"REG_b"), %%mm4	\n\t"

				"movd 15(%0, %%"REG_b"), %%mm1	\n\t"

				"punpcklbw %%mm7, %%mm4		\n\t"

				"punpcklbw %%mm7, %%mm1		\n\t"

				"movd 18(%0, %%"REG_b"), %%mm2	\n\t"

				"movd 21(%0, %%"REG_b"), %%mm3	\n\t"

				"punpcklbw %%mm7, %%mm2		\n\t"

				"punpcklbw %%mm7, %%mm3		\n\t"

				"pmaddwd %%mm6, %%mm4		\n\t"

				"pmaddwd %%mm6, %%mm1		\n\t"

				"pmaddwd %%mm6, %%mm2		\n\t"

				"pmaddwd %%mm6, %%mm3		\n\t"

#ifndef FAST_BGR2YV12

				"psrad $8, %%mm4		\n\t"

				"psrad $8, %%mm1		\n\t"

				"psrad $8, %%mm2		\n\t"

				"psrad $8, %%mm3		\n\t"

#endif

				"packssdw %%mm1, %%mm4		\n\t"

				"packssdw %%mm3, %%mm2		\n\t"

				"pmaddwd %%mm5, %%mm4		\n\t"

				"pmaddwd %%mm5, %%mm2		\n\t"

				"add $24, %%"REG_b"		\n\t"

				"packssdw %%mm2, %%mm4		\n\t"

				"psraw $7, %%mm4		\n\t"



				"packuswb %%mm4, %%mm0		\n\t"

				"paddusb "MANGLE(bgr2YOffset)", %%mm0	\n\t"



				MOVNTQ" %%mm0, (%1, %%"REG_a")	\n\t"

				"add $8, %%"REG_a"		\n\t"

				" js 1b				\n\t"

				: : "r" (src+width*3), "r" (ydst+width), "g" ((long)-width)

				: "%"REG_a, "%"REG_b

			);

			ydst += lumStride;

			src  += srcStride;

		}

		src -= srcStride*2;

		asm volatile(

			"mov %4, %%"REG_a"		\n\t"

			"movq "MANGLE(w1111)", %%mm5		\n\t"

			"movq "MANGLE(bgr2UCoeff)", %%mm6		\n\t"

			"pxor %%mm7, %%mm7		\n\t"

			"lea (%%"REG_a", %%"REG_a", 2), %%"REG_b"\n\t"

			"add %%"REG_b", %%"REG_b"	\n\t"

			".balign 16			\n\t"

			"1:				\n\t"

			PREFETCH" 64(%0, %%"REG_b")	\n\t"

			PREFETCH" 64(%1, %%"REG_b")	\n\t"

#if defined (HAVE_MMX2) || defined (HAVE_3DNOW)

			"movq (%0, %%"REG_b"), %%mm0	\n\t"

			"movq (%1, %%"REG_b"), %%mm1	\n\t"

			"movq 6(%0, %%"REG_b"), %%mm2	\n\t"

			"movq 6(%1, %%"REG_b"), %%mm3	\n\t"

			PAVGB" %%mm1, %%mm0		\n\t"

			PAVGB" %%mm3, %%mm2		\n\t"

			"movq %%mm0, %%mm1		\n\t"

			"movq %%mm2, %%mm3		\n\t"

			"psrlq $24, %%mm0		\n\t"

			"psrlq $24, %%mm2		\n\t"

			PAVGB" %%mm1, %%mm0		\n\t"

			PAVGB" %%mm3, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm0		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

#else

			"movd (%0, %%"REG_b"), %%mm0	\n\t"

			"movd (%1, %%"REG_b"), %%mm1	\n\t"

			"movd 3(%0, %%"REG_b"), %%mm2	\n\t"

			"movd 3(%1, %%"REG_b"), %%mm3	\n\t"

			"punpcklbw %%mm7, %%mm0		\n\t"

			"punpcklbw %%mm7, %%mm1		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm3		\n\t"

			"paddw %%mm1, %%mm0		\n\t"

			"paddw %%mm3, %%mm2		\n\t"

			"paddw %%mm2, %%mm0		\n\t"

			"movd 6(%0, %%"REG_b"), %%mm4	\n\t"

			"movd 6(%1, %%"REG_b"), %%mm1	\n\t"

			"movd 9(%0, %%"REG_b"), %%mm2	\n\t"

			"movd 9(%1, %%"REG_b"), %%mm3	\n\t"

			"punpcklbw %%mm7, %%mm4		\n\t"

			"punpcklbw %%mm7, %%mm1		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm3		\n\t"

			"paddw %%mm1, %%mm4		\n\t"

			"paddw %%mm3, %%mm2		\n\t"

			"paddw %%mm4, %%mm2		\n\t"

			"psrlw $2, %%mm0		\n\t"

			"psrlw $2, %%mm2		\n\t"

#endif

			"movq "MANGLE(bgr2VCoeff)", %%mm1		\n\t"

			"movq "MANGLE(bgr2VCoeff)", %%mm3		\n\t"



			"pmaddwd %%mm0, %%mm1		\n\t"

			"pmaddwd %%mm2, %%mm3		\n\t"

			"pmaddwd %%mm6, %%mm0		\n\t"

			"pmaddwd %%mm6, %%mm2		\n\t"

#ifndef FAST_BGR2YV12

			"psrad $8, %%mm0		\n\t"

			"psrad $8, %%mm1		\n\t"

			"psrad $8, %%mm2		\n\t"

			"psrad $8, %%mm3		\n\t"

#endif

			"packssdw %%mm2, %%mm0		\n\t"

			"packssdw %%mm3, %%mm1		\n\t"

			"pmaddwd %%mm5, %%mm0		\n\t"

			"pmaddwd %%mm5, %%mm1		\n\t"

			"packssdw %%mm1, %%mm0		\n\t" // V1 V0 U1 U0

			"psraw $7, %%mm0		\n\t"



#if defined (HAVE_MMX2) || defined (HAVE_3DNOW)

			"movq 12(%0, %%"REG_b"), %%mm4	\n\t"

			"movq 12(%1, %%"REG_b"), %%mm1	\n\t"

			"movq 18(%0, %%"REG_b"), %%mm2	\n\t"

			"movq 18(%1, %%"REG_b"), %%mm3	\n\t"

			PAVGB" %%mm1, %%mm4		\n\t"

			PAVGB" %%mm3, %%mm2		\n\t"

			"movq %%mm4, %%mm1		\n\t"

			"movq %%mm2, %%mm3		\n\t"

			"psrlq $24, %%mm4		\n\t"

			"psrlq $24, %%mm2		\n\t"

			PAVGB" %%mm1, %%mm4		\n\t"

			PAVGB" %%mm3, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm4		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

#else

			"movd 12(%0, %%"REG_b"), %%mm4	\n\t"

			"movd 12(%1, %%"REG_b"), %%mm1	\n\t"

			"movd 15(%0, %%"REG_b"), %%mm2	\n\t"

			"movd 15(%1, %%"REG_b"), %%mm3	\n\t"

			"punpcklbw %%mm7, %%mm4		\n\t"

			"punpcklbw %%mm7, %%mm1		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm3		\n\t"

			"paddw %%mm1, %%mm4		\n\t"

			"paddw %%mm3, %%mm2		\n\t"

			"paddw %%mm2, %%mm4		\n\t"

			"movd 18(%0, %%"REG_b"), %%mm5	\n\t"

			"movd 18(%1, %%"REG_b"), %%mm1	\n\t"

			"movd 21(%0, %%"REG_b"), %%mm2	\n\t"

			"movd 21(%1, %%"REG_b"), %%mm3	\n\t"

			"punpcklbw %%mm7, %%mm5		\n\t"

			"punpcklbw %%mm7, %%mm1		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"punpcklbw %%mm7, %%mm3		\n\t"

			"paddw %%mm1, %%mm5		\n\t"

			"paddw %%mm3, %%mm2		\n\t"

			"paddw %%mm5, %%mm2		\n\t"

			"movq "MANGLE(w1111)", %%mm5		\n\t"

			"psrlw $2, %%mm4		\n\t"

			"psrlw $2, %%mm2		\n\t"

#endif

			"movq "MANGLE(bgr2VCoeff)", %%mm1		\n\t"

			"movq "MANGLE(bgr2VCoeff)", %%mm3		\n\t"



			"pmaddwd %%mm4, %%mm1		\n\t"

			"pmaddwd %%mm2, %%mm3		\n\t"

			"pmaddwd %%mm6, %%mm4		\n\t"

			"pmaddwd %%mm6, %%mm2		\n\t"

#ifndef FAST_BGR2YV12

			"psrad $8, %%mm4		\n\t"

			"psrad $8, %%mm1		\n\t"

			"psrad $8, %%mm2		\n\t"

			"psrad $8, %%mm3		\n\t"

#endif

			"packssdw %%mm2, %%mm4		\n\t"

			"packssdw %%mm3, %%mm1		\n\t"

			"pmaddwd %%mm5, %%mm4		\n\t"

			"pmaddwd %%mm5, %%mm1		\n\t"

			"add $24, %%"REG_b"		\n\t"

			"packssdw %%mm1, %%mm4		\n\t" // V3 V2 U3 U2

			"psraw $7, %%mm4		\n\t"



			"movq %%mm0, %%mm1		\n\t"

			"punpckldq %%mm4, %%mm0		\n\t"

			"punpckhdq %%mm4, %%mm1		\n\t"

			"packsswb %%mm1, %%mm0		\n\t"

			"paddb "MANGLE(bgr2UVOffset)", %%mm0	\n\t"

			"movd %%mm0, (%2, %%"REG_a")	\n\t"

			"punpckhdq %%mm0, %%mm0		\n\t"

			"movd %%mm0, (%3, %%"REG_a")	\n\t"

			"add $4, %%"REG_a"		\n\t"

			" js 1b				\n\t"

			: : "r" (src+chromWidth*6), "r" (src+srcStride+chromWidth*6), "r" (udst+chromWidth), "r" (vdst+chromWidth), "g" ((long)-chromWidth)

			: "%"REG_a, "%"REG_b

		);



		udst += chromStride;

		vdst += chromStride;

		src  += srcStride*2;

	}



	asm volatile(   EMMS" \n\t"

			SFENCE" \n\t"

			:::"memory");

#else

	y=0;

#endif

	for(; y<height; y+=2)

	{

		unsigned i;

		for(i=0; i<chromWidth; i++)

		{

			unsigned int b= src[6*i+0];

			unsigned int g= src[6*i+1];

			unsigned int r= src[6*i+2];



			unsigned int Y  =  ((RY*r + GY*g + BY*b)>>RGB2YUV_SHIFT) + 16;

			unsigned int V  =  ((RV*r + GV*g + BV*b)>>RGB2YUV_SHIFT) + 128;

			unsigned int U  =  ((RU*r + GU*g + BU*b)>>RGB2YUV_SHIFT) + 128;



			udst[i] 	= U;

			vdst[i] 	= V;

			ydst[2*i] 	= Y;



			b= src[6*i+3];

			g= src[6*i+4];

			r= src[6*i+5];



			Y  =  ((RY*r + GY*g + BY*b)>>RGB2YUV_SHIFT) + 16;

			ydst[2*i+1] 	= Y;

		}

		ydst += lumStride;

		src  += srcStride;



		for(i=0; i<chromWidth; i++)

		{

			unsigned int b= src[6*i+0];

			unsigned int g= src[6*i+1];

			unsigned int r= src[6*i+2];



			unsigned int Y  =  ((RY*r + GY*g + BY*b)>>RGB2YUV_SHIFT) + 16;



			ydst[2*i] 	= Y;



			b= src[6*i+3];

			g= src[6*i+4];

			r= src[6*i+5];



			Y  =  ((RY*r + GY*g + BY*b)>>RGB2YUV_SHIFT) + 16;

			ydst[2*i+1] 	= Y;

		}

		udst += chromStride;

		vdst += chromStride;

		ydst += lumStride;

		src  += srcStride;

	}

}
