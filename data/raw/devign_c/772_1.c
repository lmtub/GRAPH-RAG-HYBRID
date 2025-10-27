static inline void RENAME(bgr24ToY)(uint8_t *dst, uint8_t *src, long width)

{

#ifdef HAVE_MMX

	asm volatile(

		"mov %2, %%"REG_a"		\n\t"

		"movq "MANGLE(bgr2YCoeff)", %%mm6		\n\t"

		"movq "MANGLE(w1111)", %%mm5		\n\t"

		"pxor %%mm7, %%mm7		\n\t"

		"lea (%%"REG_a", %%"REG_a", 2), %%"REG_d"\n\t"

		ASMALIGN(4)

		"1:				\n\t"

		PREFETCH" 64(%0, %%"REG_d")	\n\t"

		"movd (%0, %%"REG_d"), %%mm0	\n\t"

		"movd 3(%0, %%"REG_d"), %%mm1	\n\t"

		"punpcklbw %%mm7, %%mm0		\n\t"

		"punpcklbw %%mm7, %%mm1		\n\t"

		"movd 6(%0, %%"REG_d"), %%mm2	\n\t"

		"movd 9(%0, %%"REG_d"), %%mm3	\n\t"

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



		"movd 12(%0, %%"REG_d"), %%mm4	\n\t"

		"movd 15(%0, %%"REG_d"), %%mm1	\n\t"

		"punpcklbw %%mm7, %%mm4		\n\t"

		"punpcklbw %%mm7, %%mm1		\n\t"

		"movd 18(%0, %%"REG_d"), %%mm2	\n\t"

		"movd 21(%0, %%"REG_d"), %%mm3	\n\t"

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

		"add $24, %%"REG_d"		\n\t"

		"packssdw %%mm2, %%mm4		\n\t"

		"psraw $7, %%mm4		\n\t"



		"packuswb %%mm4, %%mm0		\n\t"

		"paddusb "MANGLE(bgr2YOffset)", %%mm0	\n\t"



		"movq %%mm0, (%1, %%"REG_a")	\n\t"

		"add $8, %%"REG_a"		\n\t"

		" js 1b				\n\t"

		: : "r" (src+width*3), "r" (dst+width), "g" (-width)

		: "%"REG_a, "%"REG_d

	);

#else

	int i;

	for(i=0; i<width; i++)

	{

		int b= src[i*3+0];

		int g= src[i*3+1];

		int r= src[i*3+2];



		dst[i]= ((RY*r + GY*g + BY*b + (33<<(RGB2YUV_SHIFT-1)) )>>RGB2YUV_SHIFT);

	}

#endif

}
