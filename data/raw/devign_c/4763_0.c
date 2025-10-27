static inline void RENAME(hScale)(int16_t *dst, int dstW, uint8_t *src, int srcW, int xInc,

				  int16_t *filter, int16_t *filterPos, long filterSize)

{

#ifdef HAVE_MMX

	assert(filterSize % 4 == 0 && filterSize>0);

	if(filterSize==4) // allways true for upscaling, sometimes for down too

	{

		long counter= -2*dstW;

		filter-= counter*2;

		filterPos-= counter/2;

		dst-= counter/2;

		asm volatile(

			"pxor %%mm7, %%mm7		\n\t"

			"movq "MANGLE(w02)", %%mm6	\n\t"

			"push %%"REG_BP"		\n\t" // we use 7 regs here ...

			"mov %%"REG_a", %%"REG_BP"	\n\t"

			ASMALIGN16

			"1:				\n\t"

			"movzwl (%2, %%"REG_BP"), %%eax	\n\t"

			"movzwl 2(%2, %%"REG_BP"), %%ebx\n\t"

			"movq (%1, %%"REG_BP", 4), %%mm1\n\t"

			"movq 8(%1, %%"REG_BP", 4), %%mm3\n\t"

			"movd (%3, %%"REG_a"), %%mm0	\n\t"

			"movd (%3, %%"REG_b"), %%mm2	\n\t"

			"punpcklbw %%mm7, %%mm0		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"pmaddwd %%mm1, %%mm0		\n\t"

			"pmaddwd %%mm2, %%mm3		\n\t"

			"psrad $8, %%mm0		\n\t"

			"psrad $8, %%mm3		\n\t"

			"packssdw %%mm3, %%mm0		\n\t"

			"pmaddwd %%mm6, %%mm0		\n\t"

			"packssdw %%mm0, %%mm0		\n\t"

			"movd %%mm0, (%4, %%"REG_BP")	\n\t"

			"add $4, %%"REG_BP"		\n\t"

			" jnc 1b			\n\t"



			"pop %%"REG_BP"			\n\t"

			: "+a" (counter)

			: "c" (filter), "d" (filterPos), "S" (src), "D" (dst)

			: "%"REG_b

		);

	}

	else if(filterSize==8)

	{

		long counter= -2*dstW;

		filter-= counter*4;

		filterPos-= counter/2;

		dst-= counter/2;

		asm volatile(

			"pxor %%mm7, %%mm7		\n\t"

			"movq "MANGLE(w02)", %%mm6	\n\t"

			"push %%"REG_BP"		\n\t" // we use 7 regs here ...

			"mov %%"REG_a", %%"REG_BP"	\n\t"

			ASMALIGN16

			"1:				\n\t"

			"movzwl (%2, %%"REG_BP"), %%eax	\n\t"

			"movzwl 2(%2, %%"REG_BP"), %%ebx\n\t"

			"movq (%1, %%"REG_BP", 8), %%mm1\n\t"

			"movq 16(%1, %%"REG_BP", 8), %%mm3\n\t"

			"movd (%3, %%"REG_a"), %%mm0	\n\t"

			"movd (%3, %%"REG_b"), %%mm2	\n\t"

			"punpcklbw %%mm7, %%mm0		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"pmaddwd %%mm1, %%mm0		\n\t"

			"pmaddwd %%mm2, %%mm3		\n\t"



			"movq 8(%1, %%"REG_BP", 8), %%mm1\n\t"

			"movq 24(%1, %%"REG_BP", 8), %%mm5\n\t"

			"movd 4(%3, %%"REG_a"), %%mm4	\n\t"

			"movd 4(%3, %%"REG_b"), %%mm2	\n\t"

			"punpcklbw %%mm7, %%mm4		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"pmaddwd %%mm1, %%mm4		\n\t"

			"pmaddwd %%mm2, %%mm5		\n\t"

			"paddd %%mm4, %%mm0		\n\t"

			"paddd %%mm5, %%mm3		\n\t"

						

			"psrad $8, %%mm0		\n\t"

			"psrad $8, %%mm3		\n\t"

			"packssdw %%mm3, %%mm0		\n\t"

			"pmaddwd %%mm6, %%mm0		\n\t"

			"packssdw %%mm0, %%mm0		\n\t"

			"movd %%mm0, (%4, %%"REG_BP")	\n\t"

			"add $4, %%"REG_BP"		\n\t"

			" jnc 1b			\n\t"



			"pop %%"REG_BP"			\n\t"

			: "+a" (counter)

			: "c" (filter), "d" (filterPos), "S" (src), "D" (dst)

			: "%"REG_b

		);

	}

	else

	{

		uint8_t *offset = src+filterSize;

		long counter= -2*dstW;

//		filter-= counter*filterSize/2;

		filterPos-= counter/2;

		dst-= counter/2;

		asm volatile(

			"pxor %%mm7, %%mm7		\n\t"

			"movq "MANGLE(w02)", %%mm6	\n\t"

			ASMALIGN16

			"1:				\n\t"

			"mov %2, %%"REG_c"		\n\t"

			"movzwl (%%"REG_c", %0), %%eax	\n\t"

			"movzwl 2(%%"REG_c", %0), %%ebx	\n\t"

			"mov %5, %%"REG_c"		\n\t"

			"pxor %%mm4, %%mm4		\n\t"

			"pxor %%mm5, %%mm5		\n\t"

			"2:				\n\t"

			"movq (%1), %%mm1		\n\t"

			"movq (%1, %6), %%mm3		\n\t"

			"movd (%%"REG_c", %%"REG_a"), %%mm0\n\t"

			"movd (%%"REG_c", %%"REG_b"), %%mm2\n\t"

			"punpcklbw %%mm7, %%mm0		\n\t"

			"punpcklbw %%mm7, %%mm2		\n\t"

			"pmaddwd %%mm1, %%mm0		\n\t"

			"pmaddwd %%mm2, %%mm3		\n\t"

			"paddd %%mm3, %%mm5		\n\t"

			"paddd %%mm0, %%mm4		\n\t"

			"add $8, %1			\n\t"

			"add $4, %%"REG_c"		\n\t"

			"cmp %4, %%"REG_c"		\n\t"

			" jb 2b				\n\t"

			"add %6, %1			\n\t"

			"psrad $8, %%mm4		\n\t"

			"psrad $8, %%mm5		\n\t"

			"packssdw %%mm5, %%mm4		\n\t"

			"pmaddwd %%mm6, %%mm4		\n\t"

			"packssdw %%mm4, %%mm4		\n\t"

			"mov %3, %%"REG_a"		\n\t"

			"movd %%mm4, (%%"REG_a", %0)	\n\t"

			"add $4, %0			\n\t"

			" jnc 1b			\n\t"



			: "+r" (counter), "+r" (filter)

			: "m" (filterPos), "m" (dst), "m"(offset),

			  "m" (src), "r" (filterSize*2)

			: "%"REG_b, "%"REG_a, "%"REG_c

		);

	}

#else

#ifdef HAVE_ALTIVEC

	hScale_altivec_real(dst, dstW, src, srcW, xInc, filter, filterPos, filterSize);

#else

	int i;

	for(i=0; i<dstW; i++)

	{

		int j;

		int srcPos= filterPos[i];

		int val=0;

//		printf("filterPos: %d\n", filterPos[i]);

		for(j=0; j<filterSize; j++)

		{

//			printf("filter: %d, src: %d\n", filter[i], src[srcPos + j]);

			val += ((int)src[srcPos + j])*filter[filterSize*i + j];

		}

//		filter += hFilterSize;

		dst[i] = FFMIN(FFMAX(0, val>>7), (1<<15)-1); // the cubic equation does overflow ...

//		dst[i] = val>>7;

	}

#endif

#endif

}
