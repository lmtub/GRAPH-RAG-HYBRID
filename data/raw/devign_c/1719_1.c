static inline void RENAME(uyvyToUV)(uint8_t *dstU, uint8_t *dstV, uint8_t *src1, uint8_t *src2, int width)

{

#if defined (HAVE_MMX2) || defined (HAVE_3DNOW)

	asm volatile(

		"movq "MANGLE(bm01010101)", %%mm4\n\t"

		"mov %0, %%"REG_a"		\n\t"

		"1:				\n\t"

		"movq (%1, %%"REG_a",4), %%mm0	\n\t"

		"movq 8(%1, %%"REG_a",4), %%mm1	\n\t"

		"movq (%2, %%"REG_a",4), %%mm2	\n\t"

		"movq 8(%2, %%"REG_a",4), %%mm3	\n\t"

		PAVGB(%%mm2, %%mm0)

		PAVGB(%%mm3, %%mm1)

		"pand %%mm4, %%mm0		\n\t"

		"pand %%mm4, %%mm1		\n\t"

		"packuswb %%mm1, %%mm0		\n\t"

		"movq %%mm0, %%mm1		\n\t"

		"psrlw $8, %%mm0		\n\t"

		"pand %%mm4, %%mm1		\n\t"

		"packuswb %%mm0, %%mm0		\n\t"

		"packuswb %%mm1, %%mm1		\n\t"

		"movd %%mm0, (%4, %%"REG_a")	\n\t"

		"movd %%mm1, (%3, %%"REG_a")	\n\t"

		"add $4, %%"REG_a"		\n\t"

		" js 1b				\n\t"

		: : "g" ((long)-width), "r" (src1+width*4), "r" (src2+width*4), "r" (dstU+width), "r" (dstV+width)

		: "%"REG_a

	);

#else

	int i;

	for(i=0; i<width; i++)

	{

		dstU[i]= (src1[4*i + 0] + src2[4*i + 0])>>1;

		dstV[i]= (src1[4*i + 2] + src2[4*i + 2])>>1;

	}

#endif

}
