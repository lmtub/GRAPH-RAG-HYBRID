static inline void RENAME(rgb32tobgr32)(const uint8_t *src, uint8_t *dst, long src_size)

{

	long idx = 15 - src_size;

	uint8_t *s = (uint8_t *) src-idx, *d = dst-idx;

#ifdef HAVE_MMX

	__asm __volatile(

		"test %0, %0			\n\t"

		"jns 2f				\n\t"

		PREFETCH" (%1, %0)		\n\t"

		"movq %3, %%mm7			\n\t"

		"pxor %4, %%mm7			\n\t"

		"movq %%mm7, %%mm6		\n\t"

		"pxor %5, %%mm7			\n\t"

		ASMALIGN(4)

		"1:				\n\t"

		PREFETCH" 32(%1, %0)		\n\t"

		"movq (%1, %0), %%mm0		\n\t"

		"movq 8(%1, %0), %%mm1		\n\t"

# ifdef HAVE_MMX2

		"pshufw $177, %%mm0, %%mm3	\n\t"

		"pshufw $177, %%mm1, %%mm5	\n\t"

		"pand %%mm7, %%mm0		\n\t"

		"pand %%mm6, %%mm3		\n\t"

		"pand %%mm7, %%mm1		\n\t"

		"pand %%mm6, %%mm5		\n\t"

		"por %%mm3, %%mm0		\n\t"

		"por %%mm5, %%mm1		\n\t"

# else

		"movq %%mm0, %%mm2		\n\t"

		"movq %%mm1, %%mm4		\n\t"

		"pand %%mm7, %%mm0		\n\t"

		"pand %%mm6, %%mm2		\n\t"

		"pand %%mm7, %%mm1		\n\t"

		"pand %%mm6, %%mm4		\n\t"

		"movq %%mm2, %%mm3		\n\t"

		"movq %%mm4, %%mm5		\n\t"

		"pslld $16, %%mm2		\n\t"

		"psrld $16, %%mm3		\n\t"

		"pslld $16, %%mm4		\n\t"

		"psrld $16, %%mm5		\n\t"

		"por %%mm2, %%mm0		\n\t"

		"por %%mm4, %%mm1		\n\t"

		"por %%mm3, %%mm0		\n\t"

		"por %%mm5, %%mm1		\n\t"

# endif

		MOVNTQ" %%mm0, (%2, %0)		\n\t"

		MOVNTQ" %%mm1, 8(%2, %0)	\n\t"

		"add $16, %0			\n\t"

		"js 1b				\n\t"

		SFENCE"				\n\t"

		EMMS"				\n\t"

		"2:				\n\t"

		: "+&r"(idx)

		: "r" (s), "r" (d), "m" (mask32b), "m" (mask32r), "m" (mmx_one)

		: "memory");

#endif

	for (; idx<15; idx+=4) {

		register int v = *(uint32_t *)&s[idx], g = v & 0xff00ff00;

		v &= 0xff00ff;

		*(uint32_t *)&d[idx] = (v>>16) + g + (v<<16);

	}

}
