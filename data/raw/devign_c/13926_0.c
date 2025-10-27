static inline void RENAME(rgb32to16)(const uint8_t *src, uint8_t *dst, unsigned src_size)

{

	const uint8_t *s = src;

	const uint8_t *end;

#ifdef HAVE_MMX

	const uint8_t *mm_end;

#endif

	uint16_t *d = (uint16_t *)dst;

	end = s + src_size;

#ifdef HAVE_MMX

	mm_end = end - 15;

#if 1 //is faster only if multiplies are reasonable fast (FIXME figure out on which cpus this is faster, on Athlon its slightly faster)

	asm volatile(

		"movq %3, %%mm5			\n\t"

		"movq %4, %%mm6			\n\t"

		"movq %5, %%mm7			\n\t"

		".balign 16			\n\t"

		"1:				\n\t"

		PREFETCH" 32(%1)		\n\t"

		"movd	(%1), %%mm0		\n\t"

		"movd	4(%1), %%mm3		\n\t"

		"punpckldq 8(%1), %%mm0		\n\t"

		"punpckldq 12(%1), %%mm3	\n\t"

		"movq %%mm0, %%mm1		\n\t"

		"movq %%mm3, %%mm4		\n\t"

		"pand %%mm6, %%mm0		\n\t"

		"pand %%mm6, %%mm3		\n\t"

		"pmaddwd %%mm7, %%mm0		\n\t"

		"pmaddwd %%mm7, %%mm3		\n\t"

		"pand %%mm5, %%mm1		\n\t"

		"pand %%mm5, %%mm4		\n\t"

		"por %%mm1, %%mm0		\n\t"	

		"por %%mm4, %%mm3		\n\t"

		"psrld $5, %%mm0		\n\t"

		"pslld $11, %%mm3		\n\t"

		"por %%mm3, %%mm0		\n\t"

		MOVNTQ"	%%mm0, (%0)		\n\t"

		"addl $16, %1			\n\t"

		"addl $8, %0			\n\t"

		"cmpl %2, %1			\n\t"

		" jb 1b				\n\t"

		: "+r" (d), "+r"(s)

		: "r" (mm_end), "m" (mask3216g), "m" (mask3216br), "m" (mul3216)

	);

#else

	__asm __volatile(PREFETCH"	%0"::"m"(*src):"memory");

	__asm __volatile(

	    "movq	%0, %%mm7\n\t"

	    "movq	%1, %%mm6\n\t"

	    ::"m"(red_16mask),"m"(green_16mask));

	while(s < mm_end)

	{

	    __asm __volatile(

		PREFETCH" 32%1\n\t"

		"movd	%1, %%mm0\n\t"

		"movd	4%1, %%mm3\n\t"

		"punpckldq 8%1, %%mm0\n\t"

		"punpckldq 12%1, %%mm3\n\t"

		"movq	%%mm0, %%mm1\n\t"

		"movq	%%mm0, %%mm2\n\t"

		"movq	%%mm3, %%mm4\n\t"

		"movq	%%mm3, %%mm5\n\t"

		"psrlq	$3, %%mm0\n\t"

		"psrlq	$3, %%mm3\n\t"

		"pand	%2, %%mm0\n\t"

		"pand	%2, %%mm3\n\t"

		"psrlq	$5, %%mm1\n\t"

		"psrlq	$5, %%mm4\n\t"

		"pand	%%mm6, %%mm1\n\t"

		"pand	%%mm6, %%mm4\n\t"

		"psrlq	$8, %%mm2\n\t"

		"psrlq	$8, %%mm5\n\t"

		"pand	%%mm7, %%mm2\n\t"

		"pand	%%mm7, %%mm5\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm4, %%mm3\n\t"

		"por	%%mm2, %%mm0\n\t"

		"por	%%mm5, %%mm3\n\t"

		"psllq	$16, %%mm3\n\t"

		"por	%%mm3, %%mm0\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		:"=m"(*d):"m"(*s),"m"(blue_16mask):"memory");

		d += 4;

		s += 16;

	}

#endif

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	while(s < end)

	{

		const int src= *((uint32_t*)s)++;

		*d++ = ((src&0xFF)>>3) + ((src&0xFC00)>>5) + ((src&0xF80000)>>8);

//		*d++ = ((src>>3)&0x1F) + ((src>>5)&0x7E0) + ((src>>8)&0xF800);

	}

}
