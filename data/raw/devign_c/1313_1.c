static inline void RENAME(rgb24tobgr16)(const uint8_t *src, uint8_t *dst, long src_size)

{

	const uint8_t *s = src;

	const uint8_t *end;

#ifdef HAVE_MMX

	const uint8_t *mm_end;

#endif

	uint16_t *d = (uint16_t *)dst;

	end = s + src_size;

#ifdef HAVE_MMX

	__asm __volatile(PREFETCH"	%0"::"m"(*src):"memory");

	__asm __volatile(

	    "movq	%0, %%mm7\n\t"

	    "movq	%1, %%mm6\n\t"

	    ::"m"(red_16mask),"m"(green_16mask));

	mm_end = end - 15;

	while(s < mm_end)

	{

	    __asm __volatile(

		PREFETCH" 32%1\n\t"

		"movd	%1, %%mm0\n\t"

		"movd	3%1, %%mm3\n\t"

		"punpckldq 6%1, %%mm0\n\t"

		"punpckldq 9%1, %%mm3\n\t"

		"movq	%%mm0, %%mm1\n\t"

		"movq	%%mm0, %%mm2\n\t"

		"movq	%%mm3, %%mm4\n\t"

		"movq	%%mm3, %%mm5\n\t"

		"psllq	$8, %%mm0\n\t"

		"psllq	$8, %%mm3\n\t"

		"pand	%%mm7, %%mm0\n\t"

		"pand	%%mm7, %%mm3\n\t"

		"psrlq	$5, %%mm1\n\t"

		"psrlq	$5, %%mm4\n\t"

		"pand	%%mm6, %%mm1\n\t"

		"pand	%%mm6, %%mm4\n\t"

		"psrlq	$19, %%mm2\n\t"

		"psrlq	$19, %%mm5\n\t"

		"pand	%2, %%mm2\n\t"

		"pand	%2, %%mm5\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm4, %%mm3\n\t"

		"por	%%mm2, %%mm0\n\t"

		"por	%%mm5, %%mm3\n\t"

		"psllq	$16, %%mm3\n\t"

		"por	%%mm3, %%mm0\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		:"=m"(*d):"m"(*s),"m"(blue_16mask):"memory");

		d += 4;

		s += 12;

	}

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	while(s < end)

	{

		const int r= *s++;

		const int g= *s++;

		const int b= *s++;

		*d++ = (b>>3) | ((g&0xFC)<<3) | ((r&0xF8)<<8);

	}

}
