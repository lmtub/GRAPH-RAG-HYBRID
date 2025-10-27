static inline void RENAME(rgb15to24)(const uint8_t *src, uint8_t *dst, long src_size)

{

	const uint16_t *end;

#ifdef HAVE_MMX

	const uint16_t *mm_end;

#endif

	uint8_t *d = (uint8_t *)dst;

	const uint16_t *s = (uint16_t *)src;

	end = s + src_size/2;

#ifdef HAVE_MMX

	__asm __volatile(PREFETCH"	%0"::"m"(*s):"memory");

	mm_end = end - 7;

	while(s < mm_end)

	{

	    __asm __volatile(

		PREFETCH" 32%1\n\t"

		"movq	%1, %%mm0\n\t"

		"movq	%1, %%mm1\n\t"

		"movq	%1, %%mm2\n\t"

		"pand	%2, %%mm0\n\t"

		"pand	%3, %%mm1\n\t"

		"pand	%4, %%mm2\n\t"

		"psllq	$3, %%mm0\n\t"

		"psrlq	$2, %%mm1\n\t"

		"psrlq	$7, %%mm2\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"movq	%%mm1, %%mm4\n\t"

		"movq	%%mm2, %%mm5\n\t"

		"punpcklwd %5, %%mm0\n\t"

		"punpcklwd %5, %%mm1\n\t"

		"punpcklwd %5, %%mm2\n\t"

		"punpckhwd %5, %%mm3\n\t"

		"punpckhwd %5, %%mm4\n\t"

		"punpckhwd %5, %%mm5\n\t"

		"psllq	$8, %%mm1\n\t"

		"psllq	$16, %%mm2\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm2, %%mm0\n\t"

		"psllq	$8, %%mm4\n\t"

		"psllq	$16, %%mm5\n\t"

		"por	%%mm4, %%mm3\n\t"

		"por	%%mm5, %%mm3\n\t"



		"movq	%%mm0, %%mm6\n\t"

		"movq	%%mm3, %%mm7\n\t"



		"movq	8%1, %%mm0\n\t"

		"movq	8%1, %%mm1\n\t"

		"movq	8%1, %%mm2\n\t"

		"pand	%2, %%mm0\n\t"

		"pand	%3, %%mm1\n\t"

		"pand	%4, %%mm2\n\t"

		"psllq	$3, %%mm0\n\t"

		"psrlq	$2, %%mm1\n\t"

		"psrlq	$7, %%mm2\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"movq	%%mm1, %%mm4\n\t"

		"movq	%%mm2, %%mm5\n\t"

		"punpcklwd %5, %%mm0\n\t"

		"punpcklwd %5, %%mm1\n\t"

		"punpcklwd %5, %%mm2\n\t"

		"punpckhwd %5, %%mm3\n\t"

		"punpckhwd %5, %%mm4\n\t"

		"punpckhwd %5, %%mm5\n\t"

		"psllq	$8, %%mm1\n\t"

		"psllq	$16, %%mm2\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm2, %%mm0\n\t"

		"psllq	$8, %%mm4\n\t"

		"psllq	$16, %%mm5\n\t"

		"por	%%mm4, %%mm3\n\t"

		"por	%%mm5, %%mm3\n\t"



		:"=m"(*d)

		:"m"(*s),"m"(mask15b),"m"(mask15g),"m"(mask15r), "m"(mmx_null)

		:"memory");

	    /* Borrowed 32 to 24 */

	    __asm __volatile(

		"movq	%%mm0, %%mm4\n\t"

		"movq	%%mm3, %%mm5\n\t"

		"movq	%%mm6, %%mm0\n\t"

		"movq	%%mm7, %%mm1\n\t"



		"movq	%%mm4, %%mm6\n\t"

		"movq	%%mm5, %%mm7\n\t"

		"movq	%%mm0, %%mm2\n\t"

		"movq	%%mm1, %%mm3\n\t"



		"psrlq	$8, %%mm2\n\t"

		"psrlq	$8, %%mm3\n\t"

		"psrlq	$8, %%mm6\n\t"

		"psrlq	$8, %%mm7\n\t"

		"pand	%2, %%mm0\n\t"

		"pand	%2, %%mm1\n\t"

		"pand	%2, %%mm4\n\t"

		"pand	%2, %%mm5\n\t"

		"pand	%3, %%mm2\n\t"

		"pand	%3, %%mm3\n\t"

		"pand	%3, %%mm6\n\t"

		"pand	%3, %%mm7\n\t"

		"por	%%mm2, %%mm0\n\t"

		"por	%%mm3, %%mm1\n\t"

		"por	%%mm6, %%mm4\n\t"

		"por	%%mm7, %%mm5\n\t"



		"movq	%%mm1, %%mm2\n\t"

		"movq	%%mm4, %%mm3\n\t"

		"psllq	$48, %%mm2\n\t"

		"psllq	$32, %%mm3\n\t"

		"pand	%4, %%mm2\n\t"

		"pand	%5, %%mm3\n\t"

		"por	%%mm2, %%mm0\n\t"

		"psrlq	$16, %%mm1\n\t"

		"psrlq	$32, %%mm4\n\t"

		"psllq	$16, %%mm5\n\t"

		"por	%%mm3, %%mm1\n\t"

		"pand	%6, %%mm5\n\t"

		"por	%%mm5, %%mm4\n\t"



		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm1, 8%0\n\t"

		MOVNTQ"	%%mm4, 16%0"



		:"=m"(*d)

		:"m"(*s),"m"(mask24l),"m"(mask24h),"m"(mask24hh),"m"(mask24hhh),"m"(mask24hhhh)

		:"memory");

		d += 24;

		s += 8;

	}

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	while(s < end)

	{

		register uint16_t bgr;

		bgr = *s++;

		*d++ = (bgr&0x1F)<<3;

		*d++ = (bgr&0x3E0)>>2;

		*d++ = (bgr&0x7C00)>>7;

	}

}
