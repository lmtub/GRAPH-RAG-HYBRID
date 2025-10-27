static inline void RENAME(rgb16to15)(const uint8_t *src,uint8_t *dst,long src_size)

{

  register const uint8_t* s=src;

  register uint8_t* d=dst;

  register const uint8_t *end;

  const uint8_t *mm_end;

  end = s + src_size;

#ifdef HAVE_MMX

  __asm __volatile(PREFETCH"	%0"::"m"(*s));

  __asm __volatile("movq	%0, %%mm7"::"m"(mask15rg));

  __asm __volatile("movq	%0, %%mm6"::"m"(mask15b));

  mm_end = end - 15;

  while(s<mm_end)

  {

	__asm __volatile(

		PREFETCH"	32%1\n\t"

		"movq	%1, %%mm0\n\t"

		"movq	8%1, %%mm2\n\t"

		"movq	%%mm0, %%mm1\n\t"

		"movq	%%mm2, %%mm3\n\t"

		"psrlq	$1, %%mm0\n\t"

		"psrlq	$1, %%mm2\n\t"

		"pand	%%mm7, %%mm0\n\t"

		"pand	%%mm7, %%mm2\n\t"

		"pand	%%mm6, %%mm1\n\t"

		"pand	%%mm6, %%mm3\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm3, %%mm2\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm2, 8%0"

		:"=m"(*d)

		:"m"(*s)

		);

	d+=16;

	s+=16;

  }

  __asm __volatile(SFENCE:::"memory");

  __asm __volatile(EMMS:::"memory");

#endif

    mm_end = end - 3;

    while(s < mm_end)

    {

	register uint32_t x= *((uint32_t *)s);

	*((uint32_t *)d) = ((x>>1)&0x7FE07FE0) | (x&0x001F001F);

	s+=4;

	d+=4;

    }

    if(s < end)

    {

	register uint16_t x= *((uint16_t *)s);

	*((uint16_t *)d) = ((x>>1)&0x7FE0) | (x&0x001F);

	s+=2;

	d+=2;

    }

}
