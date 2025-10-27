static inline void RENAME(rgb24to32)(const uint8_t *src,uint8_t *dst,unsigned src_size)

{

  uint8_t *dest = dst;

  const uint8_t *s = src;

  const uint8_t *end;

#ifdef HAVE_MMX

  const uint8_t *mm_end;

#endif

  end = s + src_size;

#ifdef HAVE_MMX

  __asm __volatile(PREFETCH"	%0"::"m"(*s):"memory");

  mm_end = end - 23;

  __asm __volatile("movq	%0, %%mm7"::"m"(mask32):"memory");

  while(s < mm_end)

  {

    __asm __volatile(

	PREFETCH"	32%1\n\t"

	"movd	%1, %%mm0\n\t"

	"punpckldq 3%1, %%mm0\n\t"

	"movd	6%1, %%mm1\n\t"

	"punpckldq 9%1, %%mm1\n\t"

	"movd	12%1, %%mm2\n\t"

	"punpckldq 15%1, %%mm2\n\t"

	"movd	18%1, %%mm3\n\t"

	"punpckldq 21%1, %%mm3\n\t"

	"pand	%%mm7, %%mm0\n\t"

	"pand	%%mm7, %%mm1\n\t"

	"pand	%%mm7, %%mm2\n\t"

	"pand	%%mm7, %%mm3\n\t"

	MOVNTQ"	%%mm0, %0\n\t"

	MOVNTQ"	%%mm1, 8%0\n\t"

	MOVNTQ"	%%mm2, 16%0\n\t"

	MOVNTQ"	%%mm3, 24%0"

	:"=m"(*dest)

	:"m"(*s)

	:"memory");

    dest += 32;

    s += 24;

  }

  __asm __volatile(SFENCE:::"memory");

  __asm __volatile(EMMS:::"memory");

#endif

  while(s < end)

  {

#ifdef WORDS_BIGENDIAN

    *dest++ = 0;

    *dest++ = *s++;

    *dest++ = *s++;

    *dest++ = *s++;

#else

    *dest++ = *s++;

    *dest++ = *s++;

    *dest++ = *s++;

    *dest++ = 0;

#endif

  }

}
