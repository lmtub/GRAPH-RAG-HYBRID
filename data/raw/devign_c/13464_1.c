static inline void RENAME(rgb32to24)(const uint8_t *src,uint8_t *dst,long src_size)

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

  mm_end = end - 31;

  while(s < mm_end)

  {

    __asm __volatile(

	PREFETCH"	32%1\n\t"

	"movq	%1, %%mm0\n\t"

	"movq	8%1, %%mm1\n\t"

	"movq	16%1, %%mm4\n\t"

	"movq	24%1, %%mm5\n\t"

	"movq	%%mm0, %%mm2\n\t"

	"movq	%%mm1, %%mm3\n\t"

	"movq	%%mm4, %%mm6\n\t"

	"movq	%%mm5, %%mm7\n\t"

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

	:"=m"(*dest)

	:"m"(*s),"m"(mask24l),

	 "m"(mask24h),"m"(mask24hh),"m"(mask24hhh),"m"(mask24hhhh)

	:"memory");

    dest += 24;

    s += 32;

  }

  __asm __volatile(SFENCE:::"memory");

  __asm __volatile(EMMS:::"memory");

#endif

  while(s < end)

  {

#ifdef WORDS_BIGENDIAN

    /* RGB32 (= A,B,G,R) -> RGB24 (= R,G,B) */

    s++;

    dest[2] = *s++;

    dest[1] = *s++;

    dest[0] = *s++;

    dest += 3;

#else

    *dest++ = *s++;

    *dest++ = *s++;

    *dest++ = *s++;

    s++;

#endif

  }

}
