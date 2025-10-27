void rgb15to16(const uint8_t *src,uint8_t *dst,uint32_t src_size)

{

#ifdef HAVE_MMX

  register const char* s=src+src_size;

  register char* d=dst+src_size;

  register int offs=-src_size;

  __asm __volatile(PREFETCH"	%0"::"m"(*(s+offs)):"memory");

  __asm __volatile(

	"movq	%0, %%mm4\n\t"

	"movq	%1, %%mm5"

	::"m"(mask15b), "m"(mask15rg):"memory");

  while(offs<0)

  {

	__asm __volatile(

		PREFETCH"	32%1\n\t"

		"movq	%1, %%mm0\n\t"

		"movq	8%1, %%mm2\n\t"

		"movq	%%mm0, %%mm1\n\t"

		"movq	%%mm2, %%mm3\n\t"

		"pand	%%mm4, %%mm0\n\t"

		"pand	%%mm5, %%mm1\n\t"

		"pand	%%mm4, %%mm2\n\t"

		"pand	%%mm5, %%mm3\n\t"

		"psllq	$1, %%mm1\n\t"

		"psllq	$1, %%mm3\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm3, %%mm2\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm2, 8%0"

		:"=m"(*(d+offs))

		:"m"(*(s+offs))

		:"memory");

	offs+=16;

  }

  __asm __volatile(SFENCE:::"memory");

  __asm __volatile(EMMS:::"memory");

#else

   const uint16_t *s1=( uint16_t * )src;

   uint16_t *d1=( uint16_t * )dst;

   uint16_t *e=((uint8_t *)s1)+src_size;

   while( s1<e ){

     register int x=*( s1++ );

     /* rrrrrggggggbbbbb

        0rrrrrgggggbbbbb

        0111 1111 1110 0000=0x7FE0

        00000000000001 1111=0x001F */

     *( d1++ )=( x&0x001F )|( ( x&0x7FE0 )<<1 );

   }

#endif

}
