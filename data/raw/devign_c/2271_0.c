static void put_pixels_x2_mmx(UINT8 *block, const UINT8 *pixels, int line_size, int h)

{

#if 0

  UINT8 *p;

  const UINT8 *pix;

  p = block;

  pix = pixels;

  MOVQ_ZERO(mm7);

  MOVQ_WONE(mm4);

  JUMPALIGN();

  do {

      __asm __volatile(

	"movq	%1, %%mm0\n\t"

	"movq	1%1, %%mm1\n\t"

	"movq	%%mm0, %%mm2\n\t"

	"movq	%%mm1, %%mm3\n\t"

	"punpcklbw %%mm7, %%mm0\n\t"

	"punpcklbw %%mm7, %%mm1\n\t"

	"punpckhbw %%mm7, %%mm2\n\t"

	"punpckhbw %%mm7, %%mm3\n\t"

	"paddusw %%mm1, %%mm0\n\t"

	"paddusw %%mm3, %%mm2\n\t"

	"paddusw %%mm4, %%mm0\n\t"

	"paddusw %%mm4, %%mm2\n\t"

	"psrlw	$1, %%mm0\n\t"

	"psrlw	$1, %%mm2\n\t"

	"packuswb  %%mm2, %%mm0\n\t"

	"movq	%%mm0, %0\n\t"

	:"=m"(*p)

	:"m"(*pix)

		       :"memory");

   pix += line_size; p += line_size;

  } while (--h);

#else

  __asm __volatile(

  	MOVQ_BFE(%%mm7)

	"lea (%3, %3), %%eax	\n\t"

	".balign 8     		\n\t"

	"1:			\n\t"

	"movq (%1), %%mm0	\n\t"

	"movq (%1, %3), %%mm2	\n\t"

	"movq 1(%1), %%mm1	\n\t"

	"movq 1(%1, %3), %%mm3	\n\t"

	PAVG_MMX(%%mm0, %%mm1)

	"movq %%mm6, (%2)	\n\t"

	PAVG_MMX(%%mm2, %%mm3)

	"movq %%mm6, (%2, %3)	\n\t"

	"addl %%eax, %1		\n\t"

	"addl %%eax, %2		\n\t"

#if LONG_UNROLL

	"movq (%1), %%mm0	\n\t"

	"movq (%1, %3), %%mm2	\n\t"

	"movq 1(%1), %%mm1	\n\t"

	"movq 1(%1, %3), %%mm3	\n\t"

	PAVG_MMX(%%mm0, %%mm1)

	"movq %%mm6, (%2)	\n\t"

	PAVG_MMX(%%mm2, %%mm3)

	"movq %%mm6, (%2, %3)	\n\t"

	"addl %%eax, %1		\n\t"

	"addl %%eax, %2		\n\t"

	"subl $4, %0		\n\t"

#else

	"subl $2, %0		\n\t"

#endif

	"jnz 1b			\n\t"

	:"+g"(h), "+S"(pixels), "+D"(block)

	:"r"(line_size)

	:"eax", "memory");

#endif

}
