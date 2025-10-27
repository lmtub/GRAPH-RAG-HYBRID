static void   put_no_rnd_pixels_x2_mmx( UINT8  *block, const UINT8 *pixels, int line_size, int h)

{

  UINT8  *p;

  const UINT8 *pix;

  p = block;

  pix = pixels;

  MOVQ_ZERO(mm7);

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

	"psrlw	$1, %%mm0\n\t"

	"psrlw	$1, %%mm2\n\t"

	"packuswb  %%mm2, %%mm0\n\t"

	"movq	%%mm0, %0\n\t"

	:"=m"(*p)

	:"m"(*pix)

	:"memory");

   pix += line_size;

   p +=   line_size;

  } while (--h);

}
