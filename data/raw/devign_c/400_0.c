static void dct_unquantize_mpeg1_mmx(MpegEncContext *s,

                                     DCTELEM *block, int n, int qscale)

{

    int i, level;

    const UINT16 *quant_matrix;

    if (s->mb_intra) {

        if (n < 4) 

            block[0] = block[0] * s->y_dc_scale;

        else

            block[0] = block[0] * s->c_dc_scale;

        if (s->out_format == FMT_H263) {

            i = 1;

            goto unquant_even;

        }

        /* XXX: only mpeg1 */

        quant_matrix = s->intra_matrix;

	i=1;

	/* Align on 4 elements boundary */

	while(i&3)

	{

            level = block[i];

            if (level) {

                if (level < 0) level = -level;

                    level = (int)(level * qscale * quant_matrix[i]) >> 3;

                    level = (level - 1) | 1;

                if (block[i] < 0) level = -level;

                block[i] = level;

            }

	    i++;

	}

	__asm __volatile(

	"movd	%0, %%mm6\n\t"       /* mm6 = qscale | 0  */

	"punpckldq %%mm6, %%mm6\n\t" /* mm6 = qscale | qscale */

	"movq	%2, %%mm4\n\t"

	"movq	%%mm6, %%mm7\n\t"

	"movq	%1, %%mm5\n\t"

	"packssdw %%mm6, %%mm7\n\t" /* mm7 = qscale | qscale | qscale | qscale */

	"pxor	%%mm6, %%mm6\n\t"

	::"g"(qscale),"m"(mm_wone),"m"(mm_wabs):"memory");

        for(;i<64;i+=4) {

		__asm __volatile(

			"movq	%1, %%mm0\n\t"

			"movq	%%mm7, %%mm1\n\t"

			"movq	%%mm0, %%mm2\n\t"

			"movq	%%mm0, %%mm3\n\t"

			"pcmpgtw %%mm6, %%mm2\n\t"

			"pmullw	%2, %%mm1\n\t"

			"pandn	%%mm4, %%mm2\n\t"

			"por	%%mm5, %%mm2\n\t"

			"pmullw	%%mm2, %%mm0\n\t" /* mm0 = abs(block[i]). */



			"pcmpeqw %%mm6, %%mm3\n\t"

			"pmullw	%%mm0, %%mm1\n\t"

			"psraw	$3, %%mm1\n\t"

			"psubw	%%mm5, %%mm1\n\t"   /* block[i] --; */

			"pandn	%%mm4, %%mm3\n\t"  /* fake of pcmpneqw : mm0 != 0 then mm1 = -1 */

			"por	%%mm5, %%mm1\n\t"   /* block[i] |= 1 */

			"pmullw %%mm2, %%mm1\n\t"   /* change signs again */



			"pand	%%mm3, %%mm1\n\t" /* nullify if was zero */

			"movq	%%mm1, %0"

			:"=m"(block[i])

			:"m"(block[i]), "m"(quant_matrix[i])

			:"memory");

        }

    } else {

        i = 0;

    unquant_even:

        quant_matrix = s->non_intra_matrix;

	/* Align on 4 elements boundary */

	while(i&7)

	{

	    level = block[i];

            if (level) {

                if (level < 0) level = -level;

                    level = (((level << 1) + 1) * qscale *

                             ((int) quant_matrix[i])) >> 4;

                    level = (level - 1) | 1;

                if(block[i] < 0) level = -level;

                block[i] = level;

	    }

	    i++;

	}



asm volatile(

		"pcmpeqw %%mm7, %%mm7		\n\t"

		"psrlw $15, %%mm7		\n\t"

		"movd %2, %%mm6			\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

		"1:				\n\t"

		"movq (%0, %3), %%mm0		\n\t"

		"movq 8(%0, %3), %%mm1		\n\t"

		"movq (%1, %3), %%mm4		\n\t"

		"movq 8(%1, %3), %%mm5		\n\t"

		"pmullw %%mm6, %%mm4		\n\t" // q=qscale*quant_matrix[i]

		"pmullw %%mm6, %%mm5		\n\t" // q=qscale*quant_matrix[i]

		"pxor %%mm2, %%mm2		\n\t"

		"pxor %%mm3, %%mm3		\n\t"

		"pcmpgtw %%mm0, %%mm2		\n\t" // block[i] < 0 ? -1 : 0

		"pcmpgtw %%mm1, %%mm3		\n\t" // block[i] < 0 ? -1 : 0

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t" // abs(block[i])

		"psubw %%mm3, %%mm1		\n\t" // abs(block[i])

		"paddw %%mm0, %%mm0		\n\t" // abs(block[i])*2

		"paddw %%mm1, %%mm1		\n\t" // abs(block[i])*2

		"paddw %%mm7, %%mm0		\n\t" // abs(block[i])*2 + 1

		"paddw %%mm7, %%mm1		\n\t" // abs(block[i])*2 + 1

		"pmullw %%mm4, %%mm0		\n\t" // (abs(block[i])*2 + 1)*q

		"pmullw %%mm5, %%mm1		\n\t" // (abs(block[i])*2 + 1)*q

		"pxor %%mm4, %%mm4		\n\t"

		"pxor %%mm5, %%mm5		\n\t" // FIXME slow

		"pcmpeqw (%0, %3), %%mm4	\n\t" // block[i] == 0 ? -1 : 0

		"pcmpeqw 8(%0, %3), %%mm5	\n\t" // block[i] == 0 ? -1 : 0

		"psraw $4, %%mm0		\n\t"

		"psraw $4, %%mm1		\n\t"

		"psubw %%mm7, %%mm0		\n\t"

		"psubw %%mm7, %%mm1		\n\t"

		"por %%mm7, %%mm0		\n\t"

		"por %%mm7, %%mm1		\n\t"

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t"

		"psubw %%mm3, %%mm1		\n\t"

		"pandn %%mm0, %%mm4		\n\t"

		"pandn %%mm1, %%mm5		\n\t"

		"movq %%mm4, (%0, %3)		\n\t"

		"movq %%mm5, 8(%0, %3)		\n\t"



		"addl $16, %3			\n\t"

		"cmpl $128, %3			\n\t"

		"jb 1b				\n\t"

		::"r" (block), "r"(quant_matrix), "g" (qscale), "r" (2*i)

		: "memory"

	);



#if 0

	__asm __volatile(

	"movd	%0, %%mm6\n\t"       /* mm6 = qscale | 0  */

	"punpckldq %%mm6, %%mm6\n\t" /* mm6 = qscale | qscale */

	"movq	%2, %%mm4\n\t"

	"movq	%%mm6, %%mm7\n\t"

	"movq	%1, %%mm5\n\t"

	"packssdw %%mm6, %%mm7\n\t" /* mm7 = qscale | qscale | qscale | qscale */

	"pxor	%%mm6, %%mm6\n\t"

	::"g"(qscale),"m"(mm_wone),"m"(mm_wabs));

        for(;i<64;i+=4) {

		__asm __volatile(

			"movq	%1, %%mm0\n\t"

			"movq	%%mm7, %%mm1\n\t"

			"movq	%%mm0, %%mm2\n\t"

			"movq	%%mm0, %%mm3\n\t"

			"pcmpgtw %%mm6, %%mm2\n\t"

			"pmullw	%2, %%mm1\n\t"

			"pandn	%%mm4, %%mm2\n\t"

			"por	%%mm5, %%mm2\n\t"

			"pmullw	%%mm2, %%mm0\n\t" /* mm0 = abs(block[i]). */

			"psllw	$1, %%mm0\n\t" /* block[i] <<= 1 */

			"paddw	%%mm5, %%mm0\n\t" /* block[i] ++ */



			"pmullw	%%mm0, %%mm1\n\t"

			"psraw	$4, %%mm1\n\t"

			"pcmpeqw %%mm6, %%mm3\n\t"

			"psubw	%%mm5, %%mm1\n\t"   /* block[i] --; */

			"pandn	%%mm4, %%mm3\n\t"  /* fake of pcmpneqw : mm0 != 0 then mm1 = -1 */

			"por	%%mm5, %%mm1\n\t"   /* block[i] |= 1 */

			"pmullw %%mm2, %%mm1\n\t"   /* change signs again */



			"pand	%%mm3, %%mm1\n\t" /* nullify if was zero */

			"movq	%%mm1, %0"

			:"=m"(block[i])

			:"m"(block[i]), "m"(quant_matrix[i]));

        }

#endif

    }

}
