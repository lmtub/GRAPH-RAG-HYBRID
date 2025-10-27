static inline void blockCopy(uint8_t dst[], int dstStride, uint8_t src[], int srcStride,

	int numLines, int levelFix)

{

	int i;

	if(levelFix)

	{

#ifdef HAVE_MMX

					asm volatile(

						"movl %4, %%eax \n\t"

						"movl %%eax, temp0\n\t"

						"pushl %0 \n\t"

						"pushl %1 \n\t"

						"leal (%2,%2), %%eax	\n\t"

						"leal (%3,%3), %%ebx	\n\t"

						"movq packedYOffset, %%mm2	\n\t"

						"movq packedYScale, %%mm3	\n\t"

						"pxor %%mm4, %%mm4	\n\t"



#define SCALED_CPY					\

						"movq (%0), %%mm0	\n\t"\

						"movq (%0,%2), %%mm1	\n\t"\

						"psubusb %%mm2, %%mm0	\n\t"\

						"psubusb %%mm2, %%mm1	\n\t"\

						"movq %%mm0, %%mm5	\n\t"\

						"punpcklbw %%mm4, %%mm0 \n\t"\

						"punpckhbw %%mm4, %%mm5 \n\t"\

						"psllw $7, %%mm0	\n\t"\

						"psllw $7, %%mm5	\n\t"\

						"pmulhw %%mm3, %%mm0	\n\t"\

						"pmulhw %%mm3, %%mm5	\n\t"\

						"packuswb %%mm5, %%mm0	\n\t"\

						"movq %%mm0, (%1)	\n\t"\

						"movq %%mm1, %%mm5	\n\t"\

						"punpcklbw %%mm4, %%mm1 \n\t"\

						"punpckhbw %%mm4, %%mm5 \n\t"\

						"psllw $7, %%mm1	\n\t"\

						"psllw $7, %%mm5	\n\t"\

						"pmulhw %%mm3, %%mm1	\n\t"\

						"pmulhw %%mm3, %%mm5	\n\t"\

						"packuswb %%mm5, %%mm1	\n\t"\

						"movq %%mm1, (%1, %3)	\n\t"\



						"1:			\n\t"

SCALED_CPY

						"addl %%eax, %0		\n\t"

						"addl %%ebx, %1		\n\t"

SCALED_CPY

						"addl %%eax, %0		\n\t"

						"addl %%ebx, %1		\n\t"

						"decl temp0		\n\t"

						"jnz 1b			\n\t"



						"popl %1 \n\t"

						"popl %0 \n\t"

						: : "r" (src),

						"r" (dst),

						"r" (srcStride),

						"r" (dstStride),

						"m" (numLines>>2)

						: "%eax", "%ebx"

					);

#else

				for(i=0; i<numLines; i++)

					memcpy(	&(dst[dstStride*i]),

						&(src[srcStride*i]), BLOCK_SIZE);

#endif

	}

	else

	{

#ifdef HAVE_MMX

					asm volatile(

						"movl %4, %%eax \n\t"

						"movl %%eax, temp0\n\t"

						"pushl %0 \n\t"

						"pushl %1 \n\t"

						"leal (%2,%2), %%eax	\n\t"

						"leal (%3,%3), %%ebx	\n\t"

						"movq packedYOffset, %%mm2	\n\t"

						"movq packedYScale, %%mm3	\n\t"



#define SIMPLE_CPY					\

						"movq (%0), %%mm0	\n\t"\

						"movq (%0,%2), %%mm1	\n\t"\

						"movq %%mm0, (%1)	\n\t"\

						"movq %%mm1, (%1, %3)	\n\t"\



						"1:			\n\t"

SIMPLE_CPY

						"addl %%eax, %0		\n\t"

						"addl %%ebx, %1		\n\t"

SIMPLE_CPY

						"addl %%eax, %0		\n\t"

						"addl %%ebx, %1		\n\t"

						"decl temp0		\n\t"

						"jnz 1b			\n\t"



						"popl %1 \n\t"

						"popl %0 \n\t"

						: : "r" (src),

						"r" (dst),

						"r" (srcStride),

						"r" (dstStride),

						"m" (numLines>>2)

						: "%eax", "%ebx"

					);

#else

				for(i=0; i<numLines; i++)

					memcpy(	&(dst[dstStride*i]),

						&(src[srcStride*i]), BLOCK_SIZE);

#endif

	}

}
