static inline void RENAME(rgb24tobgr24)(const uint8_t *src, uint8_t *dst, unsigned int src_size)

{

	unsigned i;

#ifdef HAVE_MMX

	long mmx_size= 23 - src_size;

	asm volatile (

		"movq "MANGLE(mask24r)", %%mm5	\n\t"

		"movq "MANGLE(mask24g)", %%mm6	\n\t"

		"movq "MANGLE(mask24b)", %%mm7	\n\t"

		".balign 16			\n\t"

		"1:				\n\t"

		PREFETCH" 32(%1, %%"REG_a")	\n\t"

		"movq   (%1, %%"REG_a"), %%mm0	\n\t" // BGR BGR BG

		"movq   (%1, %%"REG_a"), %%mm1	\n\t" // BGR BGR BG

		"movq  2(%1, %%"REG_a"), %%mm2	\n\t" // R BGR BGR B

		"psllq $16, %%mm0		\n\t" // 00 BGR BGR

		"pand %%mm5, %%mm0		\n\t"

		"pand %%mm6, %%mm1		\n\t"

		"pand %%mm7, %%mm2		\n\t"

		"por %%mm0, %%mm1		\n\t"

		"por %%mm2, %%mm1		\n\t"                

		"movq  6(%1, %%"REG_a"), %%mm0	\n\t" // BGR BGR BG

		MOVNTQ" %%mm1,   (%2, %%"REG_a")\n\t" // RGB RGB RG

		"movq  8(%1, %%"REG_a"), %%mm1	\n\t" // R BGR BGR B

		"movq 10(%1, %%"REG_a"), %%mm2	\n\t" // GR BGR BGR

		"pand %%mm7, %%mm0		\n\t"

		"pand %%mm5, %%mm1		\n\t"

		"pand %%mm6, %%mm2		\n\t"

		"por %%mm0, %%mm1		\n\t"

		"por %%mm2, %%mm1		\n\t"                

		"movq 14(%1, %%"REG_a"), %%mm0	\n\t" // R BGR BGR B

		MOVNTQ" %%mm1,  8(%2, %%"REG_a")\n\t" // B RGB RGB R

		"movq 16(%1, %%"REG_a"), %%mm1	\n\t" // GR BGR BGR

		"movq 18(%1, %%"REG_a"), %%mm2	\n\t" // BGR BGR BG

		"pand %%mm6, %%mm0		\n\t"

		"pand %%mm7, %%mm1		\n\t"

		"pand %%mm5, %%mm2		\n\t"

		"por %%mm0, %%mm1		\n\t"

		"por %%mm2, %%mm1		\n\t"                

		MOVNTQ" %%mm1, 16(%2, %%"REG_a")\n\t"

		"add $24, %%"REG_a"		\n\t"

		" js 1b				\n\t"

		: "+a" (mmx_size)

		: "r" (src-mmx_size), "r"(dst-mmx_size)

	);



	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");



	if(mmx_size==23) return; //finihsed, was multiple of 8



	src+= src_size;

	dst+= src_size;

	src_size= 23-mmx_size;

	src-= src_size;

	dst-= src_size;

#endif

	for(i=0; i<src_size; i+=3)

	{

		register uint8_t x;

		x          = src[i + 2];

		dst[i + 1] = src[i + 1];

		dst[i + 2] = src[i + 0];

		dst[i + 0] = x;

	}

}
