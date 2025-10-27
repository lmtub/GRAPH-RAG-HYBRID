static inline void RENAME(rgb32tobgr32)(const uint8_t *src, uint8_t *dst, unsigned int src_size)

{

#ifdef HAVE_MMX

/* TODO: unroll this loop */

	asm volatile (

		"xor %%"REG_a", %%"REG_a"	\n\t"

		".balign 16			\n\t"

		"1:				\n\t"

		PREFETCH" 32(%0, %%"REG_a")	\n\t"

		"movq (%0, %%"REG_a"), %%mm0	\n\t"

		"movq %%mm0, %%mm1		\n\t"

		"movq %%mm0, %%mm2		\n\t"

		"pslld $16, %%mm0		\n\t"

		"psrld $16, %%mm1		\n\t"

		"pand "MANGLE(mask32r)", %%mm0	\n\t"

		"pand "MANGLE(mask32g)", %%mm2	\n\t"

		"pand "MANGLE(mask32b)", %%mm1	\n\t"

		"por %%mm0, %%mm2		\n\t"

		"por %%mm1, %%mm2		\n\t"

		MOVNTQ" %%mm2, (%1, %%"REG_a")	\n\t"

		"add $8, %%"REG_a"		\n\t"

		"cmp %2, %%"REG_a"		\n\t"

		" jb 1b				\n\t"

		:: "r" (src), "r"(dst), "r" ((long)src_size-7)

		: "%"REG_a

	);



	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#else

	unsigned i;

	unsigned num_pixels = src_size >> 2;

	for(i=0; i<num_pixels; i++)

	{

#ifdef WORDS_BIGENDIAN  

	  dst[4*i + 1] = src[4*i + 3];

	  dst[4*i + 2] = src[4*i + 2];

	  dst[4*i + 3] = src[4*i + 1];

#else

	  dst[4*i + 0] = src[4*i + 2];

	  dst[4*i + 1] = src[4*i + 1];

	  dst[4*i + 2] = src[4*i + 0];

#endif

	}

#endif

}
