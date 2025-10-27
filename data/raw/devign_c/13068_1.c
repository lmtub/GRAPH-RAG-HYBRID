static inline void RENAME(vu9_to_vu12)(const uint8_t *src1, const uint8_t *src2,

			uint8_t *dst1, uint8_t *dst2,

			long width, long height,

			long srcStride1, long srcStride2,

			long dstStride1, long dstStride2)

{

    long y,x,w,h;

    w=width/2; h=height/2;

#ifdef HAVE_MMX

    asm volatile(

	PREFETCH" %0\n\t"

	PREFETCH" %1\n\t"

	::"m"(*(src1+srcStride1)),"m"(*(src2+srcStride2)):"memory");

#endif

    for(y=0;y<h;y++){

	const uint8_t* s1=src1+srcStride1*(y>>1);

	uint8_t* d=dst1+dstStride1*y;

	x=0;

#ifdef HAVE_MMX

	for(;x<w-31;x+=32)

	{

	    asm volatile(

		PREFETCH" 32%1\n\t"

	        "movq	%1, %%mm0\n\t"

	        "movq	8%1, %%mm2\n\t"

	        "movq	16%1, %%mm4\n\t"

	        "movq	24%1, %%mm6\n\t"

	        "movq	%%mm0, %%mm1\n\t"

	        "movq	%%mm2, %%mm3\n\t"

	        "movq	%%mm4, %%mm5\n\t"

	        "movq	%%mm6, %%mm7\n\t"

		"punpcklbw %%mm0, %%mm0\n\t"

		"punpckhbw %%mm1, %%mm1\n\t"

		"punpcklbw %%mm2, %%mm2\n\t"

		"punpckhbw %%mm3, %%mm3\n\t"

		"punpcklbw %%mm4, %%mm4\n\t"

		"punpckhbw %%mm5, %%mm5\n\t"

		"punpcklbw %%mm6, %%mm6\n\t"

		"punpckhbw %%mm7, %%mm7\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm1, 8%0\n\t"

		MOVNTQ"	%%mm2, 16%0\n\t"

		MOVNTQ"	%%mm3, 24%0\n\t"

		MOVNTQ"	%%mm4, 32%0\n\t"

		MOVNTQ"	%%mm5, 40%0\n\t"

		MOVNTQ"	%%mm6, 48%0\n\t"

		MOVNTQ"	%%mm7, 56%0"

		:"=m"(d[2*x])

		:"m"(s1[x])

		:"memory");

	}

#endif

	for(;x<w;x++) d[2*x]=d[2*x+1]=s1[x];

    }

    for(y=0;y<h;y++){

	const uint8_t* s2=src2+srcStride2*(y>>1);

	uint8_t* d=dst2+dstStride2*y;

	x=0;

#ifdef HAVE_MMX

	for(;x<w-31;x+=32)

	{

	    asm volatile(

		PREFETCH" 32%1\n\t"

	        "movq	%1, %%mm0\n\t"

	        "movq	8%1, %%mm2\n\t"

	        "movq	16%1, %%mm4\n\t"

	        "movq	24%1, %%mm6\n\t"

	        "movq	%%mm0, %%mm1\n\t"

	        "movq	%%mm2, %%mm3\n\t"

	        "movq	%%mm4, %%mm5\n\t"

	        "movq	%%mm6, %%mm7\n\t"

		"punpcklbw %%mm0, %%mm0\n\t"

		"punpckhbw %%mm1, %%mm1\n\t"

		"punpcklbw %%mm2, %%mm2\n\t"

		"punpckhbw %%mm3, %%mm3\n\t"

		"punpcklbw %%mm4, %%mm4\n\t"

		"punpckhbw %%mm5, %%mm5\n\t"

		"punpcklbw %%mm6, %%mm6\n\t"

		"punpckhbw %%mm7, %%mm7\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm1, 8%0\n\t"

		MOVNTQ"	%%mm2, 16%0\n\t"

		MOVNTQ"	%%mm3, 24%0\n\t"

		MOVNTQ"	%%mm4, 32%0\n\t"

		MOVNTQ"	%%mm5, 40%0\n\t"

		MOVNTQ"	%%mm6, 48%0\n\t"

		MOVNTQ"	%%mm7, 56%0"

		:"=m"(d[2*x])

		:"m"(s2[x])

		:"memory");

	}

#endif

	for(;x<w;x++) d[2*x]=d[2*x+1]=s2[x];

    }

#ifdef HAVE_MMX

	asm(

		EMMS" \n\t"

		SFENCE" \n\t"

		::: "memory"

		);

#endif

}
