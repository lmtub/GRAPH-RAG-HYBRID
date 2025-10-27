static inline void RENAME(yuvPlanartoyuy2)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,

	long width, long height,

	long lumStride, long chromStride, long dstStride, long vertLumPerChroma)

{

	long y;

	const long chromWidth= width>>1;

	for(y=0; y<height; y++)

	{

#ifdef HAVE_MMX

//FIXME handle 2 lines a once (fewer prefetch, reuse some chrom, but very likely limited by mem anyway)

		asm volatile(

			"xor %%"REG_a", %%"REG_a"	\n\t"

			ASMALIGN(4)

			"1:				\n\t"

			PREFETCH" 32(%1, %%"REG_a", 2)	\n\t"

			PREFETCH" 32(%2, %%"REG_a")	\n\t"

			PREFETCH" 32(%3, %%"REG_a")	\n\t"

			"movq (%2, %%"REG_a"), %%mm0	\n\t" // U(0)

			"movq %%mm0, %%mm2		\n\t" // U(0)

			"movq (%3, %%"REG_a"), %%mm1	\n\t" // V(0)

			"punpcklbw %%mm1, %%mm0		\n\t" // UVUV UVUV(0)

			"punpckhbw %%mm1, %%mm2		\n\t" // UVUV UVUV(8)



			"movq (%1, %%"REG_a",2), %%mm3	\n\t" // Y(0)

			"movq 8(%1, %%"REG_a",2), %%mm5	\n\t" // Y(8)

			"movq %%mm3, %%mm4		\n\t" // Y(0)

			"movq %%mm5, %%mm6		\n\t" // Y(8)

			"punpcklbw %%mm0, %%mm3		\n\t" // YUYV YUYV(0)

			"punpckhbw %%mm0, %%mm4		\n\t" // YUYV YUYV(4)

			"punpcklbw %%mm2, %%mm5		\n\t" // YUYV YUYV(8)

			"punpckhbw %%mm2, %%mm6		\n\t" // YUYV YUYV(12)



			MOVNTQ" %%mm3, (%0, %%"REG_a", 4)\n\t"

			MOVNTQ" %%mm4, 8(%0, %%"REG_a", 4)\n\t"

			MOVNTQ" %%mm5, 16(%0, %%"REG_a", 4)\n\t"

			MOVNTQ" %%mm6, 24(%0, %%"REG_a", 4)\n\t"



			"add $8, %%"REG_a"		\n\t"

			"cmp %4, %%"REG_a"		\n\t"

			" jb 1b				\n\t"

			::"r"(dst), "r"(ysrc), "r"(usrc), "r"(vsrc), "g" (chromWidth)

			: "%"REG_a

		);

#else



#if defined ARCH_ALPHA && defined HAVE_MVI

#define pl2yuy2(n)					\

	y1 = yc[n];					\

	y2 = yc2[n];					\

	u = uc[n];					\

	v = vc[n];					\

	asm("unpkbw %1, %0" : "=r"(y1) : "r"(y1));	\

	asm("unpkbw %1, %0" : "=r"(y2) : "r"(y2));	\

	asm("unpkbl %1, %0" : "=r"(u) : "r"(u));	\

	asm("unpkbl %1, %0" : "=r"(v) : "r"(v));	\

	yuv1 = (u << 8) + (v << 24);			\

	yuv2 = yuv1 + y2;				\

	yuv1 += y1;					\

	qdst[n] = yuv1;					\

	qdst2[n] = yuv2;



		int i;

		uint64_t *qdst = (uint64_t *) dst;

		uint64_t *qdst2 = (uint64_t *) (dst + dstStride);

		const uint32_t *yc = (uint32_t *) ysrc;

		const uint32_t *yc2 = (uint32_t *) (ysrc + lumStride);

		const uint16_t *uc = (uint16_t*) usrc, *vc = (uint16_t*) vsrc;

		for(i = 0; i < chromWidth; i += 8){

			uint64_t y1, y2, yuv1, yuv2;

			uint64_t u, v;

			/* Prefetch */

			asm("ldq $31,64(%0)" :: "r"(yc));

			asm("ldq $31,64(%0)" :: "r"(yc2));

			asm("ldq $31,64(%0)" :: "r"(uc));

			asm("ldq $31,64(%0)" :: "r"(vc));



			pl2yuy2(0);

			pl2yuy2(1);

			pl2yuy2(2);

			pl2yuy2(3);



			yc += 4;

			yc2 += 4;

			uc += 4;

			vc += 4;

			qdst += 4;

			qdst2 += 4;

		}

		y++;

		ysrc += lumStride;

		dst += dstStride;



#elif __WORDSIZE >= 64

		int i;

		uint64_t *ldst = (uint64_t *) dst;

		const uint8_t *yc = ysrc, *uc = usrc, *vc = vsrc;

		for(i = 0; i < chromWidth; i += 2){

			uint64_t k, l;

			k = yc[0] + (uc[0] << 8) +

			    (yc[1] << 16) + (vc[0] << 24);

			l = yc[2] + (uc[1] << 8) +

			    (yc[3] << 16) + (vc[1] << 24);

			*ldst++ = k + (l << 32);

			yc += 4;

			uc += 2;

			vc += 2;

		}



#else

		int i, *idst = (int32_t *) dst;

		const uint8_t *yc = ysrc, *uc = usrc, *vc = vsrc;

		for(i = 0; i < chromWidth; i++){

#ifdef WORDS_BIGENDIAN

			*idst++ = (yc[0] << 24)+ (uc[0] << 16) +

			    (yc[1] << 8) + (vc[0] << 0);

#else

			*idst++ = yc[0] + (uc[0] << 8) +

			    (yc[1] << 16) + (vc[0] << 24);

#endif

			yc += 2;

			uc++;

			vc++;

		}

#endif

#endif

		if((y&(vertLumPerChroma-1))==(vertLumPerChroma-1) )

		{

			usrc += chromStride;

			vsrc += chromStride;

		}

		ysrc += lumStride;

		dst += dstStride;

	}

#ifdef HAVE_MMX

asm(    EMMS" \n\t"

        SFENCE" \n\t"

        :::"memory");

#endif

}
