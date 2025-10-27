static inline void RENAME(yuy2toyv12)(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

	long width, long height,

	long lumStride, long chromStride, long srcStride)

{

	long y;

	const long chromWidth= width>>1;

	for(y=0; y<height; y+=2)

	{

#ifdef HAVE_MMX

		asm volatile(

			"xor %%"REG_a", %%"REG_a"	\n\t"

			"pcmpeqw %%mm7, %%mm7		\n\t"

			"psrlw $8, %%mm7		\n\t" // FF,00,FF,00...

			ASMALIGN16

			"1:				\n\t"

			PREFETCH" 64(%0, %%"REG_a", 4)	\n\t"

			"movq (%0, %%"REG_a", 4), %%mm0	\n\t" // YUYV YUYV(0)

			"movq 8(%0, %%"REG_a", 4), %%mm1\n\t" // YUYV YUYV(4)

			"movq %%mm0, %%mm2		\n\t" // YUYV YUYV(0)

			"movq %%mm1, %%mm3		\n\t" // YUYV YUYV(4)

			"psrlw $8, %%mm0		\n\t" // U0V0 U0V0(0)

			"psrlw $8, %%mm1		\n\t" // U0V0 U0V0(4)

			"pand %%mm7, %%mm2		\n\t" // Y0Y0 Y0Y0(0)

			"pand %%mm7, %%mm3		\n\t" // Y0Y0 Y0Y0(4)

			"packuswb %%mm1, %%mm0		\n\t" // UVUV UVUV(0)

			"packuswb %%mm3, %%mm2		\n\t" // YYYY YYYY(0)



			MOVNTQ" %%mm2, (%1, %%"REG_a", 2)\n\t"



			"movq 16(%0, %%"REG_a", 4), %%mm1\n\t" // YUYV YUYV(8)

			"movq 24(%0, %%"REG_a", 4), %%mm2\n\t" // YUYV YUYV(12)

			"movq %%mm1, %%mm3		\n\t" // YUYV YUYV(8)

			"movq %%mm2, %%mm4		\n\t" // YUYV YUYV(12)

			"psrlw $8, %%mm1		\n\t" // U0V0 U0V0(8)

			"psrlw $8, %%mm2		\n\t" // U0V0 U0V0(12)

			"pand %%mm7, %%mm3		\n\t" // Y0Y0 Y0Y0(8)

			"pand %%mm7, %%mm4		\n\t" // Y0Y0 Y0Y0(12)

			"packuswb %%mm2, %%mm1		\n\t" // UVUV UVUV(8)

			"packuswb %%mm4, %%mm3		\n\t" // YYYY YYYY(8)



			MOVNTQ" %%mm3, 8(%1, %%"REG_a", 2)\n\t"



			"movq %%mm0, %%mm2		\n\t" // UVUV UVUV(0)

			"movq %%mm1, %%mm3		\n\t" // UVUV UVUV(8)

			"psrlw $8, %%mm0		\n\t" // V0V0 V0V0(0)

			"psrlw $8, %%mm1		\n\t" // V0V0 V0V0(8)

			"pand %%mm7, %%mm2		\n\t" // U0U0 U0U0(0)

			"pand %%mm7, %%mm3		\n\t" // U0U0 U0U0(8)

			"packuswb %%mm1, %%mm0		\n\t" // VVVV VVVV(0)

			"packuswb %%mm3, %%mm2		\n\t" // UUUU UUUU(0)



			MOVNTQ" %%mm0, (%3, %%"REG_a")	\n\t"

			MOVNTQ" %%mm2, (%2, %%"REG_a")	\n\t"



			"add $8, %%"REG_a"		\n\t"

			"cmp %4, %%"REG_a"		\n\t"

			" jb 1b				\n\t"

			::"r"(src), "r"(ydst), "r"(udst), "r"(vdst), "g" (chromWidth)

			: "memory", "%"REG_a

		);



		ydst += lumStride;

		src  += srcStride;



		asm volatile(

			"xor %%"REG_a", %%"REG_a"	\n\t"

			ASMALIGN16

			"1:				\n\t"

			PREFETCH" 64(%0, %%"REG_a", 4)	\n\t"

			"movq (%0, %%"REG_a", 4), %%mm0	\n\t" // YUYV YUYV(0)

			"movq 8(%0, %%"REG_a", 4), %%mm1\n\t" // YUYV YUYV(4)

			"movq 16(%0, %%"REG_a", 4), %%mm2\n\t" // YUYV YUYV(8)

			"movq 24(%0, %%"REG_a", 4), %%mm3\n\t" // YUYV YUYV(12)

			"pand %%mm7, %%mm0		\n\t" // Y0Y0 Y0Y0(0)

			"pand %%mm7, %%mm1		\n\t" // Y0Y0 Y0Y0(4)

			"pand %%mm7, %%mm2		\n\t" // Y0Y0 Y0Y0(8)

			"pand %%mm7, %%mm3		\n\t" // Y0Y0 Y0Y0(12)

			"packuswb %%mm1, %%mm0		\n\t" // YYYY YYYY(0)

			"packuswb %%mm3, %%mm2		\n\t" // YYYY YYYY(8)



			MOVNTQ" %%mm0, (%1, %%"REG_a", 2)\n\t"

			MOVNTQ" %%mm2, 8(%1, %%"REG_a", 2)\n\t"



			"add $8, %%"REG_a"		\n\t"

			"cmp %4, %%"REG_a"		\n\t"

			" jb 1b				\n\t"



			::"r"(src), "r"(ydst), "r"(udst), "r"(vdst), "g" (chromWidth)

			: "memory", "%"REG_a

		);

#else

		long i;

		for(i=0; i<chromWidth; i++)

		{

			ydst[2*i+0] 	= src[4*i+0];

			udst[i] 	= src[4*i+1];

			ydst[2*i+1] 	= src[4*i+2];

			vdst[i] 	= src[4*i+3];

		}

		ydst += lumStride;

		src  += srcStride;



		for(i=0; i<chromWidth; i++)

		{

			ydst[2*i+0] 	= src[4*i+0];

			ydst[2*i+1] 	= src[4*i+2];

		}

#endif

		udst += chromStride;

		vdst += chromStride;

		ydst += lumStride;

		src  += srcStride;

	}

#ifdef HAVE_MMX

asm volatile(   EMMS" \n\t"

        	SFENCE" \n\t"

        	:::"memory");

#endif

}
