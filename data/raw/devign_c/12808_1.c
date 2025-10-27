static inline void RENAME(hyscale)(uint16_t *dst, long dstWidth, uint8_t *src, int srcW, int xInc,

				   int flags, int canMMX2BeUsed, int16_t *hLumFilter,

				   int16_t *hLumFilterPos, int hLumFilterSize, void *funnyYCode,

				   int srcFormat, uint8_t *formatConvBuffer, int16_t *mmx2Filter,

				   int32_t *mmx2FilterPos, uint8_t *pal)

{

    if(srcFormat==PIX_FMT_YUYV422 || srcFormat==PIX_FMT_GRAY16BE)

    {

	RENAME(yuy2ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_UYVY422 || srcFormat==PIX_FMT_GRAY16LE)

    {

	RENAME(uyvyToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_RGB32)

    {

	RENAME(bgr32ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_BGR24)

    {

	RENAME(bgr24ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_BGR565)

    {

	RENAME(bgr16ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_BGR555)

    {

	RENAME(bgr15ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_BGR32)

    {

	RENAME(rgb32ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_RGB24)

    {

	RENAME(rgb24ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_RGB565)

    {

	RENAME(rgb16ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_RGB555)

    {

	RENAME(rgb15ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==PIX_FMT_RGB8 || srcFormat==PIX_FMT_BGR8 || srcFormat==PIX_FMT_PAL8 || srcFormat==PIX_FMT_BGR4_BYTE  || srcFormat==PIX_FMT_RGB4_BYTE)

    {

	RENAME(palToY)(formatConvBuffer, src, srcW, pal);

	src= formatConvBuffer;

    }



#ifdef HAVE_MMX

	// use the new MMX scaler if the mmx2 can't be used (its faster than the x86asm one)

    if(!(flags&SWS_FAST_BILINEAR) || (!canMMX2BeUsed))

#else

    if(!(flags&SWS_FAST_BILINEAR))

#endif

    {

    	RENAME(hScale)(dst, dstWidth, src, srcW, xInc, hLumFilter, hLumFilterPos, hLumFilterSize);

    }

    else // Fast Bilinear upscale / crap downscale

    {

#if defined(ARCH_X86)

#ifdef HAVE_MMX2

	int i;

#if defined(PIC)

	uint64_t ebxsave __attribute__((aligned(8)));

#endif

	if(canMMX2BeUsed)

	{

		asm volatile(

#if defined(PIC)

			"mov %%"REG_b", %5    \n\t"

#endif

			"pxor %%mm7, %%mm7		\n\t"

			"mov %0, %%"REG_c"		\n\t"

			"mov %1, %%"REG_D"		\n\t"

			"mov %2, %%"REG_d"		\n\t"

			"mov %3, %%"REG_b"		\n\t"

			"xor %%"REG_a", %%"REG_a"	\n\t" // i

			PREFETCH" (%%"REG_c")		\n\t"

			PREFETCH" 32(%%"REG_c")		\n\t"

			PREFETCH" 64(%%"REG_c")		\n\t"



#ifdef ARCH_X86_64



#define FUNNY_Y_CODE \

			"movl (%%"REG_b"), %%esi	\n\t"\

			"call *%4			\n\t"\

			"movl (%%"REG_b", %%"REG_a"), %%esi\n\t"\

			"add %%"REG_S", %%"REG_c"	\n\t"\

			"add %%"REG_a", %%"REG_D"	\n\t"\

			"xor %%"REG_a", %%"REG_a"	\n\t"\



#else



#define FUNNY_Y_CODE \

			"movl (%%"REG_b"), %%esi	\n\t"\

			"call *%4			\n\t"\

			"addl (%%"REG_b", %%"REG_a"), %%"REG_c"\n\t"\

			"add %%"REG_a", %%"REG_D"	\n\t"\

			"xor %%"REG_a", %%"REG_a"	\n\t"\



#endif



FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE



#if defined(PIC)

			"mov %5, %%"REG_b"    \n\t"

#endif

			:: "m" (src), "m" (dst), "m" (mmx2Filter), "m" (mmx2FilterPos),

			"m" (funnyYCode)

#if defined(PIC)

			,"m" (ebxsave)

#endif

			: "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S, "%"REG_D

#if !defined(PIC)

			,"%"REG_b

#endif

		);

		for(i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--) dst[i] = src[srcW-1]*128;

	}

	else

	{

#endif

	long xInc_shr16 = xInc >> 16;

	uint16_t xInc_mask = xInc & 0xffff;

	//NO MMX just normal asm ...

	asm volatile(

		"xor %%"REG_a", %%"REG_a"	\n\t" // i

		"xor %%"REG_d", %%"REG_d"	\n\t" // xx

		"xorl %%ecx, %%ecx		\n\t" // 2*xalpha

		ASMALIGN(4)

		"1:				\n\t"

		"movzbl  (%0, %%"REG_d"), %%edi	\n\t" //src[xx]

		"movzbl 1(%0, %%"REG_d"), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"mov %1, %%"REG_D"		\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, (%%"REG_D", %%"REG_a", 2)\n\t"

		"addw %4, %%cx			\n\t" //2*xalpha += xInc&0xFF

		"adc %3, %%"REG_d"		\n\t" //xx+= xInc>>8 + carry



		"movzbl (%0, %%"REG_d"), %%edi	\n\t" //src[xx]

		"movzbl 1(%0, %%"REG_d"), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"mov %1, %%"REG_D"		\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, 2(%%"REG_D", %%"REG_a", 2)\n\t"

		"addw %4, %%cx			\n\t" //2*xalpha += xInc&0xFF

		"adc %3, %%"REG_d"		\n\t" //xx+= xInc>>8 + carry





		"add $2, %%"REG_a"		\n\t"

		"cmp %2, %%"REG_a"		\n\t"

		" jb 1b				\n\t"





		:: "r" (src), "m" (dst), "m" (dstWidth), "m" (xInc_shr16), "m" (xInc_mask)

		: "%"REG_a, "%"REG_d, "%ecx", "%"REG_D, "%esi"

		);

#ifdef HAVE_MMX2

	} //if MMX2 can't be used

#endif

#else

	int i;

	unsigned int xpos=0;

	for(i=0;i<dstWidth;i++)

	{

		register unsigned int xx=xpos>>16;

		register unsigned int xalpha=(xpos&0xFFFF)>>9;

		dst[i]= (src[xx]<<7) + (src[xx+1] - src[xx])*xalpha;

		xpos+=xInc;

	}

#endif

    }

}
