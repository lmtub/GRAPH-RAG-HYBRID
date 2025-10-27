static inline void RENAME(hyscale)(uint16_t *dst, int dstWidth, uint8_t *src, int srcW, int xInc,

				   int flags, int canMMX2BeUsed, int16_t *hLumFilter,

				   int16_t *hLumFilterPos, int hLumFilterSize, void *funnyYCode, 

				   int srcFormat, uint8_t *formatConvBuffer)

{

    if(srcFormat==IMGFMT_YUY2)

    {

	RENAME(yuy2ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_BGR32)

    {

	RENAME(bgr32ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_BGR24)

    {

	RENAME(bgr24ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_BGR16)

    {

	RENAME(bgr16ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_BGR15)

    {

	RENAME(bgr15ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_RGB32)

    {

	RENAME(rgb32ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }

    else if(srcFormat==IMGFMT_RGB24)

    {

	RENAME(rgb24ToY)(formatConvBuffer, src, srcW);

	src= formatConvBuffer;

    }



#ifdef HAVE_MMX

	// use the new MMX scaler if th mmx2 cant be used (its faster than the x86asm one)

    if(!(flags&SWS_FAST_BILINEAR) || (!canMMX2BeUsed))

#else

    if(!(flags&SWS_FAST_BILINEAR))

#endif

    {

    	RENAME(hScale)(dst, dstWidth, src, srcW, xInc, hLumFilter, hLumFilterPos, hLumFilterSize);

    }

    else // Fast Bilinear upscale / crap downscale

    {

#ifdef ARCH_X86

#ifdef HAVE_MMX2

	int i;

	if(canMMX2BeUsed)

	{

		asm volatile(

			"pxor %%mm7, %%mm7		\n\t"

			"pxor %%mm2, %%mm2		\n\t" // 2*xalpha

			"movd %5, %%mm6			\n\t" // xInc&0xFFFF

			"punpcklwd %%mm6, %%mm6		\n\t"

			"punpcklwd %%mm6, %%mm6		\n\t"

			"movq %%mm6, %%mm2		\n\t"

			"psllq $16, %%mm2		\n\t"

			"paddw %%mm6, %%mm2		\n\t"

			"psllq $16, %%mm2		\n\t"

			"paddw %%mm6, %%mm2		\n\t"

			"psllq $16, %%mm2		\n\t" //0,t,2t,3t		t=xInc&0xFF

			"movq %%mm2, %%mm4		\n\t"

			"movd %4, %%mm6			\n\t" //(xInc*4)&0xFFFF

			"punpcklwd %%mm6, %%mm6		\n\t"

			"punpcklwd %%mm6, %%mm6		\n\t"

			"xorl %%eax, %%eax		\n\t" // i

			"movl %0, %%esi			\n\t" // src

			"movl %1, %%edi			\n\t" // buf1

			"movl %3, %%edx			\n\t" // (xInc*4)>>16

			"xorl %%ecx, %%ecx		\n\t"

			"xorl %%ebx, %%ebx		\n\t"

			"movw %4, %%bx			\n\t" // (xInc*4)&0xFFFF



#define FUNNY_Y_CODE \

			PREFETCH" 1024(%%esi)		\n\t"\

			PREFETCH" 1056(%%esi)		\n\t"\

			PREFETCH" 1088(%%esi)		\n\t"\

			"call *%6			\n\t"\

			"movq %%mm4, %%mm2		\n\t"\

			"xorl %%ecx, %%ecx		\n\t"



FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE

FUNNY_Y_CODE



			:: "m" (src), "m" (dst), "m" (dstWidth), "m" ((xInc*4)>>16),

			"m" ((xInc*4)&0xFFFF), "m" (xInc&0xFFFF), "m" (funnyYCode)

			: "%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi"

		);

		for(i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--) dst[i] = src[srcW-1]*128;

	}

	else

	{

#endif

	//NO MMX just normal asm ...

	asm volatile(

		"xorl %%eax, %%eax		\n\t" // i

		"xorl %%ebx, %%ebx		\n\t" // xx

		"xorl %%ecx, %%ecx		\n\t" // 2*xalpha

		".balign 16			\n\t"

		"1:				\n\t"

		"movzbl  (%0, %%ebx), %%edi	\n\t" //src[xx]

		"movzbl 1(%0, %%ebx), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"movl %1, %%edi			\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, (%%edi, %%eax, 2)	\n\t"

		"addw %4, %%cx			\n\t" //2*xalpha += xInc&0xFF

		"adcl %3, %%ebx			\n\t" //xx+= xInc>>8 + carry



		"movzbl (%0, %%ebx), %%edi	\n\t" //src[xx]

		"movzbl 1(%0, %%ebx), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"movl %1, %%edi			\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, 2(%%edi, %%eax, 2)	\n\t"

		"addw %4, %%cx			\n\t" //2*xalpha += xInc&0xFF

		"adcl %3, %%ebx			\n\t" //xx+= xInc>>8 + carry





		"addl $2, %%eax			\n\t"

		"cmpl %2, %%eax			\n\t"

		" jb 1b				\n\t"





		:: "r" (src), "m" (dst), "m" (dstWidth), "m" (xInc>>16), "m" (xInc&0xFFFF)

		: "%eax", "%ebx", "%ecx", "%edi", "%esi"

		);

#ifdef HAVE_MMX2

	} //if MMX2 cant be used

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
