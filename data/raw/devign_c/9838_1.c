static void initMMX2HScaler(int dstW, int xInc, uint8_t *funnyCode)

{

	uint8_t *fragment;

	int imm8OfPShufW1;

	int imm8OfPShufW2;

	int fragmentLength;



	int xpos, i;



	// create an optimized horizontal scaling routine



	//code fragment



	asm volatile(

		"jmp 9f				\n\t"

	// Begin

		"0:				\n\t"

		"movq (%%esi), %%mm0		\n\t" //FIXME Alignment

		"movq %%mm0, %%mm1		\n\t"

		"psrlq $8, %%mm0		\n\t"

		"punpcklbw %%mm7, %%mm1	\n\t"

		"movq %%mm2, %%mm3		\n\t"

		"punpcklbw %%mm7, %%mm0	\n\t"

		"addw %%bx, %%cx		\n\t" //2*xalpha += (4*lumXInc)&0xFFFF

		"pshufw $0xFF, %%mm1, %%mm1	\n\t"

		"1:				\n\t"

		"adcl %%edx, %%esi		\n\t" //xx+= (4*lumXInc)>>16 + carry

		"pshufw $0xFF, %%mm0, %%mm0	\n\t"

		"2:				\n\t"

		"psrlw $9, %%mm3		\n\t"

		"psubw %%mm1, %%mm0		\n\t"

		"pmullw %%mm3, %%mm0		\n\t"

		"paddw %%mm6, %%mm2		\n\t" // 2*alpha += xpos&0xFFFF

		"psllw $7, %%mm1		\n\t"

		"paddw %%mm1, %%mm0		\n\t"



		"movq %%mm0, (%%edi, %%eax)	\n\t"



		"addl $8, %%eax			\n\t"

	// End

		"9:				\n\t"

//		"int $3\n\t"

		"leal 0b, %0			\n\t"

		"leal 1b, %1			\n\t"

		"leal 2b, %2			\n\t"

		"decl %1			\n\t"

		"decl %2			\n\t"

		"subl %0, %1			\n\t"

		"subl %0, %2			\n\t"

		"leal 9b, %3			\n\t"

		"subl %0, %3			\n\t"

		:"=r" (fragment), "=r" (imm8OfPShufW1), "=r" (imm8OfPShufW2),

		"=r" (fragmentLength)

	);



	xpos= 0; //lumXInc/2 - 0x8000; // difference between pixel centers



	for(i=0; i<dstW/8; i++)

	{

		int xx=xpos>>16;



		if((i&3) == 0)

		{

			int a=0;

			int b=((xpos+xInc)>>16) - xx;

			int c=((xpos+xInc*2)>>16) - xx;

			int d=((xpos+xInc*3)>>16) - xx;



			memcpy(funnyCode + fragmentLength*i/4, fragment, fragmentLength);



			funnyCode[fragmentLength*i/4 + imm8OfPShufW1]=

			funnyCode[fragmentLength*i/4 + imm8OfPShufW2]=

				a | (b<<2) | (c<<4) | (d<<6);



			// if we dont need to read 8 bytes than dont :), reduces the chance of

			// crossing a cache line

			if(d<3) funnyCode[fragmentLength*i/4 + 1]= 0x6E;



			funnyCode[fragmentLength*(i+4)/4]= RET;

		}

		xpos+=xInc;

	}

}
